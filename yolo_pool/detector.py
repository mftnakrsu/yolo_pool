"""Core detection module - PoolPersonDetector with dual-model architecture."""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import time

from yolo_pool.utils import compute_iou, create_video_writer
from yolo_pool.visualization import draw_skeleton, draw_bbox, draw_status


class PoolPersonDetector:
    """Dual-model pool safety detector: custom detection + pose estimation.

    Uses a custom YOLOv26m model for adult/child classification and
    YOLOv8-pose for skeleton keypoints. Tracks persons across frames
    and analyzes movement patterns for drowning detection.
    """

    def __init__(self, model_path='best_pool_adult_child.pt',
                 pose_model_path='yolov8n-pose.pt',
                 conf_threshold=0.25):
        self.det_model = YOLO(model_path)
        self.pose_model = YOLO(pose_model_path)
        self.conf_threshold = conf_threshold

        # Tracking history (per person ID)
        self.track_history = defaultdict(lambda: {
            'positions': [],
            'last_seen': 0,
            'danger_score': 0,
        })

        # Timing parameters
        self.fps = 30
        self.stationary_threshold_seconds = 5.0
        self.drowning_threshold_seconds = 10.0
        self.movement_threshold = 20  # pixels

    def detect_and_track(self, image):
        """Run dual-model detection on a single frame.

        Returns:
            Tuple of (det_results, annotated_image).
        """
        # 1) Custom model: adult/child detection with tracking
        det_results = self.det_model.track(
            image, persist=True, conf=self.conf_threshold, verbose=False
        )

        # 2) Pose model: skeleton keypoints
        pose_results = self.pose_model(
            image, conf=self.conf_threshold, verbose=False
        )

        annotated = image.copy()
        current_time = time.time()

        # Extract pose boxes & keypoints
        pose_boxes = []
        pose_keypoints = []
        if pose_results[0].boxes is not None and pose_results[0].keypoints is not None:
            pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
            pose_keypoints = pose_results[0].keypoints.data.cpu()

        # Process detection results
        det = det_results[0]
        if det.boxes and det.boxes.id is not None:
            boxes = det.boxes.xywh.cpu()
            xyxy_boxes = det.boxes.xyxy.cpu().numpy()
            track_ids = det.boxes.id.int().cpu().tolist()
            class_ids = det.boxes.cls.int().cpu().tolist()
            confs = det.boxes.conf.cpu().tolist()

            for i, (box, track_id, cls_id, conf) in enumerate(
                zip(boxes, track_ids, class_ids, confs)
            ):
                x, y, w, h = box
                center = (float(x), float(y))
                class_name = self.det_model.names[cls_id]

                # Match detection box to pose keypoints via IoU
                kpts = self._match_pose(xyxy_boxes[i], pose_boxes, pose_keypoints)

                # Update tracking history
                history = self.track_history[track_id]
                history['last_seen'] = current_time
                history['positions'].append(center)

                max_history = int(self.fps * 15)
                if len(history['positions']) > max_history:
                    history['positions'] = history['positions'][-max_history:]

                # Draw skeleton
                if kpts is not None:
                    draw_skeleton(annotated, kpts)

                # Analyze danger status
                status, color = self._analyze_status(track_id, kpts)

                # Draw bounding box and status
                draw_bbox(annotated, box, class_name, conf)
                draw_status(annotated, box, status, color)

        return det_results, annotated

    def _match_pose(self, det_box, pose_boxes, pose_keypoints):
        """Match a detection bbox to the closest pose bbox using IoU."""
        if len(pose_boxes) == 0:
            return None

        best_iou = 0.0
        best_idx = -1

        for j, pbox in enumerate(pose_boxes):
            iou = compute_iou(det_box, pbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou > 0.3 and best_idx >= 0:
            return pose_keypoints[best_idx]
        return None

    def _analyze_status(self, track_id, keypoints):
        """Analyze person status based on movement and head visibility."""
        history = self.track_history[track_id]
        positions = history['positions']

        min_frames = int(self.fps * 2)
        if len(positions) < min_frames:
            return "Analyzing...", (255, 255, 0)

        # Movement analysis
        recent = positions[-min_frames:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        movement = np.sqrt(dx * dx + dy * dy)
        is_stationary = movement < self.movement_threshold

        # Head visibility (keypoints 0-4: nose, eyes, ears)
        head_visible = True
        if keypoints is not None:
            head_visible = False
            for idx in range(5):
                if idx < len(keypoints) and keypoints[idx][2] > 0.5:
                    head_visible = True
                    break

        # Update danger score
        if is_stationary:
            history['danger_score'] += 1
        else:
            history['danger_score'] = max(0, history['danger_score'] - 2)

        score = history['danger_score']

        # Determine status
        if score > self.drowning_threshold_seconds * self.fps:
            if not head_visible:
                return "DROWNING ALERT!", (0, 0, 255)      # red
            return "STATIONARY (Danger)", (0, 165, 255)     # orange
        elif score > self.stationary_threshold_seconds * self.fps:
            return "Stationary", (0, 255, 255)              # yellow
        return "Active", (0, 255, 0)                        # green

    def process_video(self, video_path, output_path=None, show_preview=True):
        """Process video file with dual-model detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return

        self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if output_path:
            out = create_video_writer(output_path, self.fps, width, height)

        print(f"Processing: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {self.fps}, Frames: {total_frames}")
        print("Press 'q' to stop.")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, annotated = self.detect_and_track(frame)

            # Frame counter
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if out:
                out.write(annotated)

            if show_preview:
                cv2.imshow('Pool Safety Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user")
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  [{frame_count}/{total_frames}] frames processed")

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Done! Total frames: {frame_count}")
        if output_path:
            print(f"Output saved: {output_path}")

    def process_image(self, image_path, output_path=None, show_preview=True):
        """Process a single image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot open image: {image_path}")
            return

        _, annotated = self.detect_and_track(image)

        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Saved: {output_path}")

        if show_preview:
            cv2.imshow('Detection Result', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def process_webcam(self, camera_index=0):
        """Real-time webcam detection."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        self.fps = 30
        print("Webcam started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, annotated = self.detect_and_track(frame)

            cv2.imshow('Pool Safety - Live', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
