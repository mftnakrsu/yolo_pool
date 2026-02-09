"""
Pool Safety System - Adult/Child Detection + Pose Estimation
Uses dual YOLO models:
  1. Custom YOLOv26m -> adult/child classification (bounding boxes)
  2. YOLOv8-pose   -> skeleton keypoints (17 COCO keypoints)

Drowning detection via movement analysis + head visibility check.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
from collections import defaultdict
import time


class PoolPersonDetector:
    """Dual-model pool safety detector: custom detection + pose estimation"""

    # COCO skeleton connections (1-indexed pairs)
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
        [6, 12], [7, 13],   # torso-legs
        [6, 7],              # shoulders
        [6, 8], [7, 9],     # upper arms
        [8, 10], [9, 11],   # lower arms
        [2, 3], [1, 2], [1, 3],  # face
        [2, 4], [3, 5],     # ears
        [4, 6], [5, 7]      # ear-shoulder
    ]

    # Left/right keypoint indices for coloring
    LEFT_INDICES = {1, 3, 5, 7, 9, 11, 13, 15}
    RIGHT_INDICES = {2, 4, 6, 8, 10, 12, 14, 16}

    def __init__(self, model_path='best_pool_adult_child.pt',
                 pose_model_path='yolov8n-pose.pt',
                 conf_threshold=0.25):
        """
        Args:
            model_path: Custom detection model (adult/child)
            pose_model_path: Pose estimation model (skeleton)
            conf_threshold: Detection confidence threshold
        """
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
        """
        Run dual-model detection on a single frame.
        Returns (results, annotated_image).
        """
        # 1) Custom model: adult/child detection with tracking
        det_results = self.det_model.track(
            image, persist=True, conf=self.conf_threshold, verbose=False
        )

        # 2) Pose model: skeleton keypoints (no tracking needed)
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
                    self._draw_keypoints(annotated, kpts)

                # Analyze danger status
                status, color = self._analyze_status(track_id, kpts)

                # Draw bounding box and label
                self._draw_status(annotated, box, status, color, track_id, class_name, conf)

        return det_results, annotated

    def _match_pose(self, det_box, pose_boxes, pose_keypoints):
        """Match a detection bbox to the closest pose bbox using IoU."""
        if len(pose_boxes) == 0:
            return None

        best_iou = 0.0
        best_idx = -1

        for j, pbox in enumerate(pose_boxes):
            iou = self._compute_iou(det_box, pbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou > 0.3 and best_idx >= 0:
            return pose_keypoints[best_idx]
        return None

    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def _draw_keypoints(self, image, keypoints):
        """Draw pose skeleton on image using COCO 17 keypoint format."""
        kpt_color = (0, 255, 255)     # yellow (center)
        left_color = (255, 128, 0)    # orange (left side)
        right_color = (51, 153, 255)  # blue (right side)

        # Draw keypoint dots
        for idx, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if conf > 0.5:
                if idx in self.LEFT_INDICES:
                    color = left_color
                elif idx in self.RIGHT_INDICES:
                    color = right_color
                else:
                    color = kpt_color
                cv2.circle(image, (int(x), int(y)), 3, color, -1)
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 0), 1)

        # Draw skeleton lines
        for idx1, idx2 in self.SKELETON:
            idx1 -= 1  # convert to 0-indexed
            idx2 -= 1
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
                if kpt1[2] > 0.5 and kpt2[2] > 0.5:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

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
                return "DROWNING ALERT!", (0, 0, 255)     # red
            return "STATIONARY (Danger)", (0, 165, 255)    # orange
        elif score > self.stationary_threshold_seconds * self.fps:
            return "Stationary", (0, 255, 255)             # yellow
        return "Active", (0, 255, 0)                       # green

    def _draw_status(self, image, box, status, color, track_id, class_name="", conf=0.0):
        """Draw bounding box, label, and status for a detected person."""
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # Color by class
        if class_name == 'child':
            box_color = (255, 147, 0)   # orange
        elif class_name == 'adult':
            box_color = (0, 200, 255)   # cyan
        else:
            box_color = (147, 20, 255)  # purple

        # Semi-transparent bbox fill
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

        # Bbox border
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)

        # Label
        label = f"{class_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        pad_x, pad_y = 6, 4
        lx1 = x1
        ly1 = y1 - th - pad_y * 2
        lx2 = x1 + tw + pad_x * 2
        ly2 = y1

        # Rounded label background
        r = 5
        overlay2 = image.copy()
        cv2.rectangle(overlay2, (lx1 + r, ly1), (lx2 - r, ly2), box_color, -1)
        cv2.rectangle(overlay2, (lx1, ly1 + r), (lx2, ly2 - r), box_color, -1)
        cv2.circle(overlay2, (lx1 + r, ly1 + r), r, box_color, -1)
        cv2.circle(overlay2, (lx2 - r, ly1 + r), r, box_color, -1)
        cv2.circle(overlay2, (lx1 + r, ly2 - r), r, box_color, -1)
        cv2.circle(overlay2, (lx2 - r, ly2 - r), r, box_color, -1)
        cv2.addWeighted(overlay2, 0.85, image, 0.15, 0, image)

        # Label text with shadow
        tx = lx1 + pad_x
        ty = ly2 - pad_y
        cv2.putText(image, label, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(image, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Status text below bbox
        if status and status != "Active":
            status_label = f"{status}"
            (sw, sh), _ = cv2.getTextSize(status_label, font, 0.4, 1)
            sx = x1
            sy = y2 + sh + 6
            cv2.putText(image, status_label, (sx + 1, sy + 1), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, status_label, (sx, sy), font, 0.4, color, 1, cv2.LINE_AA)

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
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            except Exception:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

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


def main():
    parser = argparse.ArgumentParser(
        description='Pool Safety System - Adult/Child Detection + Pose Estimation'
    )
    parser.add_argument('--input', '-i', type=str,
                        help='Input file (image or video)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path')
    parser.add_argument('--model', '-m', type=str, default='best_pool_adult_child.pt',
                        help='Custom detection model (default: best_pool_adult_child.pt)')
    parser.add_argument('--pose-model', '-p', type=str, default='yolov8n-pose.pt',
                        help='Pose estimation model (default: yolov8n-pose.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--webcam', '-w', action='store_true',
                        help='Use webcam')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable live preview')

    args = parser.parse_args()

    detector = PoolPersonDetector(
        model_path=args.model,
        pose_model_path=args.pose_model,
        conf_threshold=args.conf
    )

    if args.webcam:
        detector.process_webcam()
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {args.input}")
            return

        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            detector.process_image(args.input, args.output,
                                   show_preview=not args.no_preview)
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            detector.process_video(args.input, args.output,
                                   show_preview=not args.no_preview)
        else:
            print(f"Error: Unsupported format: {input_path.suffix}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
