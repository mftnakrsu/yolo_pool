"""
Real-time Pool Safety Detection
Dual model: Custom YOLOv26m (adult/child) + Pose estimation (skeleton)
Webcam with live FPS, confidence control, and screenshot capture.
"""

import cv2
from ultralytics import YOLO
import argparse
import time

# Default skeleton connections (COCO 17 keypoints, 1-indexed)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
]

LEFT_INDICES = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_INDICES = {2, 4, 6, 8, 10, 12, 14, 16}

# Class colors (BGR)
CLASS_COLORS = {
    'adult': (0, 200, 255),   # cyan
    'child': (255, 147, 0),   # orange
}


def compute_iou(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def draw_skeleton(frame, keypoints):
    """Draw pose skeleton on frame."""
    kpt_color = (0, 255, 255)
    left_color = (255, 128, 0)
    right_color = (51, 153, 255)

    for idx, kpt in enumerate(keypoints):
        x, y, conf = kpt
        if conf > 0.5:
            if idx in LEFT_INDICES:
                c = left_color
            elif idx in RIGHT_INDICES:
                c = right_color
            else:
                c = kpt_color
            cv2.circle(frame, (int(x), int(y)), 3, c, -1)
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 0), 1)

    for i1, i2 in SKELETON:
        i1 -= 1
        i2 -= 1
        if i1 < len(keypoints) and i2 < len(keypoints):
            k1, k2 = keypoints[i1], keypoints[i2]
            if k1[2] > 0.5 and k2[2] > 0.5:
                cv2.line(frame, (int(k1[0]), int(k1[1])),
                         (int(k2[0]), int(k2[1])), (0, 255, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='Real-time Pool Safety Detection')
    parser.add_argument('--model', '-m', type=str, default='best_pool_adult_child.pt',
                        help='Custom detection model (default: best_pool_adult_child.pt)')
    parser.add_argument('--pose-model', '-p', type=str, default='yolov8n-pose.pt',
                        help='Pose estimation model (default: yolov8n-pose.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280, help='Frame width')
    parser.add_argument('--height', type=int, default=720, help='Frame height')
    args = parser.parse_args()

    # Load models
    print(f"Loading detection model: {args.model}")
    det_model = YOLO(args.model)
    print(f"  Classes: {det_model.names}")

    print(f"Loading pose model: {args.pose_model}")
    pose_model = YOLO(args.pose_model)

    # Open camera
    print(f"Opening camera (index: {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        print("  - Check if camera is connected")
        print("  - Try a different --camera index (0, 1, 2)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera ready: {actual_w}x{actual_h}")
    print("-" * 40)
    print("Controls:")
    print("  q     - Quit")
    print("  s     - Save screenshot")
    print("  +/-   - Adjust confidence")
    print("-" * 40)

    conf_threshold = args.conf
    frame_count = 0
    start_time = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Custom model: adult/child detection
        det_results = det_model(frame, verbose=False, conf=conf_threshold)

        # 2) Pose model: skeleton keypoints
        pose_results = pose_model(frame, verbose=False, conf=conf_threshold)

        # Extract pose data
        pose_boxes = []
        pose_kpts = []
        if pose_results[0].boxes is not None and pose_results[0].keypoints is not None:
            pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
            pose_kpts = pose_results[0].keypoints.data.cpu()

        # Process detections
        adult_count = 0
        child_count = 0

        for result in det_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = det_model.names[cls]

                if class_name == 'adult':
                    adult_count += 1
                elif class_name == 'child':
                    child_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = CLASS_COLORS.get(class_name, (255, 255, 255))

                # Match to pose and draw skeleton
                det_xyxy = box.xyxy[0].cpu().numpy()
                best_iou = 0.0
                best_idx = -1
                for j, pbox in enumerate(pose_boxes):
                    iou = compute_iou(det_xyxy, pbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_iou > 0.3 and best_idx >= 0:
                    draw_skeleton(frame, pose_kpts[best_idx])

                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label with pill background
                label = f"{class_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Info panel
        cv2.rectangle(frame, (0, 0), (260, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {conf_threshold:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Adult: {adult_count}  Child: {child_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Pool Safety Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key in (ord('+'), ord('=')):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"Confidence: {conf_threshold:.2f}")
        elif key == ord('-'):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"Confidence: {conf_threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
