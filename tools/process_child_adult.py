#!/usr/bin/env python3
"""Child-Adult detection with warning for unattended children.

Shows a warning banner when a child is detected without any adult present.
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo_pool.utils import create_video_writer


CLASS_NAMES = {0: 'child', 1: 'adult'}


def process_video_with_child_warning(video_path, model_path, output_path, conf_threshold=0.25):
    """Process video with child-alone warning system."""
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = create_video_writer(output_path, fps, width, height)

    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"Output: {output_path}")
    print("-" * 50)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = frame.copy()

        child_count = 0
        adult_count = 0
        detections = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, cls_id, conf in zip(boxes, class_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = CLASS_NAMES.get(cls_id, 'unknown')

                if cls_id == 0:
                    child_count += 1
                elif cls_id == 1:
                    adult_count += 1

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'class_name': class_name,
                    'conf': conf
                })

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls_id = det['class_id']
            class_name = det['class_name']
            conf = det['conf']

            box_color = (0, 165, 255) if cls_id == 0 else (0, 200, 255)

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            label = f"{class_name} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Warning: child alone
        if child_count >= 1 and adult_count == 0:
            warning_text = "WARNING: CHILD ALONE!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(warning_text, font, 1.5, 3)

            banner_height = th + 40
            overlay_warning = annotated_frame.copy()
            cv2.rectangle(overlay_warning, (0, 0), (width, banner_height), (0, 0, 200), -1)
            cv2.addWeighted(overlay_warning, 0.7, annotated_frame, 0.3, 0, annotated_frame)

            text_x = (width - tw) // 2
            text_y = th + 20
            cv2.putText(annotated_frame, warning_text, (text_x + 2, text_y + 2), font, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, warning_text, (text_x, text_y), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

            if (frame_count // 5) % 2 == 0:
                cv2.rectangle(annotated_frame, (5, 5), (width - 5, height - 5), (0, 0, 255), 8)

        # Info overlay
        info_text = f"Child: {child_count} | Adult: {adult_count} | Frame: {frame_count}/{total_frames}"
        cv2.putText(annotated_frame, info_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, info_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(annotated_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})", flush=True)

    cap.release()
    out.release()
    print(f"\nDone! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Child-Adult detection with unattended child warning')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output video file (default: <input>_warning.mp4)')
    parser.add_argument('--model', '-m', type=str, default='best_pool_adult_child.pt',
                        help='Detection model (default: best_pool_adult_child.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        p = Path(args.input)
        output_path = str(p.parent / f"{p.stem}_warning{p.suffix}")

    process_video_with_child_warning(args.input, args.model, output_path, args.conf)


if __name__ == "__main__":
    main()
