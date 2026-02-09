#!/usr/bin/env python3
"""Batch video processing with stylish bounding boxes."""

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo_pool.utils import create_video_writer


COLORS = {
    'person': (0, 200, 255),
    'default': (255, 180, 100),
}


def draw_stylish_box(img, box, label, conf, color):
    """Draw elegant bounding box with glow effect."""
    x1, y1, x2, y2 = map(int, box)

    # Semi-transparent fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    # Glow effect
    cv2.rectangle(img, (x1-1, y1-1), (x2+1, y2+1), (20, 20, 20), 3, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label
    label_text = f"{label} {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label_text, font, 0.5, 1)

    pad_x, pad_y = 8, 5
    lx1, ly1 = x1, y1 - th - pad_y * 2 - 2
    lx2, ly2 = x1 + tw + pad_x * 2, y1 - 2

    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (25, 25, 25), -1, cv2.LINE_AA)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)
    cv2.putText(img, label_text, (lx1 + pad_x, ly2 - pad_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def process_video(model, input_path, output_path, conf):
    """Process a single video file."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = create_video_writer(output_path, fps, width, height)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, classes=[0], verbose=False)
        annotated = frame.copy()

        if results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, c in zip(boxes, confs):
                annotated = draw_stylish_box(annotated, box, 'person', c, COLORS['person'])

        out.write(annotated)
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\r  Progress: {progress:.0f}% ({frame_count}/{total_frames})", end="", flush=True)

    cap.release()
    out.release()
    print(f"\r  Done: {frame_count} frames")
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch video processing with stylish bounding boxes')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory with video files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: <input>/outputs)')
    parser.add_argument('--model', '-m', type=str, default='yolov8m.pt',
                        help='YOLO model (default: yolov8m.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.35,
                        help='Confidence threshold (default: 0.35)')
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.input, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    videos = sorted([f for f in os.listdir(args.input) if f.endswith('.mp4')])
    print(f"Found {len(videos)} videos")
    print(f"Output: {output_dir}")
    print("-" * 50)

    model = YOLO(args.model)

    for i, video in enumerate(videos, 1):
        input_path = os.path.join(args.input, video)
        output_path = os.path.join(output_dir, f"output_{video}")
        print(f"\n[{i}/{len(videos)}] {video}")
        process_video(model, input_path, output_path, args.conf)

    print(f"\nAll done! Outputs: {output_dir}")


if __name__ == "__main__":
    main()
