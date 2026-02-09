#!/usr/bin/env python3
"""Video segmentation utility - split videos into fixed-duration chunks."""

import sys
import cv2
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo_pool.detector import PoolPersonDetector


def split_video(input_path, output_dir, segment_duration=6):
    """Split video into segments of given duration (seconds)."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_per_segment = fps * segment_duration
    segment_count = 0
    frame_count = 0
    out = None
    base_name = Path(input_path).stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_per_segment == 0:
            if out:
                out.release()
            segment_count += 1
            output_path = os.path.join(output_dir, f"{base_name}_part{segment_count}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Segment {segment_count}: {output_path}")

        out.write(frame)
        frame_count += 1

    if out:
        out.release()
    cap.release()
    print(f"Created {segment_count} segments")
    return segment_count


def process_and_split(input_path, model_path, output_dir, segment_duration=6):
    """Process video with detection and split into segments."""
    os.makedirs(output_dir, exist_ok=True)

    detector = PoolPersonDetector(model_path=model_path, conf_threshold=0.25)

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_per_segment = fps * segment_duration
    segment_count = 0
    frame_count = 0
    out = None

    print(f"Processing: {input_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_per_segment == 0:
            if out:
                out.release()
            segment_count += 1
            output_path = os.path.join(output_dir, f"output_part{segment_count}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output segment {segment_count}: {output_path}")

        _, annotated_frame = detector.detect_and_track(frame)
        out.write(annotated_frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count} processed...")

    if out:
        out.release()
    cap.release()
    print(f"Created {segment_count} output segments")


def main():
    parser = argparse.ArgumentParser(description='Split videos into segments')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video file')
    parser.add_argument('--output', '-o', type=str, default='segments',
                        help='Output directory (default: segments)')
    parser.add_argument('--duration', '-d', type=int, default=6,
                        help='Segment duration in seconds (default: 6)')
    parser.add_argument('--process', action='store_true',
                        help='Process video with detection before splitting')
    parser.add_argument('--model', '-m', type=str, default='best_pool_adult_child.pt',
                        help='Detection model (only with --process)')
    args = parser.parse_args()

    if args.process:
        process_and_split(args.input, args.model, args.output, args.duration)
    else:
        split_video(args.input, args.output, args.duration)


if __name__ == "__main__":
    main()
