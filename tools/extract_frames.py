#!/usr/bin/env python3
"""Extract frames from videos for dataset creation."""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_folder, frame_interval, image_counter):
    """Extract frames from a single video at given interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {Path(video_path).name}")
        return 0, image_counter

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"image{image_counter}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            image_counter += 1
        frame_count += 1

    cap.release()
    return saved_count, image_counter


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for dataset creation')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video file or directory of videos')
    parser.add_argument('--output', '-o', type=str, default='extracted_frames',
                        help='Output directory for frames (default: extracted_frames)')
    parser.add_argument('--interval', '-n', type=int, default=30,
                        help='Extract every N-th frame (default: 30, ~1 per second)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.MOV')

    if input_path.is_file():
        videos = [str(input_path)]
    elif input_path.is_dir():
        videos = [str(input_path / f) for f in os.listdir(input_path)
                  if f.endswith(video_extensions)]
    else:
        print(f"Error: {args.input} not found")
        return

    print(f"Found {len(videos)} video(s)")
    print(f"Frame interval: every {args.interval} frames")
    print(f"Output: {args.output}")
    print("-" * 50)

    total_saved = 0
    image_counter = 0

    for i, video_path in enumerate(videos, 1):
        video_name = os.path.basename(video_path)
        print(f"[{i}/{len(videos)}] {video_name[:50]}...")
        saved, image_counter = extract_frames(video_path, args.output, args.interval, image_counter)
        total_saved += saved
        print(f"  -> {saved} frames saved")

    print("-" * 50)
    print(f"Done! {total_saved} frames extracted to {args.output}")


if __name__ == "__main__":
    main()
