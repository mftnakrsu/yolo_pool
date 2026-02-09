#!/usr/bin/env python3
"""Pool Safety System - Adult/Child Detection + Pose Estimation.

Uses dual YOLO models:
  1. Custom YOLOv26m -> adult/child classification (bounding boxes)
  2. YOLOv8-pose   -> skeleton keypoints (17 COCO keypoints)

Drowning detection via movement analysis + head visibility check.
"""

import argparse
from pathlib import Path

from yolo_pool.detector import PoolPersonDetector


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
