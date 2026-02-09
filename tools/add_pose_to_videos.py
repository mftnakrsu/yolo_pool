#!/usr/bin/env python3
"""Add YOLOv8 pose estimation overlay to existing videos."""

import sys
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo_pool.visualization import draw_skeleton
from yolo_pool.utils import create_video_writer


def process_video(pose_model, input_path, output_path=None, conf_threshold=0.25, show_preview=False):
    """Add pose estimation overlay to a video file."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_pose{p.suffix}")

    out = create_video_writer(str(output_path), fps, width, height)

    print(f"\nProcessing: {Path(input_path).name}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame, conf=conf_threshold, verbose=False)
        annotated = frame.copy()

        if results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            for person_kpts in keypoints_data:
                draw_skeleton(annotated, person_kpts)

        out.write(annotated)
        frame_count += 1

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')

        if show_preview:
            cv2.imshow('Pose Estimation', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n  Done! Output: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Add YOLOv8 pose estimation to videos')
    parser.add_argument('--input', '-i', type=str, help='Single video file')
    parser.add_argument('--folder', '-f', type=str, help='Folder of videos')
    parser.add_argument('--output', '-o', type=str, help='Output file/folder')
    parser.add_argument('--model', '-m', type=str, default='yolov8n-pose.pt',
                        help='YOLO pose model (default: yolov8n-pose.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--preview', action='store_true', help='Show preview')
    args = parser.parse_args()

    pose_model = YOLO(args.model)

    if args.folder:
        folder = Path(args.folder)
        videos = list(folder.glob('*.mp4')) + list(folder.glob('*.avi')) + list(folder.glob('*.mov'))
        print(f"Found {len(videos)} videos")

        for video_path in videos:
            if '_pose' in video_path.stem:
                print(f"Skipping (already processed): {video_path.name}")
                continue
            process_video(pose_model, video_path, conf_threshold=args.conf, show_preview=args.preview)
    elif args.input:
        process_video(pose_model, args.input, args.output, conf_threshold=args.conf, show_preview=args.preview)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
