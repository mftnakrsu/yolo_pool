#!/usr/bin/env python3
"""Auto-labeling script - Detect persons with YOLO and generate YOLO-format labels.

All persons are labeled as class 0 (adult) by default.
Manually reclassify children as class 1 afterwards.
"""

from ultralytics import YOLO
import cv2
import os
import shutil
import argparse
from pathlib import Path

PERSON_CLASS_COCO = 0
COLOR = (0, 255, 0)
CLASS_NAME = "people"


def auto_label(source_dir, output_dir, model_name, conf_threshold):
    if os.path.exists(output_dir):
        print(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    preview_dir = os.path.join(output_dir, "preview")

    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(images)} images in {source_dir}")
    print(f"Output: {output_dir}")
    print("-" * 50)

    labeled_count = 0
    total_detections = 0

    for i, img_name in enumerate(images):
        src_path = os.path.join(source_dir, img_name)
        img = cv2.imread(src_path)
        if img is None:
            continue

        results = model(src_path, verbose=False, conf=conf_threshold, classes=[PERSON_CLASS_COCO])

        label_name = Path(img_name).stem + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    xn, yn, wn, hn = box.xywhn[0].tolist()
                    detections.append(f"0 {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}")

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR, 2)
                    label_text = f"{CLASS_NAME} {conf:.2f}"
                    cv2.putText(img, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)

        shutil.copy2(src_path, os.path.join(images_dir, img_name))

        with open(label_path, 'w') as f:
            f.write('\n'.join(detections))

        cv2.imwrite(os.path.join(preview_dir, img_name), img)

        if detections:
            labeled_count += 1
            total_detections += len(detections)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(images)}] processed...")

    print("-" * 50)
    print(f"Done! {len(images)} images processed")
    print(f"  {labeled_count} images with detections")
    print(f"  {total_detections} total detections")
    print(f"\nNext steps:")
    print(f"  1. Check preview/ for bounding boxes")
    print(f"  2. In labels/*.txt, change 0 -> 1 for children")


def main():
    parser = argparse.ArgumentParser(description='Auto-label images with YOLO person detection')
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Source directory with images to label')
    parser.add_argument('--output', '-o', type=str, default='dataset',
                        help='Output directory (default: dataset)')
    parser.add_argument('--model', '-m', type=str, default='yolo26m.pt',
                        help='YOLO model for detection (default: yolo26m.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    args = parser.parse_args()

    auto_label(args.source, args.output, args.model, args.conf)


if __name__ == "__main__":
    main()
