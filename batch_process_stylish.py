"""
Stylish YOLOv8 Video Processor
Minimal, production-ready bounding boxes
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os


class StylishDetector:
    """Production-ready minimal bounding box style"""

    # Modern color palette (BGR)
    COLORS = {
        'person': (0, 200, 255),      # Vibrant orange-yellow
        'default': (255, 180, 100),   # Soft blue
    }

    def __init__(self, model_path='yolo26m.pt', conf_threshold=0.35):
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def draw_stylish_box(self, img, box, label, conf, color):
        """Draw elegant, visible bounding box with glow effect"""
        x1, y1, x2, y2 = map(int, box)

        # Semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

        # Glow effect (dark outline behind colored line)
        cv2.rectangle(img, (x1-1, y1-1), (x2+1, y2+1), (20, 20, 20), 3, cv2.LINE_AA)

        # Main rectangle - clean and visible
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label
        label_text = f"{label} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

        # Label background - pill style above box
        pad_x, pad_y = 8, 5
        lx1 = x1
        ly1 = y1 - th - pad_y * 2 - 2
        lx2 = x1 + tw + pad_x * 2
        ly2 = y1 - 2

        # Dark background with color accent
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (25, 25, 25), -1, cv2.LINE_AA)
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)

        # Label text
        cv2.putText(img, label_text, (lx1 + pad_x, ly2 - pad_y), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        return img

    def process_frame(self, frame):
        """Process single frame"""
        results = self.model(frame, conf=self.conf, classes=[0], verbose=False)

        annotated = frame.copy()

        if results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                color = self.COLORS['person']
                annotated = self.draw_stylish_box(annotated, box, 'person', conf, color)

        return annotated

    def process_video(self, input_path, output_path):
        """Process video file"""
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Cannot open {input_path}")
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output writer (H.264)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = self.process_frame(frame)
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
    input_dir = "/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi"
    output_dir = "/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi/outputs_yolo26m"

    os.makedirs(output_dir, exist_ok=True)

    # Get all videos
    videos = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])

    print(f"Found {len(videos)} videos")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Initialize detector
    detector = StylishDetector(model_path='yolov8m.pt', conf_threshold=0.35)

    for i, video in enumerate(videos, 1):
        input_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, f"output_{video}")

        print(f"\n[{i}/{len(videos)}] {video}")
        detector.process_video(input_path, output_path)

    print("\n" + "=" * 50)
    print(f"All done! Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
