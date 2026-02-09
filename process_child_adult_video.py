"""
Child-Adult Detection with Warning for Alone Child
Class 0: Child
Class 1: Adult

If a child is detected without any adult present, show warning.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys


def process_video_with_child_warning(video_path, model_path, output_path, conf_threshold=0.25):
    """
    Video işleme - Yalnız çocuk uyarısı ile
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Çözünürlük: {width}x{height}, FPS: {fps}, Toplam Frame: {total_frames}")
    print(f"Çıktı: {output_path}")
    print("-" * 50)

    frame_count = 0

    # Class names (0: child, 1: adult)
    class_names = {0: 'child', 1: 'adult'}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
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
                class_name = class_names.get(cls_id, 'unknown')

                if cls_id == 0:  # Child
                    child_count += 1
                elif cls_id == 1:  # Adult
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

            # Colors based on class
            if cls_id == 0:  # Child
                box_color = (0, 165, 255)  # Orange
            else:  # Adult
                box_color = (0, 200, 255)  # Cyan

            # Semi-transparent fill
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)

            # Box outline
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Label
            label = f"{class_name} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Label background
            cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)

            # Label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Check for alone child warning
        # Condition: At least one child exists and no adults
        show_warning = child_count >= 1 and adult_count == 0

        if show_warning:
            # Draw warning banner at top
            warning_text = "UYARI: COCUK YALNIZ!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3

            (tw, th), baseline = cv2.getTextSize(warning_text, font, font_scale, font_thickness)

            # Red banner
            banner_height = th + 40
            overlay_warning = annotated_frame.copy()
            cv2.rectangle(overlay_warning, (0, 0), (width, banner_height), (0, 0, 200), -1)
            cv2.addWeighted(overlay_warning, 0.7, annotated_frame, 0.3, 0, annotated_frame)

            # Warning text centered
            text_x = (width - tw) // 2
            text_y = th + 20

            # White text with shadow
            cv2.putText(annotated_frame, warning_text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, warning_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Flashing border effect
            if (frame_count // 5) % 2 == 0:
                cv2.rectangle(annotated_frame, (5, 5), (width - 5, height - 5), (0, 0, 255), 8)

        # Info overlay at bottom
        info_text = f"Child: {child_count} | Adult: {adult_count} | Frame: {frame_count}/{total_frames}"
        cv2.putText(annotated_frame, info_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, info_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(annotated_frame)
        frame_count += 1

        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"İlerleme: {progress:.1f}% ({frame_count}/{total_frames})", flush=True)

    cap.release()
    out.release()
    print(f"\nTamamlandı! Çıktı: {output_path}")


def main():
    videos = [
        "/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi/baby_pool_demo6.mp4",
        "/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi/baby_pool_demo7.mp4",
        "/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi/baby_pool_demo9.mp4",
        "/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi/baby_pool_demo13.mp4",
    ]

    models = [
        ("/Users/suleakarsu/Desktop/yolo_pool/custom_best_pool.pt", "100ep"),
        ("/Users/suleakarsu/Desktop/yolo_pool/custom_best_pool_125.pt", "125ep"),
    ]

    output_dir = Path("/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi/outputs_custom")
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        video_name = Path(video_path).stem
        for model_path, model_suffix in models:
            output_filename = f"prediction_{model_suffix}_{video_name}.mp4"
            output_path = str(output_dir / output_filename)

            print("=" * 60)
            print(f"Video: {video_name} | Model: {model_suffix}")
            print("=" * 60)

            process_video_with_child_warning(
                video_path=video_path,
                model_path=model_path,
                output_path=output_path,
                conf_threshold=0.25
            )
            print("\n")


if __name__ == "__main__":
    main()
