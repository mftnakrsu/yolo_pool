"""
Videolardan frame çıkarma scripti - Object Detection dataset için
"""

import cv2
import os
from pathlib import Path

# Ayarlar
VIDEO_DIR = "/Users/suleakarsu/Desktop/yolo_pool/Risk Kademelendirmesi"
OUTPUT_DIR = "/Users/suleakarsu/Desktop/yolo_pool/extracted_frames"
FRAME_INTERVAL = 30  # Her kaç frame'de bir resim çıkarılsın (30 = yaklaşık saniyede 1)

def extract_frames(video_path, output_folder, frame_interval, image_counter):
    """Bir videodan belirli aralıklarla frame çıkarır"""

    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [HATA] Video açılamadı: {video_name}")
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
    # Output klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Video dosyalarını bul
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.MOV')
    videos = [
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if f.endswith(video_extensions)
    ]

    print(f"Toplam {len(videos)} video bulundu")
    print(f"Frame aralığı: Her {FRAME_INTERVAL} frame'de 1 resim")
    print(f"Çıktı klasörü: {OUTPUT_DIR}")
    print("-" * 50)

    total_saved = 0
    image_counter = 0  # image0, image1, image2, ...

    for i, video_path in enumerate(videos, 1):
        video_name = os.path.basename(video_path)
        print(f"[{i}/{len(videos)}] İşleniyor: {video_name[:50]}...")

        saved, image_counter = extract_frames(video_path, OUTPUT_DIR, FRAME_INTERVAL, image_counter)
        total_saved += saved
        print(f"  -> {saved} frame kaydedildi (image0 - image{image_counter-1})")

    print("-" * 50)
    print(f"TAMAMLANDI! Toplam {total_saved} frame çıkarıldı")
    print(f"Dosyalar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
