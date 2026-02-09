"""
Auto-labeling script - YOLOv26m ile person detect et
Tüm person'ları class 0 (people) olarak kaydet
Sonra manuel olarak child olanları 1 yap
"""

from ultralytics import YOLO
import cv2
import os
import shutil
from pathlib import Path

# Ayarlar
MODEL_NAME = "yolo26m.pt"  # YOLOv26m - otomatik indirilecek
BASE_DIR = "/Users/suleakarsu/Desktop/yolo_pool/dataset_new"
IMAGES_DIR = f"{BASE_DIR}/images"
LABELS_DIR = f"{BASE_DIR}/labels"
PREVIEW_DIR = f"{BASE_DIR}/preview"
SOURCE_IMAGES = "/Users/suleakarsu/Desktop/yolo_pool/extracted_frames"
CONFIDENCE_THRESHOLD = 0.25

# COCO'da person = class 0, biz de 0 olarak kaydedeceğiz (people)
PERSON_CLASS_COCO = 0  # COCO dataset'te person

# Preview için
COLOR = (0, 255, 0)  # Yeşil
CLASS_NAME = "people"

def auto_label():
    # Eski klasörü temizle
    if os.path.exists(BASE_DIR):
        print(f"Eski klasör siliniyor: {BASE_DIR}")
        shutil.rmtree(BASE_DIR)

    print(f"Model yükleniyor: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    images = sorted([f for f in os.listdir(SOURCE_IMAGES) if f.endswith(('.jpg', '.png'))])
    print(f"Toplam {len(images)} resim bulundu")
    print(f"Çıktı klasörü: {BASE_DIR}")
    print(f"  - images/  (orijinal)")
    print(f"  - labels/  (YOLO txt - tüm person → class 0)")
    print(f"  - preview/ (bbox çizilmiş)")
    print("-" * 50)

    labeled_count = 0
    total_detections = 0

    for i, img_name in enumerate(images):
        src_path = os.path.join(SOURCE_IMAGES, img_name)
        img = cv2.imread(src_path)
        h, w = img.shape[:2]

        # Sadece person class'ını detect et (COCO class 0)
        results = model(src_path, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=[PERSON_CLASS_COCO])

        label_name = Path(img_name).stem + ".txt"
        label_path = os.path.join(LABELS_DIR, label_name)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])

                    # YOLO format (normalized) - hepsini class 0 (people) olarak kaydet
                    xn, yn, wn, hn = box.xywhn[0].tolist()
                    detections.append(f"0 {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}")

                    # Bbox çiz (pixel coordinates)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR, 2)
                    label_text = f"{CLASS_NAME} {conf:.2f}"
                    cv2.putText(img, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)

        # Orijinal resmi kopyala
        img_dest = os.path.join(IMAGES_DIR, img_name)
        shutil.copy2(src_path, img_dest)

        # Label kaydet
        with open(label_path, 'w') as f:
            f.write('\n'.join(detections))

        # Preview kaydet
        preview_path = os.path.join(PREVIEW_DIR, img_name)
        cv2.imwrite(preview_path, img)

        if detections:
            labeled_count += 1
            total_detections += len(detections)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(images)}] İşlendi...")

    print("-" * 50)
    print(f"TAMAMLANDI!")
    print(f"- {len(images)} resim işlendi")
    print(f"- {labeled_count} resimde person bulundu")
    print(f"- Toplam {total_detections} person detection")
    print(f"\nÇıktı: {BASE_DIR}/")
    print(f"  ├── images/   ({len(images)} orijinal resim)")
    print(f"  ├── labels/   ({len(images)} YOLO txt - tümü class 0)")
    print(f"  └── preview/  ({len(images)} bbox çizilmiş)")
    print(f"\nSonraki adımlar:")
    print(f"1. preview/ → bbox'ları kontrol et")
    print(f"2. labels/*.txt → child olanları 0 → 1 yap")
    print(f"   (0 = adult, 1 = child)")

if __name__ == "__main__":
    auto_label()
