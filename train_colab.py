# ============================================
# YOLO26m Pool Person Detection Training
# Google Colab - 2 class (adult / child)
# ============================================

# Cell 1: GPU kontrolü
# --------------------
!nvidia-smi

# Cell 2: Mount Drive & Install
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics unrar

# Cell 3: Dataset hazırlığı - Drive'dan kopyala ve aç
# ----------------------------------------------------
import os
import shutil
import zipfile
import subprocess
import random
from pathlib import Path

DRIVE_DIR = "/content/drive/MyDrive/pool_training"
WORK_DIR = "/content/pool_dataset"
RAW_IMAGES = f"{WORK_DIR}/raw/images"
RAW_LABELS = f"{WORK_DIR}/raw/labels"

# Temiz başla
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
os.makedirs(RAW_IMAGES, exist_ok=True)
os.makedirs(RAW_LABELS, exist_ok=True)

# 1) Image'ları kopyala
print("Image'lar kopyalanıyor...")
src_images = os.path.join(DRIVE_DIR, "images")
img_count = 0
for f in os.listdir(src_images):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        shutil.copy2(os.path.join(src_images, f), os.path.join(RAW_IMAGES, f))
        img_count += 1
print(f"  {img_count} image kopyalandı")

# 2) Label'ları aç - önce zip'i dene, yoksa rar'ı
print("Label'lar açılıyor...")

# Zip dosyasını bul
zip_files = [f for f in os.listdir(DRIVE_DIR) if f.endswith('.zip')]
rar_files = [f for f in os.listdir(DRIVE_DIR) if f.endswith('.rar')]

label_temp = f"{WORK_DIR}/label_temp"
os.makedirs(label_temp, exist_ok=True)

# Önce rar'ı aç (266 KB - daha büyük, muhtemelen ana label dosyası)
if rar_files:
    rar_path = os.path.join(DRIVE_DIR, rar_files[0])
    print(f"  RAR açılıyor: {rar_files[0]}")
    subprocess.run(["unrar", "x", "-o+", rar_path, label_temp], check=True)

# Zip'i de aç (üzerine yazabilir veya farklı dosyalar olabilir)
if zip_files:
    zip_path = os.path.join(DRIVE_DIR, zip_files[0])
    print(f"  ZIP açılıyor: {zip_files[0]}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(label_temp)

# Label txt dosyalarını bul (iç içe klasörlerde olabilir)
txt_count = 0
for root, dirs, files in os.walk(label_temp):
    for f in files:
        if f.endswith('.txt') and f != 'classes.txt':
            shutil.copy2(os.path.join(root, f), os.path.join(RAW_LABELS, f))
            txt_count += 1
        elif f == 'classes.txt':
            # classes.txt varsa oku - sınıf isimlerini göster
            with open(os.path.join(root, f)) as cf:
                classes = [line.strip() for line in cf.readlines() if line.strip()]
                print(f"  classes.txt bulundu: {classes}")

print(f"  {txt_count} label dosyası çıkarıldı")

# Temp klasörü temizle
shutil.rmtree(label_temp)

# Cell 4: Dataset analizi ve temizlik
# ------------------------------------
from collections import Counter

images = {Path(f).stem for f in os.listdir(RAW_IMAGES) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}
labels = {Path(f).stem for f in os.listdir(RAW_LABELS) if f.endswith('.txt')}

matched = images & labels
img_only = images - labels  # image var, label yok
lbl_only = labels - images  # label var, image yok

# Sınıf dağılımını analiz et
class_counts = Counter()
empty_labels = 0
total_boxes = 0

for stem in matched:
    label_path = os.path.join(RAW_LABELS, f"{stem}.txt")
    with open(label_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        empty_labels += 1
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            class_counts[int(parts[0])] += 1
            total_boxes += 1

print("=" * 50)
print("DATASET ANALİZİ")
print("=" * 50)
print(f"Toplam image:          {len(images)}")
print(f"Toplam label:          {len(labels)}")
print(f"Eşleşen:               {len(matched)}")
print(f"Label'sız image:       {len(img_only)}")
print(f"Image'sız label:       {len(lbl_only)}")
print(f"Boş label (bg):        {empty_labels}")
print(f"Toplam bbox:           {total_boxes}")
print(f"Sınıf dağılımı:        {dict(class_counts)}")
print("=" * 50)

# Label'sız image'lar için boş label oluştur (background olarak)
for stem in img_only:
    open(os.path.join(RAW_LABELS, f"{stem}.txt"), 'w').close()
    print(f"  Boş label oluşturuldu: {stem}")

# Cell 5: Train/Val/Test split (%80/%15/%5)
# ------------------------------------------
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05

DATASET_DIR = f"{WORK_DIR}/dataset"

for split in ['train', 'val', 'test']:
    os.makedirs(f"{DATASET_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/{split}/labels", exist_ok=True)

# Tüm eşleşen + label'sız image'lar dahil
all_stems = sorted(images)  # tüm image'lar dahil
random.seed(42)
random.shuffle(all_stems)

n = len(all_stems)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_stems = all_stems[:n_train]
val_stems = all_stems[n_train:n_train + n_val]
test_stems = all_stems[n_train + n_val:]

def copy_split(stems, split_name):
    for stem in stems:
        # Image'ı bul (uzantı farklı olabilir)
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(RAW_IMAGES, f"{stem}{ext}")
            if os.path.exists(candidate):
                img_file = candidate
                break
        if img_file:
            shutil.copy2(img_file, f"{DATASET_DIR}/{split_name}/images/")

        label_file = os.path.join(RAW_LABELS, f"{stem}.txt")
        if os.path.exists(label_file):
            shutil.copy2(label_file, f"{DATASET_DIR}/{split_name}/labels/")
        else:
            # Boş label oluştur
            open(f"{DATASET_DIR}/{split_name}/labels/{stem}.txt", 'w').close()

copy_split(train_stems, 'train')
copy_split(val_stems, 'val')
copy_split(test_stems, 'test')

print(f"Train: {len(train_stems)} image")
print(f"Val:   {len(val_stems)} image")
print(f"Test:  {len(test_stems)} image")
print(f"Toplam: {len(all_stems)} image")

# Cell 6: data.yaml oluştur
# --------------------------
yaml_content = f"""train: {DATASET_DIR}/train/images
val: {DATASET_DIR}/val/images
test: {DATASET_DIR}/test/images

nc: 2
names: ['adult', 'child']
"""

yaml_path = f"{DATASET_DIR}/data.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"data.yaml oluşturuldu: {yaml_path}")
print(yaml_content)

# Cell 7: Eğitim - YOLOv26m
# --------------------------
from ultralytics import YOLO

model = YOLO('yolo26m.pt')

results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    save=True,
    project='/content/runs',
    name='pool_adult_child',
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)

# Cell 8: Sonuçları değerlendir
# -----------------------------
metrics = model.val()
print(f"\nmAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# Sınıf bazlı sonuçlar
for i, name in enumerate(['adult', 'child']):
    if i < len(metrics.box.ap50):
        print(f"  {name}: mAP50={metrics.box.ap50[i]:.4f}")

# Cell 9: Confusion matrix ve grafikler
# --------------------------------------
# Eğitim sonuçları otomatik kaydedilir:
from IPython.display import Image, display

results_dir = '/content/runs/pool_adult_child'
for img_name in ['confusion_matrix.png', 'results.png', 'val_batch0_pred.jpg']:
    img_path = f"{results_dir}/{img_name}"
    if os.path.exists(img_path):
        print(f"\n{img_name}:")
        display(Image(filename=img_path, width=800))

# Cell 10: Model kaydet - Drive'a
# --------------------------------
best_model = '/content/runs/pool_adult_child/weights/best.pt'
last_model = '/content/runs/pool_adult_child/weights/last.pt'

save_dir = "/content/drive/MyDrive/pool_training/models"
os.makedirs(save_dir, exist_ok=True)

shutil.copy2(best_model, f"{save_dir}/best_pool_adult_child.pt")
shutil.copy2(last_model, f"{save_dir}/last_pool_adult_child.pt")

print(f"Modeller kaydedildi: {save_dir}/")
print(f"  - best_pool_adult_child.pt")
print(f"  - last_pool_adult_child.pt")

# Cell 11: Hızlı test - birkaç image'da dene
# --------------------------------------------
import cv2
import matplotlib.pyplot as plt

test_model = YOLO(best_model)
test_images = os.listdir(f"{DATASET_DIR}/test/images")[:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, img_name in enumerate(test_images):
    img_path = f"{DATASET_DIR}/test/images/{img_name}"
    results = test_model(img_path, conf=0.25)
    annotated = results[0].plot()
    axes[i].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[i].set_title(img_name, fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(f"{save_dir}/test_predictions.png", dpi=150)
plt.show()
print("Test tahminleri kaydedildi!")
