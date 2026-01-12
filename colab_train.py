# Google Colab'de çalıştır
# 1. Runtime > Change runtime type > GPU seç

# Kurulum
!pip install ultralytics

# Drive'a bağlan (dataset'i yüklemek için)
from google.colab import drive
drive.mount('/content/drive')

# Dataset zip'i Drive'a yükle, sonra aç
!unzip "/content/drive/MyDrive/child and adult annotation.v1i.yolov8.zip" -d /content/dataset

# data.yaml düzelt
import yaml

data_config = {
    'train': '/content/dataset/train/images',
    'val': '/content/dataset/valid/images',
    'test': '/content/dataset/test/images',
    'nc': 2,
    'names': ['adult', 'child']
}

with open('/content/dataset/data.yaml', 'w') as f:
    yaml.dump(data_config, f)

# Training
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/content/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='child_adult_model',
    patience=10,
    device=0  # GPU
)

# Best model'i Drive'a kaydet
!cp runs/detect/child_adult_model/weights/best.pt "/content/drive/MyDrive/child_adult_best.pt"

print("Bitti! Model: /content/drive/MyDrive/child_adult_best.pt")
