"""
YOLOv8 Fine-tuning: Child & Adult Detection
"""

from ultralytics import YOLO

# Pretrained model yükle
model = YOLO('yolov8n.pt')

# Fine-tune
results = model.train(
    data='child_adult_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='child_adult_model',
    patience=10,
    save=True
)

print("Training tamamlandı!")
print(f"Best model: runs/detect/child_adult_model/weights/best.pt")
