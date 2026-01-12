# YOLO Pool - Havuz Guvenlik Sistemi

YOLOv8 tabanli havuz guvenlik ve bogulma tespit sistemi. Pose estimation kullanarak havuzdaki kisileri takip eder ve potansiyel bogulma durumlarini tespit eder.

## Ozellikler

- **Pose Detection**: YOLOv8-pose modeli ile 17 keypoint tespiti
- **Kisi Takibi**: ByteTrack ile coklu kisi takibi
- **Bogulma Tespiti**: Hareket analizi + poz analizi ile erken uyari
- **Cocuk/Yetiskin Siniflandirma**: Ozel egitilmis model destegi

## Bogulma Tespit Algoritmasi

Sistem asagidaki kriterlere gore bogulma tespiti yapar:

1. **Hareket Analizi**: Kisi 5+ saniye hareketsiz kalirsa uyari
2. **Poz Analizi**: Bas/yuz keypoint'leri gorunmuyorsa tehlike skoru artar
3. **Tehlike Skorlama**:
   - `Active` (Yesil): Normal hareket
   - `Stationary` (Sari): 5+ saniye hareketsiz
   - `STATIONARY (Danger)` (Turuncu): 10+ saniye hareketsiz
   - `DROWNING ALERT!` (Kirmizi): Hareketsiz + bas gorunmuyor

## Jetson Nano Kurulumu

### Gereksinimler

- Jetson Nano (4GB onerilir)
- JetPack 4.6+
- Python 3.8+
- USB Kamera veya IP Kamera

### 1. Sistem Hazirlik

```bash
# Sistem guncelleme
sudo apt update && sudo apt upgrade -y

# Swap alanini artir (8GB onerilir)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. Python Ortami

```bash
# Virtual environment olustur
python3 -m venv venv
source venv/bin/activate

# Pip guncelle
pip install --upgrade pip
```

### 3. PyTorch Kurulumu (Jetson icin)

```bash
# PyTorch 1.10+ for Jetson
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
pip install -e .
cd ..
```

### 4. Proje Kurulumu

```bash
# Repo'yu klonla
git clone https://github.com/mftnakrsu/yolo_pool.git
cd yolo_pool

# Gereksinimleri yukle
pip install ultralytics opencv-python numpy

# Model indir (ilk calistirmada otomatik iner)
# yolov8n-pose.pt (hafif, hizli)
# yolov8m-pose.pt (dengeli)
# yolov8l-pose.pt (yuksek dogruluk)
```

### 5. Performans Optimizasyonu

```bash
# Jetson performans modu
sudo nvpmodel -m 0
sudo jetson_clocks

# TensorRT export (opsiyonel, hiz icin)
yolo export model=yolov8n-pose.pt format=engine device=0
```

## Kullanim

### Video Dosyasi Isleme

```bash
# Temel kullanim
python pool_person_detection.py -i video.mp4 -o output.mp4

# YOLOv8m-pose ile (daha iyi dogruluk)
python pool_person_detection.py -i video.mp4 -o output.mp4 -m yolov8m-pose.pt

# Onizleme olmadan (headless server)
python pool_person_detection.py -i video.mp4 -o output.mp4 --no-preview
```

### Webcam / USB Kamera

```bash
# Varsayilan kamera
python pool_person_detection.py --webcam

# Belirli kamera index'i
python pool_person_detection.py --webcam 0
```

### IP Kamera (RTSP)

```bash
# RTSP stream
python pool_person_detection.py -i "rtsp://username:password@ip:port/stream"
```

### Parametreler

| Parametre | Aciklama | Varsayilan |
|-----------|----------|------------|
| `-i, --input` | Giris video/resim yolu | - |
| `-o, --output` | Cikis dosyasi yolu | - |
| `-m, --model` | YOLO model dosyasi | yolov8n-pose.pt |
| `-c, --conf` | Guven esigi (0-1) | 0.25 |
| `--webcam` | Webcam kullan | False |
| `--no-preview` | Onizleme gosterme | False |

## Model Secimi

| Model | Boyut | FPS (Jetson) | Dogruluk |
|-------|-------|--------------|----------|
| yolov8n-pose.pt | 6MB | ~15-20 | Dusuk |
| yolov8s-pose.pt | 23MB | ~10-15 | Orta |
| yolov8m-pose.pt | 52MB | ~5-8 | Iyi |
| yolov8l-pose.pt | 84MB | ~3-5 | Yuksek |

**Oneri**: Jetson Nano icin `yolov8n-pose.pt` veya TensorRT export edilmis model kullanin.

## Dosya Yapisi

```
yolo_pool/
├── pool_person_detection.py  # Ana tespit scripti
├── train_child_adult.py      # Cocuk/yetiskin model egitimi
├── colab_train.py            # Google Colab egitim scripti
├── split_videos.py           # Video bolme araci
├── demos/                    # Demo videolari
│   ├── output_whatsapp1_pose.mp4
│   ├── output_whatsapp2_pose.mp4
│   └── output_whatsapp3_pose.mp4
└── README.md
```

## Demo Videolar

Demo ciktilarini `demos/` klasorunde bulabilirsiniz. Bu videolar YOLOv8m-pose modeli ile islenmistir.

## Sorun Giderme

### Jetson Nano Bellek Hatasi
```bash
# Swap alanini kontrol et
free -h

# Model boyutunu dusur
python pool_person_detection.py -m yolov8n-pose.pt ...
```

### Dusuk FPS
```bash
# Performans modunu aktifle
sudo nvpmodel -m 0
sudo jetson_clocks

# TensorRT kullan
yolo export model=yolov8n-pose.pt format=engine
python pool_person_detection.py -m yolov8n-pose.engine ...
```

### Kamera Acilamadi
```bash
# Kamera kontrol
ls /dev/video*

# OpenCV test
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

## Lisans

MIT License

## Katkida Bulunma

Pull request'ler memnuniyetle karsilanir. Buyuk degisiklikler icin once bir issue aciniz.
