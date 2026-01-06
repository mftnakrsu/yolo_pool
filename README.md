# PoolGuard AI - Havuz Güvenliği İçin Yapay Zeka

YOLO tabanlı havuz etrafındaki insan tespit sistemi ve modern web arayüzü.

## 🏊 Özellikler

### YOLO İnsan Tespiti
- **Gerçek zamanlı tespit**: YOLOv8 kullanarak havuz etrafındaki insanları tespit eder
- **Görüntü işleme**: Fotoğraf ve video dosyalarını işleyebilir
- **Webcam desteği**: Canlı kamera akışından tespit yapabilir
- **Yüksek doğruluk**: COCO veri seti ile eğitilmiş model

### Web Arayüzü
- Modern, responsive tasarım
- PoolScout ve PoolAngel benzeri profesyonel arayüz
- Özellikler, fiyatlandırma ve iletişim bölümleri
- Smooth scroll ve animasyonlar

## 📦 Kurulum

### Gereksinimler
- Python 3.9+
- pip

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/kullaniciadi/yolo_pool.git
cd yolo_pool
```

2. **Python paketlerini yükleyin:**
```bash
pip install -r requirements.txt
```

3. **YOLO modeli otomatik indirilecek** (ilk çalıştırmada)

## 🚀 Kullanım

### YOLO İnsan Tespiti

**Görüntü dosyası ile:**
```bash
python pool_person_detection.py --input foto.jpg --output sonuc.jpg
```

**Video dosyası ile:**
```bash
python pool_person_detection.py --input video.mp4 --output sonuc_video.mp4
```

**Webcam ile canlı tespit:**
```bash
python pool_person_detection.py --webcam
```

**Parametreler:**
- `--input, -i`: Giriş dosyası (görüntü veya video)
- `--output, -o`: Çıkış dosyası yolu
- `--model, -m`: YOLO model dosyası (varsayılan: yolov8n.pt)
- `--conf, -c`: Güven eşiği 0-1 arası (varsayılan: 0.25)
- `--webcam, -w`: Webcam kullan
- `--no-preview`: Önizlemeyi gösterme

### Web Arayüzü

1. `index.html` dosyasını bir web tarayıcısında açın
2. Veya bir web sunucusu ile çalıştırın:
```bash
python -m http.server 8000
```
Sonra tarayıcıda `http://localhost:8000` adresine gidin

## 📁 Proje Yapısı

```
yolo_pool/
├── pool_person_detection.py  # YOLO insan tespit scripti
├── index.html                 # Web arayüzü ana sayfa
├── styles.css                 # CSS stilleri
├── script.js                  # JavaScript interaktivite
├── requirements.txt           # Python bağımlılıkları
└── README.md                  # Bu dosya
```

## 🛠️ Teknolojiler

- **YOLOv8**: Ultralytics YOLO modeli
- **OpenCV**: Görüntü işleme
- **Python**: Backend işlemler
- **HTML/CSS/JavaScript**: Web arayüzü

## 📝 Lisans

Bu proje açık kaynak kodludur.

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır. Büyük değişiklikler için lütfen önce bir issue açın.

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**Not**: Bu sistem sürekli yetişkin gözetiminin yerini tutmaz. Havuz güvenliği için gerekli tüm önlemler alınmalıdır.

