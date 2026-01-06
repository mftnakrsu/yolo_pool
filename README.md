# PoolGuard - Pool Safety Detection System

YOLO-based person detection system for pool safety monitoring with a modern web interface.

## Features

- Real-time person detection using YOLOv8
- Image and video processing
- Webcam support
- Modern web interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Image detection:**
```bash
python pool_person_detection.py --input photo.jpg --output result.jpg
```

**Video detection:**
```bash
python pool_person_detection.py --input video.mp4 --output result.mp4
```

**Webcam:**
```bash
python pool_person_detection.py --webcam
```

## Technologies

- YOLOv8 (Ultralytics)
- OpenCV
- Python
- HTML/CSS/JavaScript

## License

Open source
