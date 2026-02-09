# YOLO Pool - Swimming Pool Safety System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLO](https://img.shields.io/badge/YOLO-v26m-orange.svg)](https://docs.ultralytics.com/)

Real-time pool safety system that detects **adults and children** around swimming pools using a custom-trained YOLOv26m model, combined with **pose estimation** for skeleton visualization and **drowning detection** through movement analysis.

## Architecture

```
Video Frame
    ├──► Custom YOLOv26m ──► Adult/Child BBoxes ──► ByteTrack Tracking
    │                                                       │
    ├──► YOLOv8-pose ──► 17 COCO Keypoints ── IoU Match ◄──┘
    │                                              │
    └──► Annotated Frame ◄── Movement Analysis + Head Visibility ──► Status
```

## Features

- **Dual-Model Architecture** - Custom YOLOv26m for adult/child classification + YOLOv8-pose for skeleton keypoints
- **Drowning Detection** - Automatic alerts based on stationary behavior and head visibility
- **Pose Estimation** - 17-point COCO skeleton overlay with color-coded joints
- **Multi-Person Tracking** - ByteTrack-based tracking with per-person danger scoring
- **Real-time Webcam** - Live detection with FPS display and adjustable confidence
- **Video/Image Processing** - Batch processing with H.264 output

### Drowning Detection

| Status | Color | Condition |
|--------|-------|-----------|
| Active | Green | Normal movement detected |
| Stationary | Yellow | No movement for 5+ seconds |
| STATIONARY (Danger) | Orange | No movement for 10+ seconds |
| DROWNING ALERT! | Red | 10+ seconds stationary + head not visible |

## Installation

```bash
git clone https://github.com/mftnakrsu/yolo_pool.git
cd yolo_pool
pip install -r requirements.txt
```

### Development Install

```bash
pip install -e .
```

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)

## Usage

### Video Processing

```bash
python detect.py -i video.mp4 -o output.mp4

# Specify models
python detect.py -i video.mp4 -o output.mp4 \
    --model best_pool_adult_child.pt \
    --pose-model yolov8n-pose.pt

# Headless (no preview)
python detect.py -i video.mp4 -o output.mp4 --no-preview
```

### Image Processing

```bash
python detect.py -i photo.jpg -o result.jpg
```

### Webcam (Real-time)

```bash
# Basic
python detect.py --webcam

# Advanced real-time with FPS display and controls
python realtime.py

# Custom settings
python realtime.py --model best_pool_adult_child.pt --conf 0.4 --camera 0
```

**Real-time controls:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `+` / `-` | Adjust confidence threshold |

### IP Camera (RTSP)

```bash
python detect.py -i "rtsp://user:pass@ip:port/stream"
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i, --input` | Input video/image path | - |
| `-o, --output` | Output file path | - |
| `-m, --model` | Custom detection model | `best_pool_adult_child.pt` |
| `-p, --pose-model` | Pose estimation model | `yolov8n-pose.pt` |
| `-c, --conf` | Confidence threshold (0-1) | `0.25` |
| `-w, --webcam` | Use webcam | `False` |
| `--no-preview` | Disable live preview | `False` |

### Python API

```python
from yolo_pool import PoolPersonDetector

detector = PoolPersonDetector(
    model_path='best_pool_adult_child.pt',
    pose_model_path='yolov8n-pose.pt',
    conf_threshold=0.25
)

# Process video
detector.process_video('input.mp4', 'output.mp4')

# Process image
detector.process_image('photo.jpg', 'result.jpg')

# Webcam
detector.process_webcam()
```

## Tools

Utility scripts in `tools/` for dataset preparation and video processing:

```bash
# Auto-label images with YOLO
python tools/auto_label.py --source frames/ --output dataset/

# Extract frames from videos
python tools/extract_frames.py --input videos/ --output frames/ --interval 30

# Add pose overlay to videos
python tools/add_pose_to_videos.py --folder videos/

# Batch process videos with stylish bounding boxes
python tools/batch_process.py --input videos/ --output processed/

# Child-alone warning detection
python tools/process_child_adult.py --input video.mp4

# Split videos into segments
python tools/split_videos.py --input video.mp4 --duration 6
```

## Training

The project includes a Google Colab training pipeline in `notebooks/train_colab.ipynb`.

### Dataset Preparation

1. Label images with 2 classes: `adult` (0) and `child` (1) using [CVAT](https://www.cvat.ai/) or [Roboflow](https://roboflow.com/)
2. Export labels in YOLO format
3. Upload images and labels to Google Drive

### Training on Colab

1. Upload `notebooks/train_colab.ipynb` to Google Colab
2. Set runtime to **GPU** (T4 or better)
3. Run all cells - the notebook will:
   - Mount your Drive and extract the dataset
   - Analyze class distribution
   - Split into train/val/test (80/15/5)
   - Train YOLOv26m for 100 epochs
   - Save the best model to Drive

### Dataset

The `dataset/` directory contains 2,540 labeled pool images:

```
dataset/
├── images/   # 2,540 JPEG images (~470 MB)
└── labels/   # 2,540 YOLO format labels (~8.5 MB)
```

## Project Structure

```
yolo_pool/
├── detect.py                    # Main CLI: video/image/webcam detection
├── realtime.py                  # Real-time webcam detection with controls
│
├── yolo_pool/                   # Python package
│   ├── __init__.py
│   ├── detector.py              # PoolPersonDetector class (core)
│   ├── visualization.py         # Skeleton, bbox, status drawing
│   └── utils.py                 # IoU, video helpers
│
├── tools/                       # Utility scripts
│   ├── auto_label.py            # Auto-labeling with YOLO
│   ├── extract_frames.py        # Frame extraction from videos
│   ├── add_pose_to_videos.py    # Pose overlay for videos
│   ├── batch_process.py         # Batch video processing
│   ├── process_child_adult.py   # Child-alone warning system
│   └── split_videos.py          # Video segmentation
│
├── notebooks/
│   └── train_colab.ipynb        # Google Colab training notebook
│
├── configs/
│   └── data.yaml                # Dataset config template
│
├── dataset/                     # Training dataset
│   ├── images/                  # 2,540 labeled pool images
│   └── labels/                  # YOLO format annotations
│
├── docs/
│   └── architecture.md          # Dual model architecture details
│
├── requirements.txt             # Pinned dependencies
├── LICENSE                      # MIT License
└── README.md
```

## Model Selection

| Model | Size | Use Case |
|-------|------|----------|
| `yolov8n-pose.pt` | 6 MB | Lightweight pose estimation (edge devices) |
| `yolov8m-pose.pt` | 52 MB | Balanced pose estimation |
| `yolo26m.pt` | 44 MB | Base detection model |
| `best_pool_adult_child.pt` | ~44 MB | Custom trained adult/child detector |

## Edge Deployment (Jetson Nano)

```bash
# Performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Export to TensorRT
yolo export model=best_pool_adult_child.pt format=engine device=0

# Run with TensorRT model
python detect.py -i video.mp4 --model best_pool_adult_child.engine
```

## License

MIT License - see [LICENSE](LICENSE) for details.
