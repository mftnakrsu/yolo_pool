# YOLO Pool - Swimming Pool Safety System

A real-time pool safety system that detects **adults and children** around swimming pools using a custom-trained YOLOv26m model, combined with **pose estimation** for skeleton visualization and **drowning detection** through movement and head visibility analysis.

## Features

- **Dual-Model Architecture**: Custom YOLOv26m for adult/child classification + YOLOv8-pose for skeleton keypoints
- **Drowning Detection**: Automatic alerts based on stationary behavior and head visibility
- **Pose Estimation**: 17-point COCO skeleton overlay with color-coded joints
- **Multi-Person Tracking**: ByteTrack-based tracking with per-person danger scoring
- **Real-time Webcam**: Live detection with FPS display and adjustable confidence
- **Video/Image Processing**: Batch processing with H.264 output

## How It Works

The system runs two YOLO models simultaneously on each frame:

1. **Custom YOLOv26m** detects people and classifies them as `adult` or `child`
2. **YOLOv8-pose** extracts 17 body keypoints per person
3. Detections are matched to pose data via **IoU** (Intersection over Union)
4. Each person is tracked over time to analyze movement patterns

### Drowning Detection Algorithm

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

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)

## Usage

### Video Processing

```bash
# With custom model + pose estimation
python pool_person_detection.py -i video.mp4 -o output.mp4

# Specify models explicitly
python pool_person_detection.py -i video.mp4 -o output.mp4 \
    --model best_pool_adult_child.pt \
    --pose-model yolov8n-pose.pt

# Headless (no preview window)
python pool_person_detection.py -i video.mp4 -o output.mp4 --no-preview
```

### Image Processing

```bash
python pool_person_detection.py -i photo.jpg -o result.jpg
```

### Webcam (Real-time)

```bash
# Basic
python pool_person_detection.py --webcam

# Advanced real-time with FPS display and controls
python realtime_detection.py

# With custom settings
python realtime_detection.py --model best_pool_adult_child.pt --conf 0.4 --camera 0
```

**Real-time controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `+` / `-` | Adjust confidence threshold |

### IP Camera (RTSP)

```bash
python pool_person_detection.py -i "rtsp://user:pass@ip:port/stream"
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

## Training Your Own Model

The project includes a ready-to-use Google Colab training pipeline.

### Dataset Preparation

1. Label your images with 2 classes: `adult` (0) and `child` (1) using [CVAT](https://www.cvat.ai/) or [Roboflow](https://roboflow.com/)
2. Export labels in YOLO format
3. Upload images and labels to Google Drive

### Training on Colab

1. Upload `train_colab.ipynb` to Google Colab
2. Set runtime to **GPU** (T4 or better)
3. Run all cells - the notebook will:
   - Mount your Drive and extract the dataset
   - Analyze class distribution
   - Split into train/val/test (80/15/5)
   - Train YOLOv26m for 100 epochs
   - Save the best model to Drive

```bash
# Or use the auto-labeling script to generate initial labels
python auto_label.py
```

## Project Structure

```
yolo_pool/
├── pool_person_detection.py      # Main: dual model detection + pose + drowning
├── realtime_detection.py         # Real-time webcam detection
├── train_colab.py                # Google Colab training script
├── train_colab.ipynb             # Colab notebook (same as above)
├── auto_label.py                 # Auto-labeling with pretrained model
├── add_pose_to_videos.py         # Add pose overlay to existing videos
├── extract_frames.py             # Extract frames from videos for labeling
├── batch_process_stylish.py      # Batch video processing with styled output
├── process_child_adult_video.py  # Child-alone warning system
├── split_videos.py               # Video segmentation utility
├── requirements.txt              # Python dependencies
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

# Export to TensorRT for faster inference
yolo export model=best_pool_adult_child.pt format=engine device=0

# Run with TensorRT model
python pool_person_detection.py -i video.mp4 --model best_pool_adult_child.engine
```

## License

MIT License
