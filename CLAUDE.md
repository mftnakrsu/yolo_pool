# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO Pool is a pool safety monitoring system using dual-model YOLO detection (custom YOLOv26m for adult/child classification + YOLOv8-pose for skeleton keypoints) with drowning detection via movement analysis and head visibility tracking.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run detection on video/image
python detect.py --input video.mp4 --output output.mp4 --model best_pool_adult_child.pt

# Real-time webcam detection (controls: q=quit, s=screenshot, +/-=confidence)
python realtime.py --model best_pool_adult_child.pt --conf 0.5 --camera 0

# Tools (all use argparse, run from repo root)
python tools/auto_label.py --source frames/ --output dataset/ --model yolo26m.pt
python tools/extract_frames.py --input videos/ --output frames/ --interval 30
```

No test suite, build system, or linter is configured.

## Architecture

### Dual-Model Detection Pipeline

1. **Custom YOLOv26m** (`best_pool_adult_child.pt`) — detects adults/children with ByteTrack tracking
2. **YOLOv8-pose** (`yolov8n-pose.pt`) — 17-point COCO skeleton keypoints
3. **IoU matching** links detection boxes to pose keypoints (threshold: 0.3)
4. **Movement analysis** tracks per-person position history to determine drowning state

### Drowning Detection States

Active (moving) → Stationary (5s no movement) → Danger (10s) → DROWNING ALERT (10s + head not visible). Head visibility is determined by keypoints 0-4 (nose, eyes, ears).

### Package Structure (`yolo_pool/`)

- `detector.py` — `PoolPersonDetector` class: the core orchestrator. Runs both models, matches poses to detections, tracks movement history per person, and determines drowning state.
- `visualization.py` — Drawing functions (`draw_skeleton`, `draw_bbox`, `draw_status`) with left/right/center color-coded skeleton rendering.
- `utils.py` — `compute_iou`, `open_video_capture`, `create_video_writer` helpers.
- `__init__.py` — Exports `PoolPersonDetector`.

### Entry Points

- `detect.py` — CLI for batch processing (video/image/RTSP/webcam)
- `realtime.py` — Real-time webcam with live confidence adjustment

### Tools (`tools/`)

Utility scripts for dataset preparation and video processing. Each uses argparse. Tools that import from `yolo_pool` need `sys.path.insert(0, ...)` since they live in a subdirectory.

## Key Conventions

- **Model files** (`.pt`) are gitignored — not tracked in the repo
- **Dataset** (`dataset/`) with 2540 images + YOLO-format labels is also gitignored
- **YOLO classes**: 0=adult, 1=child (defined in `configs/data.yaml`)
- **Colors are BGR** (OpenCV convention): adult=cyan, child=orange, drowning states go green→yellow→orange→red
- **Training** is done via Google Colab notebook at `notebooks/train_colab.ipynb`
