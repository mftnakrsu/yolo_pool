# Architecture - Dual Model Detection System

## Overview

YOLO Pool uses two YOLO models simultaneously on each frame to provide both person classification and pose estimation.

## Pipeline

```
Video Frame
    │
    ├──► Custom YOLOv26m ──► Adult/Child BBoxes ──► ByteTrack IDs
    │                                                    │
    ├──► YOLOv8-pose ──► 17 COCO Keypoints               │
    │                         │                           │
    │                    IoU Matching ◄───────────────────┘
    │                         │
    │                    Per-Person Data
    │                    (bbox + skeleton + track_id)
    │                         │
    │                  Movement Analysis
    │                  (position history, danger score)
    │                         │
    │                  Head Visibility Check
    │                  (keypoints 0-4 confidence)
    │                         │
    └──► Annotated Frame ◄── Status Assignment
                              (Active → Stationary → Danger → DROWNING)
```

## Model Details

| Model | Architecture | Classes | Input | Purpose |
|-------|-------------|---------|-------|---------|
| Custom YOLOv26m | YOLOv26-medium | adult, child | 640x640 | Person classification |
| YOLOv8n-pose | YOLOv8-nano-pose | person (17 kpts) | 640x640 | Skeleton keypoints |

## Drowning Detection Algorithm

1. **Track** each person across frames using ByteTrack
2. **Measure movement** over a sliding window (last 2 seconds)
3. **Accumulate danger score** when stationary (< 20px movement)
4. **Check head visibility** using keypoints 0-4 (nose, eyes, ears)
5. **Escalate status** based on score thresholds:

| Threshold | Status | Color |
|-----------|--------|-------|
| < 5 sec | Active | Green |
| 5 sec | Stationary | Yellow |
| 10 sec | STATIONARY (Danger) | Orange |
| 10 sec + head not visible | DROWNING ALERT! | Red |

## IoU Matching

Detection boxes from the custom model are matched to pose estimation boxes using Intersection over Union (IoU). A minimum IoU of 0.3 is required for a match, ensuring skeleton overlays are drawn on the correct person.
