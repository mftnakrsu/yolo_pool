# Hardware Recommendations - Jetson Boards

## Recommended Jetson Board for PoolGuard

### **Primary Recommendation: NVIDIA Jetson Xavier NX**

**Why Xavier NX?**
- Excellent balance of performance and price
- Can run YOLOv8 in real-time (30+ FPS)
- 384 CUDA cores for efficient inference
- 8GB RAM sufficient for YOLO models
- Low power consumption (~10-15W)
- Compact form factor for pool installations
- Good thermal management

**Specifications:**
- GPU: 384-core NVIDIA Volta GPU
- CPU: 6-core NVIDIA Carmel ARM v8.2
- Memory: 8GB 128-bit LPDDR4x
- Power: 10W/15W modes
- Price: ~$499

**Performance:**
- YOLOv8n: ~40-50 FPS
- YOLOv8s: ~25-30 FPS
- YOLOv8m: ~15-20 FPS

---

### **Budget Option: NVIDIA Jetson Nano**

**Why Nano?**
- Most affordable option (~$99-149)
- Sufficient for single camera monitoring
- Lower power consumption (~5-10W)
- Good for basic YOLOv8n inference

**Specifications:**
- GPU: 128-core NVIDIA Maxwell GPU
- CPU: Quad-core ARM A57
- Memory: 4GB LPDDR4
- Power: 5W/10W modes
- Price: ~$99-149

**Performance:**
- YOLOv8n: ~15-20 FPS
- YOLOv8s: ~8-12 FPS
- YOLOv8m: ~5-8 FPS

**Note:** May struggle with higher resolution or multiple cameras.

---

### **High-Performance Option: NVIDIA Jetson AGX Orin**

**Why AGX Orin?**
- Maximum performance for multiple cameras
- Future-proof for advanced features
- Can handle 4K video processing
- Best for commercial deployments

**Specifications:**
- GPU: 2048-core NVIDIA Ampere GPU
- CPU: 12-core ARM Cortex-A78AE
- Memory: 32GB 256-bit LPDDR5
- Power: 15W-60W
- Price: ~$1,999

**Performance:**
- YOLOv8n: ~100+ FPS
- YOLOv8s: ~60-80 FPS
- YOLOv8m: ~40-50 FPS

---

## Comparison Table

| Board | Price | YOLOv8n FPS | Power | Best For |
|-------|-------|-------------|-------|----------|
| Jetson Nano | $99-149 | 15-20 | 5-10W | Budget deployments, single camera |
| Jetson Xavier NX | $499 | 40-50 | 10-15W | **Recommended** - Best balance |
| Jetson AGX Orin | $1,999 | 100+ | 15-60W | High-end, multiple cameras |

---

## Additional Hardware Requirements

### Camera
- **Recommended:** USB 3.0 webcam or IP camera
- **Resolution:** 1080p (1920x1080) or 720p (1280x720)
- **Frame Rate:** 30 FPS minimum
- **Examples:**
  - Logitech C920/C930e (USB)
  - Raspberry Pi Camera Module v3
  - IP cameras with RTSP support

### Storage
- **Minimum:** 32GB microSD card (Class 10, UHS-I)
- **Recommended:** 64GB+ for video recordings
- **Alternative:** External USB SSD for better performance

### Power Supply
- **Jetson Nano:** 5V/4A power adapter
- **Jetson Xavier NX:** 19V/3.42A power adapter
- **Jetson AGX Orin:** Included power adapter

### Cooling
- **Jetson Nano:** Active cooling recommended (fan)
- **Jetson Xavier NX:** Passive heatsink usually sufficient
- **Jetson AGX Orin:** Built-in cooling

### Enclosure
- Weatherproof enclosure for outdoor installation
- Ventilation for heat dissipation
- Mounting brackets for pool area

---

## Setup Recommendations

### For Jetson Xavier NX (Recommended):

1. **Install JetPack SDK** (latest version)
2. **Install dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install -r requirements.txt
   ```

3. **Optimize for inference:**
   ```bash
   # Enable maximum performance mode
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

4. **Use TensorRT for acceleration:**
   - Convert YOLOv8 model to TensorRT format
   - Can achieve 2-3x speedup

### For Jetson Nano:

1. **Use YOLOv8n (nano) model** for best performance
2. **Lower resolution:** 640x480 or 720p recommended
3. **Enable performance mode:**
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

---

## Cost Estimate (Xavier NX Setup)

- Jetson Xavier NX: ~$499
- Power supply: ~$30
- microSD card (64GB): ~$20
- USB Camera: ~$50-100
- Weatherproof enclosure: ~$50-100
- **Total: ~$650-750**

---

## Performance Tips

1. **Use TensorRT:** Convert YOLO models to TensorRT for 2-3x speedup
2. **Lower resolution:** 720p instead of 1080p if needed
3. **Model selection:** YOLOv8n for real-time, YOLOv8s/m for accuracy
4. **Batch processing:** Process frames in batches if possible
5. **Optimize inference:** Use half-precision (FP16) for faster inference

---

## Where to Buy

- **Official:** NVIDIA Developer Store
- **Retailers:** Amazon, Newegg, Adafruit, SparkFun
- **Regional distributors:** Check NVIDIA's official reseller list

---

## Support

For Jetson-specific issues:
- NVIDIA Jetson Forums: https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-machines/jetson-embedded-systems/
- JetsonHacks: https://www.jetsonhacks.com/

