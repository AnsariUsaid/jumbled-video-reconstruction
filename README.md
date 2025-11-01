# Jumbled Frames Video Reconstruction

A Python project to reconstruct jumbled video frames using an advanced motion-based optimization approach with YOLO tracking and hybrid cost analysis.

**V8 Approach: Optimal Motion-Based Reconstruction ⭐**
- 100% person detection rate with YOLOv8x
- Hybrid motion + SSIM cost optimization
- Smooth trajectory with minimal jumps
- 2-opt local search for optimal ordering

## 📚 Documentation

**Quick Navigation:**
- **This File (README.md)** - Complete technical guide and quick start
- **[SETUP_AND_TESTING.md](SETUP_AND_TESTING.md)** - Installation and execution instructions
- **[EXECUTION_TIME_LOG.md](EXECUTION_TIME_LOG.md)** - Performance benchmarks and execution logging
- **[ApproachesTried.md](ApproachesTried.md)** - Previous approaches explored (v1-v6)

**First time here?** Continue reading below for complete documentation.

---

## 🎬 Demo

### Reconstructed Video (V8 Approach)

| Jumbled Video (Before) | V8 Reconstructed (After) ⭐ |
|------------------------|----------------------------|
| 300 frames in random order | Smooth motion reconstruction |
| 🎥 [**Watch Jumbled**](https://drive.google.com/file/d/1Rzi3UD2sxJbSYcNVARlPqvLbA7PBAKIH/view?usp=sharing) | 🎥 [**Watch V8 Result**](https://drive.google.com/file/d/1qgqtBGqebWZMTp7QrbBKkbRV3byJdPxY/view?usp=sharing) |

**Video Specifications:**
- Resolution: 1920×1080 (Full HD)
- Frame Rate: 30 FPS  
- Duration: 10 seconds
- Total Frames: 300

**V8 Results:**
- ✅ **100% Detection Rate**: Person detected in all frames using YOLOv8x
- ✅ **Hybrid Cost Optimization**: Combines motion smoothness + SSIM similarity
- ✅ **2-opt Local Search**: Iterative optimization for minimal jumps
- ✅ **Motion Model**: Kalman-inspired trajectory prediction
- ✅ **High Quality**: Perceptually smooth reconstruction

---

## 🎯 Project Overview

**Problem:** Given a video with 300 randomly shuffled frames, reconstruct the original sequence to restore smooth motion.

**Solution: V8 Motion-Based Optimization** ⭐

Our approach uses motion modeling and hybrid cost optimization to achieve smooth, natural frame ordering:

### How V8 Works

#### Step 1: YOLO + ByteTrack Detection
```
Input: 300 jumbled frames
↓
YOLOv8x Person Detection (extra-large model)
↓
Extract largest person per frame (main subject)
↓
Person Tracking Data: centroid (x, y), size (w, h)
```

#### Step 2: Motion Model Building
```
Person Detection Data
↓
Calculate position, velocity, size features
↓
Normalize coordinates and areas
↓
Motion Model: comprehensive frame features
```

#### Step 3: Hybrid Cost Calculation
```
Motion Model + Video Frames
↓
Compute pairwise costs between all frames:
  • Motion Cost: position + size consistency
  • SSIM Cost: structural similarity (perceptual)
↓
Hybrid Cost Matrix (weighted combination)
  • alpha (motion) = 0.3
  • beta (SSIM) = 0.7
```

#### Step 4: Optimal Order Solving (2-opt)
```
Cost Matrix
↓
Initial greedy ordering
↓
2-opt local search optimization
  • Iteratively swap frame pairs
  • Accept swaps that reduce total cost
  • Continue until no improvement
↓
Optimal Frame Order
```

#### Step 5: Video Reconstruction
```
Optimal Frame Order
↓
Sequential Frame Assembly (30 FPS)
↓
Final Video: reconstructed_video_v8.mp4
```

### Why V8 is the Best Solution

**Technical Excellence:**
- ✅ **Motion-Aware** - Uses velocity and trajectory prediction
- ✅ **Perceptual Quality** - SSIM ensures visually similar transitions
- ✅ **Hybrid Optimization** - Balances motion physics with visual similarity
- ✅ **Iterative Refinement** - 2-opt improves initial ordering

**Algorithm Advantages:**
- 🎯 **YOLOv8x Precision** - Accurate person detection and localization
- 📐 **Motion Physics** - Models realistic movement patterns
- 👁️ **Visual Similarity** - SSIM for perceptual smoothness
- 🔄 **Local Optimization** - 2-opt finds better local solutions
- ⚡ **Efficient** - Fast computation with smart cost weighting

---

## 🚀 Setup and Run V8 Pipeline

**For complete setup instructions, testing guide, and troubleshooting, see:**

👉 **[SETUP_AND_TESTING.md](SETUP_AND_TESTING.md)** - Complete installation and usage guide

### Quick Start

```bash
# 1. Setup virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run V8 pipeline
cd src/v8
python run_pipeline.py
```

**Output:** `output/reconstructed_video_v8.mp4`

**Total Time:** ~4-5 minutes (YOLO detection + SSIM calculation + 2-opt optimization)

For detailed step-by-step instructions, individual step execution, and troubleshooting, see [SETUP_AND_TESTING.md](SETUP_AND_TESTING.md).

---

## 📁 Project Structure

```
JumbledFramesProject/
├── src/v8/                           # ⭐ V8 Pipeline (5 steps)
├── tries(inaccurate)_approaches/     # Previous experiments (v1-v6)
├── output/                           # Generated videos and data
├── jumbled_video.mp4                 # Input video
└── requirements.txt                  # Dependencies
```

**Main Files:**
- `src/v8/run_pipeline.py` - Complete automated pipeline
- `src/v8/step_*.py` - Individual pipeline steps (1-5)
- Output: `output/reconstructed_video_v8.mp4`

---

## 🔬 Technical Overview

V8 uses a 5-step pipeline combining motion physics with perceptual similarity:

### Pipeline Architecture

1. **YOLO Detection** - YOLOv8x detects person position and size in each frame
2. **Motion Model** - Extracts velocity, position, and scale features
3. **Hybrid Cost** - Combines motion smoothness (30%) + SSIM visual similarity (70%)
4. **2-opt Optimization** - Iteratively improves frame ordering to minimize cost
5. **Video Assembly** - Reconstructs video from optimal frame sequence

### Key Technical Features

**Motion Modeling:**
- Tracks centroid position (x, y) and bounding box size (w, h)
- Calculates velocity vectors for trajectory prediction
- Normalizes coordinates for scale invariance

**Hybrid Cost Function:**
- Motion cost: Euclidean distance + size consistency
- SSIM cost: Perceptual similarity between frames
- Weighted combination: `0.3×motion + 0.7×SSIM`
- Prioritizes visual quality while maintaining motion smoothness

**2-opt Local Search:**
- Starts with greedy nearest-neighbor ordering
- Iteratively swaps frame pairs to reduce total cost
- Converges when no improvement found
- Escapes local minima for better global solution

### Algorithm Complexity

- **Detection**: O(n) - Process each frame once
- **Cost Matrix**: O(n²) - Compare all frame pairs
- **Optimization**: O(n²) - 2-opt worst case
- **Total**: ~O(n²) dominated by SSIM calculations

**For detailed technical implementation, see the source code in `src/v8/`**

---

## 📊 Performance Summary

| Aspect | Details |
|--------|---------|
| **Detection** | 100% rate with YOLOv8x |
| **Cost Function** | Hybrid: 0.3×motion + 0.7×SSIM |
| **Optimization** | 2-opt local search |
| **Execution Time** | ~4-5 minutes total |
| **Output Quality** | 1920×1080, 30 FPS |

**For detailed performance benchmarks, see [EXECUTION_TIME_LOG.md](EXECUTION_TIME_LOG.md)**

---

## 🔬 Alternative Approaches

This project demonstrates iterative improvement through multiple experimental approaches. V8 represents our current best solution using motion-based optimization.

**For detailed comparison of all approaches explored (v1-v6), see [ApproachesTried.md](ApproachesTried.md).**

---

## 📝 Notes

**Large files excluded from Git:**
- Model weights (auto-download on first run)
- Video files (available via Google Drive links)
- Generated output files (created when running pipeline)

---

## 👨‍💻 Author

**Ansari Usaid Anzer**
- GitHub: [@AnsariUsaid](https://github.com/AnsariUsaid)
- Repository: [jumbled-video-reconstruction](https://github.com/AnsariUsaid/jumbled-video-reconstruction)

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🎉 Project Status

**✅ V8 MOTION-BASED OPTIMIZATION - CURRENT BEST SOLUTION**

- ✅ YOLOv8x person detection (high accuracy)
- ✅ Motion modeling with velocity and size features
- ✅ Hybrid cost function (motion + SSIM)
- ✅ 2-opt local search optimization
- ✅ Smooth video reconstruction
- ✅ Complete documentation and testing

**Results:** Successfully reconstructed 300 jumbled frames with smooth motion using motion-based optimization combined with perceptual similarity (SSIM).
