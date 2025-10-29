# Setup and Testing Instructions

## Main Approach: V6 Hybrid (Recommended) ⭐

**This project uses the V6 Hybrid approach as the primary solution for video frame reconstruction.** V6 combines CNN semantic understanding with YOLOv8x spatial precision to achieve near-perfect results with 100% detection rate and only 3.5px average step distance.

> **Note:** Other approaches (V1, V2, V4) are preserved for reference and educational purposes, but they are **not accurate or recommended** for production use. They were explored during development but did not achieve the quality and precision of V6.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB of free RAM
- Internet connection (for first-time model downloads)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/AnsariUsaid/jumbled-video-reconstruction.git
cd JumbledFramesProject
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- opencv-python (video and image processing)
- numpy (numerical computations)
- tensorflow (ResNet50 CNN for semantic understanding)
- scikit-learn (similarity metrics)
- ultralytics (YOLOv8x object detection)
- pandas (data tracking and analysis)
- tqdm (progress bars)

---

## Running V6 Hybrid Pipeline (⭐ Recommended)

This is the **main and recommended approach** with near-perfect reconstruction quality.

### Quick Start - Complete Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# Navigate to V6 directory
cd src/v6_hybrid_yolov8x

# Run complete pipeline
python run_pipeline.py
```

**Expected Output:**
- Stage 1: CNN semantic ordering (~2 minutes)
- Stage 2: YOLOv8x person detection (~90 seconds)
- Stage 3: Spatial re-ordering (~5 seconds)
- Stage 4: Video reconstruction (~10 seconds)
- Final video: `output/reconstructed_video_v6.mp4` (63MB)

**Total Time:** ~4 minutes  
**Quality:** Near-Perfect (3.5px avg step, 100% detection, 0.3% jump rate)

### Run Individual Steps (Optional)

For debugging or inspecting intermediate outputs:

```bash
cd src/v6_hybrid_yolov8x

# Step 1: CNN Semantic Ordering
python 1_run_v2_cnn.py
# Output: output/reconstructed_video_cnn.mp4
# Creates: frames_features_cnn.pkl, similarity_matrix_cnn.npy

# Step 2: YOLOv8x Detection
python 2_apply_yolov8x_refinement.py
# Output: frame_tracking_v6_hybrid.csv (300 rows)

# Step 3: Spatial Re-ordering
python 3_spatial_reorder.py
# Output: correct_frame_order_v6.csv
# Shows: Detection rate, avg step distance, jump statistics

# Step 4: Video Reconstruction
python 4_reconstruct_v6.py
# Output: output/reconstructed_video_v6.mp4 (final result)
```

### Expected Results

**V6 Hybrid Performance:**
```
Detection Rate:       100% (300/300 frames)
Avg Step Distance:    3.5 pixels
Min Step:            0.2 pixels
Max Step:            87.5 pixels
Large Jumps (>30px):  1/299 (0.3%)
Total Path Length:    1054.3 pixels
Visual Quality:       Near-Perfect smooth motion
```

---

## Alternative Approaches (Not Recommended)

> ⚠️ **Important:** The following approaches (V1, V2, V4) are **experimental and not accurate**. They were explored during the development process but did not achieve the required quality standards. **Use V6 Hybrid for reliable results.**

### Why Other Approaches Are Not Recommended:

- **V1 (ORB)**: Only 89% similarity, lacks semantic understanding, misses context
- **V2 (CNN)**: Good similarity but no person tracking, not optimized for motion
- **V4 (YOLOv8n)**: Missing 3 frames (99% vs 100%), larger jumps (5.7px vs 3.5px)

### If You Still Want to Explore Them:

**V1 - ORB Features (Not Accurate)**
```bash
cd src/v1_orb
python run_pipeline.py
# Output: output/reconstructed_video.mp4
# Quality: Poor (89% similarity, not reliable)
```

**V2 - CNN Features (Not Accurate)**
```bash
cd src/v2_deeplearning
python run_pipeline.py
# Output: output/reconstructed_video_cnn.mp4
# Quality: Moderate (99.6% similarity but no motion tracking)
```

**V4 - YOLOv8n (Not Accurate)**
```bash
cd src/v4_yolo_tracking
python run_pipeline.py
# Output: output/reconstructed_video_v4.mp4
# Quality: Good but inferior to V6 (missing frames, larger jumps)
```

**Again: These are NOT recommended. Use V6 Hybrid for accurate results.**

---

## Testing with Your Own Video

1. Place your jumbled video in the project root directory
2. Rename it to `jumbled_video.mp4` (or update the path in the scripts)
3. Run the V6 Hybrid pipeline as described above
4. Find the reconstructed video in `output/reconstructed_video_v6.mp4`

**Note:** V6 Hybrid is optimized for videos with a person/subject moving through frames. For other types of content, results may vary.

---

## Directory Structure

```
JumbledFramesProject/
├── src/
│   ├── v6_hybrid_yolov8x/       # ⭐ V6 Hybrid (RECOMMENDED)
│   ├── v1_orb/                  # Explored approach (not accurate)
│   ├── v2_deeplearning/         # Explored approach (not accurate)
│   └── v4_yolo_tracking/        # Explored approach (not accurate)
├── frames/                      # Extracted frames (generated)
├── output/
│   └── reconstructed_video_v6.mp4  # ⭐ Final V6 output (generated)
├── frame_tracking_v6_hybrid.csv    # Person tracking data (generated)
├── correct_frame_order_v6.csv      # Optimized frame order (generated)
├── requirements.txt             # Python dependencies
├── README.md                    # Main project documentation
├── SETUP_AND_TESTING.md         # This file
├── EXECUTION_TIME_LOG.md        # Performance benchmarks
└── APPROACHES_SUMMARY.md        # Algorithm comparisons
```

---

## Performance Notes

### V6 Hybrid
- **First Run**: YOLOv8x model (~136MB) and ResNet50 (~100MB) download automatically
- **Subsequent Runs**: Models are cached, consistent ~4 minute execution
- **Memory Usage**: ~1.5GB RAM recommended
- **CPU/GPU**: Works on CPU; GPU optional but speeds up YOLO detection
- **Storage**: Final video is 63MB, intermediate files ~300MB

### Other Approaches (Not Recommended)
- Execution times vary (2-4 minutes)
- Lower quality results
- Not suitable for production use

---

## Troubleshooting

### Issue: "No module named 'tensorflow'" or 'ultralytics'
**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Out of memory" error
**Solution:** V6 requires ~1.5GB RAM. Close other applications to free up memory.

### Issue: Model download fails (YOLOv8x or ResNet50)
**Solution:** 
- Check internet connection
- YOLOv8x (~136MB) downloads on first run
- ResNet50 (~100MB) downloads on first run
- Models are cached in `~/.cache/` for subsequent runs

### Issue: Video codec not supported
**Solution:** Install ffmpeg:
- **macOS:** `brew install ffmpeg`
- **Ubuntu:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/

### Issue: Frames directory not found
**Solution:** The V6 pipeline automatically extracts frames. If issues occur:
```bash
mkdir frames output
```

---

## Expected Results - V6 Hybrid

When you run the V6 Hybrid pipeline successfully, you should see:

**Console Output:**
```
================================================================================
V6 HYBRID - STEP 3: SPATIAL RE-ORDERING
================================================================================

Loading V2 tracking data (YOLOv8x): frame_tracking_v6_hybrid.csv
✓ Loaded 300 frames

Finding starting frame (closest to bottom-right)...
✓ Starting frame: 64
  Position: (1585.4, 737.4)

Ordering frames: 100%|████████████████████| 299/299 [00:00<00:00]

================================================================================
V6 HYBRID SPATIAL ORDERING COMPLETE
================================================================================
✓ Ordered 300 frames
✓ Total path distance: 1054.3 pixels
✓ Average step distance: 3.5 pixels
✓ Min step: 0.2 pixels
✓ Max step: 87.5 pixels
✓ Large jumps (>30px): 1/299 (0.3%)
✓ Saved frame order to: correct_frame_order_v6.csv
```

**Generated Files:**
- `output/reconstructed_video_v6.mp4` (63MB) - Final video
- `frame_tracking_v6_hybrid.csv` - Person tracking data
- `correct_frame_order_v6.csv` - Optimized frame sequence
- `frames/` directory with 300 JPG files
- `frames_features_cnn.pkl` - CNN features
- `similarity_matrix_cnn.npy` - Similarity matrix

---

## Comparison: Why V6 is Superior

| Metric | V1 (ORB) | V2 (CNN) | V4 (YOLOv8n) | V6 (Hybrid) ⭐ |
|--------|----------|----------|--------------|----------------|
| Detection Rate | N/A | N/A | 99% (297/300) | **100% (300/300)** |
| Avg Step Distance | N/A | N/A | 5.7px | **3.5px** |
| Jump Rate | N/A | N/A | 1.7% (5 jumps) | **0.3% (1 jump)** |
| Visual Quality | Poor | Moderate | Good | **Near-Perfect** |
| Accuracy | ❌ Not Accurate | ❌ Not Accurate | ❌ Not Accurate | ✅ **Accurate** |

**Recommendation: Always use V6 Hybrid for production-quality results.**

---

## Clean Up

To remove generated files and start fresh:

```bash
# Remove extracted frames
rm -rf frames/*

# Remove output videos
rm -rf output/*

# Remove intermediate files
rm *.pkl *.npy *.csv

# Keep only source code and documentation
```

**Note:** The V6 pipeline will regenerate all necessary files when run again.

---

## Support

If you encounter issues:

1. **Check Prerequisites**: Ensure Python 3.8+, sufficient RAM, and internet connection
2. **Verify Installation**: Make sure virtual environment is activated and all dependencies installed
3. **Review Documentation**: 
   - README.md for project overview
   - APPROACHES_SUMMARY.md for algorithm details
   - EXECUTION_TIME_LOG.md for performance benchmarks
4. **Use V6 Only**: Avoid V1, V2, V4 as they are not accurate

---

## Important Notes

- **Use V6 Hybrid exclusively** - It's the only accurate approach
- V1, V2, V4 are preserved for educational/reference purposes only
- All video files are excluded from version control via `.gitignore`
- Intermediate files (pkl, npy, csv) are saved for debugging
- The frames directory contains 300 images (~300MB total)
- YOLOv8x and ResNet50 models auto-download and cache locally
- Generated videos are high quality (1920×1080, 30 FPS)

---

*Last Updated: October 29, 2024*
*Main Approach: V6 Hybrid (CNN + YOLOv8x) - Near-Perfect Reconstruction*
