# Setup and Testing Instructions

## Main Approach: V8 Motion-Based Optimization (Recommended) ⭐

**This project uses the V8 approach as the primary solution for video frame reconstruction.** V8 uses motion modeling with hybrid cost optimization (motion + SSIM) and 2-opt local search to achieve smooth, high-quality reconstruction.

> **Note:** Previous approaches (v1-v6) are preserved in `tries(inaccurate)_approaches/` for reference and educational purposes. For details about these experimental approaches, see [ApproachesTried.md](ApproachesTried.md).

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
- **opencv-python** - Video and image processing
- **numpy** - Numerical computations
- **pandas** - Data tracking and analysis
- **ultralytics** - YOLOv8x object detection
- **scipy** - Distance calculations and optimization
- **scikit-image** - SSIM (Structural Similarity) calculation
- **tqdm** - Progress bars
- **tensorflow** - (Optional, for older approaches only)
- **scikit-learn** - (Optional, for older approaches only)
- **matplotlib** - (Optional, for visualization)

---

## Running V8 Pipeline (⭐ Recommended)

This is the **main and recommended approach** with optimal motion-based reconstruction.

### Quick Start - Complete Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# Navigate to V8 directory
cd src/v8

# Run complete pipeline
python run_pipeline.py
```

**Expected Output:**
```
--- Running Step 1: YOLO+ByteTrack Tracking ---
Tracking data saved to ../../output/v8_tracking_data.csv
Total frames in video: 300
Frames with person detected: 300
Detection rate: 100.0%

--- Running Step 2: Building Motion Model ---
Total detections: 300
Unique frames: 300
Detection rate: 100.0%
Motion model data saved to ../../output/v8_motion_model.csv

--- Running Step 3: Calculating Hybrid Motion + SSIM Cost ---
Building hybrid motion + SSIM cost matrix for 300 frames
Hybrid weights: alpha (motion) = 0.3, beta (SSIM) = 0.7
Loading video frames for SSIM calculation...
Cost matrix saved to ../../output/v8_cost_matrix.npy

--- Running Step 4: Solving for Optimal Order with 2-opt ---
Loaded cost matrix: (300, 300)
Starting greedy nearest neighbor ordering...
Performing 2-opt optimization...
Optimal order saved to ../../output/v8_optimal_order.csv

--- Running Step 5: Reconstructing Video ---
Loaded 300 frames from optimal order
Reconstructed video saved to ../../output/reconstructed_video_v8.mp4

--- V8 Hybrid Pipeline Complete! ---
```

**Total Time:** ~4-5 minutes  
**Output:** `output/reconstructed_video_v8.mp4`

### Run Individual Steps (Optional)

For debugging or inspecting intermediate outputs:

```bash
cd src/v8

# Step 1: YOLO Detection (~1-2 minutes)
python step_1_run_yolo_bytetrack.py
# Output: output/v8_tracking_data.csv

# Step 2: Build Motion Model (~5 seconds)
python step_2_build_motion_model.py
# Output: output/v8_motion_model.csv

# Step 3: Calculate Hybrid Cost (~2-3 minutes)
python step_3_calculate_smoothness_cost.py
# Output: output/v8_cost_matrix.npy

# Step 4: Solve for Optimal Order (~30 seconds)
python step_4_solve_optimal_order.py
# Output: output/v8_optimal_order.csv

# Step 5: Reconstruct Video (~10 seconds)
python step_5_reconstruct_video.py
# Output: output/reconstructed_video_v8.mp4
```

### View the Result

**Open the reconstructed video:**

```bash
# macOS
open output/reconstructed_video_v8.mp4

# Windows
start output\reconstructed_video_v8.mp4

# Linux
xdg-open output/reconstructed_video_v8.mp4
```

**Or watch online:**  
🎥 [**V8 Result on Google Drive**](https://drive.google.com/file/d/1qgqtBGqebWZMTp7QrbBKkbRV3byJdPxY/view?usp=sharing)

---

## Expected Results - V8

When you run the V8 pipeline successfully, you should see:

**Console Output Summary:**
- ✅ 100% detection rate (all 300 frames)
- ✅ Motion model with velocity and size features
- ✅ Hybrid cost matrix (motion + SSIM)
- ✅ 2-opt optimization converges
- ✅ Smooth reconstructed video

**Generated Files:**

1. **output/v8_tracking_data.csv** (~10KB)
   ```csv
   frame,x,y,w,h,confidence
   0,1117.01,709.64,289.8,289.9,0.93
   1,1119.52,713.85,295.0,295.0,0.94
   ...
   ```
   Person detection data: centroid (x,y), size (w,h), confidence

2. **output/v8_motion_model.csv** (~15KB)
   - Enhanced tracking with motion features
   - Velocities, areas, normalized coordinates

3. **output/v8_cost_matrix.npy** (~700KB)
   - 300×300 pairwise cost matrix
   - Hybrid: 0.3×motion + 0.7×(1-SSIM)

4. **output/v8_optimal_order.csv** (~2KB)
   ```csv
   ordered_frame
   145
   203
   78
   ...
   ```
   Final optimized frame sequence (result of 2-opt)

5. **output/reconstructed_video_v8.mp4** ⭐
   - 1920×1080, 30 FPS, 10 seconds
   - Smooth motion reconstruction

---

## Testing with Your Own Video

1. Place your jumbled video in the project root directory
2. Rename it to `jumbled_video.mp4` (or update paths in the scripts)
3. Run the V8 pipeline as described above
4. Find the reconstructed video in `output/reconstructed_video_v8.mp4`

**Note:** V8 is optimized for videos with a person/subject moving through frames. For other types of content, results may vary.

---

## Directory Structure

```
JumbledFramesProject/
├── src/
│   └── v8/                           # ⭐ Main Solution
│       ├── run_pipeline.py             # Complete pipeline
│       ├── step_1_run_yolo_bytetrack.py
│       ├── step_2_build_motion_model.py
│       ├── step_3_calculate_smoothness_cost.py
│       ├── step_4_solve_optimal_order.py
│       ├── step_5_reconstruct_video.py
│       └── yolov8x.pt                  # Auto-downloads
│
├── tries(inaccurate)_approaches/     # Previous experiments
│   ├── v1_orb/
│   ├── v2_deeplearning/
│   ├── v4_yolo_tracking/
│   └── v6_hybrid_yolov8x/
│
├── output/
│   ├── v8_tracking_data.csv          # Generated
│   ├── v8_motion_model.csv           # Generated
│   ├── v8_cost_matrix.npy            # Generated
│   ├── v8_optimal_order.csv          # Generated
│   └── reconstructed_video_v8.mp4    # ⭐ Final output
│
├── jumbled_video.mp4                 # Input video
├── requirements.txt                  # Dependencies
├── README.md                         # Main documentation
├── SETUP_AND_TESTING.md              # This file
├── EXECUTION_TIME_LOG.md             # Benchmarks
└── ApproachesTried.md                # Algorithm comparisons
```

---

## Performance Notes

### V8 Pipeline
- **First Run**: YOLOv8x model (~136MB) downloads automatically
- **Subsequent Runs**: Model is cached, consistent ~4-5 minute execution
- **Memory Usage**: ~1.5-2GB RAM recommended
- **CPU/GPU**: Works on CPU; GPU optional but speeds up YOLO detection
- **Storage**: Final video ~60MB, intermediate files ~700KB

### Hardware Requirements
- **Minimum**: 2GB RAM, 2-core CPU
- **Recommended**: 4GB RAM, 4-core CPU, GPU (optional)
- **Storage**: ~500MB for models and output files

---

## Troubleshooting

### Issue: "No module named 'ultralytics'" or 'cv2'
**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "No module named 'skimage'"
**Solution:** Install scikit-image:
```bash
pip install scikit-image
```

### Issue: "Out of memory" error
**Solution:** V8 requires ~2GB RAM. Close other applications or:
- Reduce batch size in YOLO detection
- Process frames in smaller chunks
- Use a machine with more RAM

### Issue: Model download fails (YOLOv8x)
**Solution:** 
- Check internet connection
- YOLOv8x (~136MB) downloads on first run
- Model is cached in `~/.cache/ultralytics/` for subsequent runs
- Manual download: Place `yolov8x.pt` in `src/v8/` directory

### Issue: Video codec not supported
**Solution:** Install ffmpeg:
- **macOS:** `brew install ffmpeg`
- **Ubuntu:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/

### Issue: SSIM calculation slow
**Solution:** 
- SSIM compares all frame pairs (300×300 = 90,000 comparisons)
- This is normal and takes 2-3 minutes
- GPU can speed this up if available
- Progress is shown during execution

### Issue: 2-opt takes too long
**Solution:**
- 2-opt typically converges in 30-60 seconds
- If longer, the cost matrix may have many local minima
- You can stop and use the current solution (still good)

---

## Advanced Configuration

### Adjusting Hybrid Cost Weights

In `step_3_calculate_smoothness_cost.py`, you can adjust the balance between motion and visual similarity:

```python
# Current default (prioritizes visual similarity)
calculate_smoothness_cost(motion_model_path, cost_matrix_path, video_path, 
                         alpha=0.3, beta=0.7, use_ssim=True)

# Prioritize motion continuity
alpha=0.7, beta=0.3

# Equal weights
alpha=0.5, beta=0.5

# Motion only (no SSIM)
alpha=1.0, beta=0.0, use_ssim=False
```

**Recommendations:**
- **alpha=0.3, beta=0.7** (default) - Best perceptual quality
- **alpha=0.5, beta=0.5** - Balanced approach
- **alpha=0.7, beta=0.3** - Smoother motion, less visual accuracy

### Confidence Threshold

In `step_1_run_yolo_bytetrack.py`, adjust detection confidence:

```python
run_yolo_bytetrack(video_path, output_csv_path, conf_threshold=0.2)

# Higher confidence (fewer detections, more accurate)
conf_threshold=0.5

# Lower confidence (more detections, may include false positives)
conf_threshold=0.1
```

---

## Clean Up

To remove generated files and start fresh:

```bash
# Remove output files
rm -rf output/v8_*

# Remove all output
rm -rf output/*

# Keep only source code and documentation
```

**Note:** The V8 pipeline will regenerate all necessary files when run again.

---

## Alternative Approaches (Not Recommended)

Previous experimental approaches are preserved in `tries(inaccurate)_approaches/` for reference:

- **v1_orb** - ORB feature matching (low accuracy)
- **v2_deeplearning** - CNN semantic ordering (no motion tracking)
- **v4_yolo_tracking** - Simple nearest neighbor (inferior to v8)
- **v6_hybrid_yolov8x** - Hybrid CNN+YOLO (good but v8 is better)

**These approaches are not maintained and not recommended for use.**

For details about these approaches, see [ApproachesTried.md](ApproachesTried.md).

---

## Support

If you encounter issues:

1. **Check Prerequisites**: Ensure Python 3.8+, sufficient RAM, internet connection
2. **Verify Installation**: Virtual environment activated, all dependencies installed
3. **Review Documentation**: 
   - [README.md](README.md) - Project overview and technical details
   - [ApproachesTried.md](ApproachesTried.md) - Algorithm comparisons
   - [EXECUTION_TIME_LOG.md](EXECUTION_TIME_LOG.md) - Performance benchmarks
4. **Check Output**: Look for error messages in console output
5. **Validate Files**: Ensure `jumbled_video.mp4` exists in project root

---

## Important Notes

- **Use V8 exclusively** - It's the current best approach
- Previous approaches (v1-v6) are for reference only
- All large files are excluded from version control via `.gitignore`
- Intermediate CSV/NPY files are saved for debugging
- YOLOv8x model auto-downloads and caches locally
- Generated videos are high quality (1920×1080, 30 FPS)
- SSIM calculation is computationally intensive but worth it for quality

---

## What's Next?

After successful setup and testing:

1. **View Results**: Compare jumbled vs reconstructed video
2. **Understand Pipeline**: Read through the 5 step files in `src/v8/`
3. **Experiment**: Try different hybrid cost weights
4. **Learn More**: Check [README.md](README.md) for technical deep dive
5. **Explore History**: See [ApproachesTried.md](ApproachesTried.md) for evolution

---

*Last Updated: November 1, 2024*  
*Main Approach: V8 Motion-Based Optimization with Hybrid Cost and 2-opt*
