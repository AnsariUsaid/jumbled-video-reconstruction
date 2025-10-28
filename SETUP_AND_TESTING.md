# Setup and Testing Instructions

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
- opencv-python (image processing)
- numpy (numerical computations)
- tensorflow (V2: deep learning)
- scikit-learn (similarity metrics)
- ultralytics (V4: YOLO detection)
- pandas (V4: data handling)
- tqdm (progress bars)

## Running the Code

### Option 1: V4 YOLO Approach (⭐ Recommended - Best Overall)

This is the **best-performing solution** with fastest execution and excellent quality.

```bash
cd src/v4_yolo_tracking
python run_pipeline.py
```

**Or run individual steps:**
```bash
python 1_extract_tracking_data.py      # ~60 seconds
python 2_compute_frame_order.py         # ~5 seconds
python 3_reconstruct_video.py           # ~10 seconds
```

**Expected Output:**
- Tracking data saved to `../../frame_tracking_data.csv`
- Frame order saved to `../../correct_frame_order.csv`
- Reconstructed video saved to `../../output/reconstructed_video_v4.mp4`

**Total Time:** ~2 minutes
**File Size:** 54MB
**Quality:** Excellent (5.7px avg step, 99% detection)

### Option 2: V2 Deep Learning Approach (Best Similarity)

Highest similarity score with semantic understanding.

```bash
cd src/v2_deeplearning
python run_pipeline.py
```

**Or run individual steps:**
```bash
python extract_features_cnn.py          # ~60 seconds
python build_similarity_matrix_cnn.py   # ~2 seconds
python order_frames_improved.py         # ~5 seconds
python reconstruct_video.py             # ~5 seconds
```

**Expected Output:**
- Features saved to `../../frames_features_cnn.pkl`
- Similarity matrix saved to `../../similarity_matrix_cnn.npy`
- Frame order saved to `../../frame_order_cnn.pkl`
- Reconstructed video saved to `../../output/reconstructed_video_cnn.mp4`

**Total Time:** ~3 minutes
**File Size:** 64MB
**Quality:** Excellent (99.6% similarity)

### Option 3: V1 ORB Approach (Baseline)

Fast baseline with good results.

```bash
cd src/v1_orb
python run_pipeline.py
```

**Expected Output:**
- All intermediate files saved automatically
- Reconstructed video saved to `../../output/reconstructed_video.mp4`

**Total Time:** ~4 minutes
**File Size:** 62MB
**Quality:** Good (89% similarity)

## Testing with Your Own Video

1. Place your jumbled video in the project root directory
2. Rename it to `jumbled_video.mp4` (or update the path in the scripts)
3. Run the appropriate approach as described above
4. Find the reconstructed video in the `output/` directory

## Comparing Results

All three approaches can be compared by viewing their output videos:

**V1 (ORB)**: `output/reconstructed_video.mp4` (62MB, 89% similarity)
**V2 (CNN)**: `output/reconstructed_video_cnn.mp4` (64MB, 99.6% similarity)  
**V4 (YOLO)**: `output/reconstructed_video_v4.mp4` (54MB, 5.7px avg step) ⭐

You can view them side-by-side manually or use video players to compare quality.

## Directory Structure

```
JumbledFramesProject/
├── src/
│   ├── v1_orb/              # ORB-based approach (baseline)
│   ├── v2_deeplearning/     # Deep learning approach (high similarity)
│   └── v4_yolo_tracking/    # YOLO approach (best overall) ⭐
├── frames/                  # Extracted frames (generated)
├── output/                  # Reconstructed videos (generated)
├── requirements.txt         # Python dependencies
├── README.md               # Project overview
├── EXECUTION_TIME_LOG.md   # Timing benchmarks
└── APPROACHES_SUMMARY.md   # Algorithm comparisons
```

## Troubleshooting

### Issue: "No module named 'tensorflow'" or 'ultralytics'
**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Out of memory" error
**Solution:** Close other applications to free up RAM. V2 requires ~1.2GB, V4 requires ~800MB.

### Issue: Model download fails
**Solution:** Check internet connection. 
- ResNet50 weights (~100MB) download on first V2 run
- YOLOv8n model (~6MB) downloads on first V4 run

### Issue: Video codec not supported
**Solution:** Install ffmpeg:
- **macOS:** `brew install ffmpeg`
- **Ubuntu:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/

### Issue: Frames directory not found
**Solution:** The scripts automatically create the frames directory. If you encounter issues:
```bash
mkdir frames output
```

## Performance Notes

- **First Run**: Models download automatically (V2: ~100MB, V4: ~6MB)
- **Subsequent Runs**: Models are cached, execution time is consistent
- **Memory Usage**: V4 (800MB) < V1 (500MB) < V2 (1.2GB)
- **CPU Usage**: All approaches can run on CPU; GPU not required

## Expected Results

### V1 (ORB) - Baseline
- Average similarity: 89%
- Reconstruction quality: Good
- Best for: Quick baseline, minimal dependencies

### V2 (ResNet50) - Highest Similarity
- Average similarity: 99.6%
- Reconstruction quality: Excellent
- Best for: Highest similarity score, semantic matching

### V4 (YOLO) - ⭐ **Recommended**
- Detection rate: 99%
- Average step: 5.7 pixels
- Reconstruction quality: Excellent
- Best for: **Overall best results** (speed + quality + size)

## Recommendation

**Use V4 (YOLO + Nearest Neighbor) for best overall results.**

It provides:
- ⚡ Fastest execution (~2 min)
- 💾 Smallest file size (54MB)
- 📹 Excellent quality (smooth 5.7px motion)
- 🎯 High detection (99%)

## Clean Up

To remove generated files:

```bash
# Remove frames
rm -rf frames/*

# Remove output videos
rm -rf output/*

# Remove intermediate files
rm *.pkl *.npy *.csv
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review APPROACHES_SUMMARY.md for algorithm details
3. Check specific approach README in src/vX folders
4. Verify system meets prerequisites

## Notes

- All video files and generated data are excluded from version control via `.gitignore`
- Intermediate files (pkl, npy, csv) are saved for debugging
- The frames directory may contain 300 images (~50MB total)
- V4 includes pre-downloaded YOLOv8n model for convenience

---

*Last Updated: October 27, 2024*
