# Setup and Testing Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB of free RAM
- Internet connection (for first-time model download)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
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
- tensorflow (deep learning framework)
- scikit-learn (similarity metrics)

## Running the Code

### Option 1: V2 Deep Learning Approach (Recommended)

This is the best-performing solution with 99.6% similarity.

```bash
cd src/v2_deeplearning
python extract_frames.py
python extract_features_cnn.py
python build_similarity_matrix_cnn.py
python order_frames_improved.py
python reconstruct_video.py
```

**Expected Output:**
- Frames extracted to `../../frames/` directory
- Features saved to `../../frames_features_cnn.pkl`
- Similarity matrix saved to `../../similarity_matrix_cnn.npy`
- Frame order saved to `../../frame_order_cnn.pkl`
- Reconstructed video saved to `../../output/reconstructed_video_cnn.mp4`

**Total Time:** ~57 seconds

### Option 2: V1 ORB Approach (Faster but less accurate)

This approach is faster but provides lower accuracy (~85% similarity).

```bash
cd src/v1_orb
python run_pipeline.py
```

**Expected Output:**
- All intermediate files saved automatically
- Reconstructed video saved to `../../output/reconstructed_video.mp4`

**Total Time:** ~43 seconds

## Testing with Your Own Video

1. Place your jumbled video in the project root directory
2. Rename it to `jumbled_video.mp4` (or update the path in the scripts)
3. Run the appropriate approach as described above
4. Find the reconstructed video in the `output/` directory

## Comparing Results

To compare the results of both approaches:

```bash
cd src/comparison
python compare_results.py
```

This will generate a comparison report showing:
- Frame-by-frame similarity scores
- Average similarity percentages
- Visual comparison (if applicable)

## Directory Structure

```
JumbledFramesProject/
├── src/
│   ├── v1_orb/              # ORB-based approach
│   ├── v2_deeplearning/     # Deep learning approach (recommended)
│   └── comparison/          # Comparison utilities
├── frames/                  # Extracted frames (generated)
├── output/                  # Reconstructed videos (generated)
├── requirements.txt         # Python dependencies
├── README.md               # Project overview
├── EXECUTION_TIME_LOG.md   # Timing benchmarks
└── V2_Algorithm_Description.md  # Technical documentation

```

## Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Out of memory" error
**Solution:** Close other applications to free up RAM. V2 requires ~1.2GB RAM.

### Issue: Model download fails
**Solution:** Check internet connection. ResNet50 weights (~100MB) download on first run.

### Issue: Video codec not supported
**Solution:** Install ffmpeg:
- **macOS:** `brew install ffmpeg`
- **Ubuntu:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/

### Issue: Frames directory not found
**Solution:** The scripts automatically create the frames directory. If you encounter issues:
```bash
mkdir frames
```

## Performance Notes

- **First Run**: V2 takes longer due to ResNet50 model download
- **Subsequent Runs**: Model is cached, execution time is consistent
- **Memory Usage**: V2 uses more RAM but provides significantly better results
- **CPU Usage**: Both approaches can run on CPU; GPU not required

## Expected Results

### V1 (ORB)
- Average similarity: ~85%
- Reconstruction quality: Moderate (visible jumps)
- Best for: Quick prototyping

### V2 (ResNet50) ⭐ Recommended
- Average similarity: ~99.6%
- Reconstruction quality: Excellent (smooth transitions)
- Best for: Production use and evaluation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the Algorithm_Description.md for technical details
3. Verify system meets prerequisites

## Clean Up

To remove generated files:

```bash
# Remove frames
rm -rf frames/*

# Remove output videos
rm -rf output/*

# Remove intermediate files (automatically ignored by git)
rm *.pkl *.npy
```

## Notes

- All video files and generated data are excluded from version control via `.gitignore`
- Intermediate files (pkl, npy) are saved for debugging but not committed
- The frames directory may contain 300 images (~50MB total)
