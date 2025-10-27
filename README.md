# Jumbled Frames Video Reconstruction

A Python project to reconstruct jumbled video frames using computer vision and graph-based optimization.

**Four approaches implemented:**
- **V1 (ORB)**: 89% similarity using ORB features + Hamming distance
- **V2 (CNN)**: 99.6% similarity using ResNet50 + Cosine similarity
- **V3 (Distance)**: Person tracking with distance-based ordering
- **V4 (YOLO)**: 99% detection rate using YOLOv8 + Nearest Neighbor ⭐ **BEST**

## 🎬 Demo

### Before (Jumbled) vs After (Reconstructed)

| Jumbled Video | V1 Reconstructed | V2 Reconstructed | V4 Reconstructed |
|---------------|------------------|------------------|------------------|
| Random order | ORB (89% similarity) | CNN (99.6% similarity) | YOLO (5.7px avg step) ⭐ |
| 🎥 [**Watch**](https://drive.google.com/file/d/1Rzi3UD2sxJbSYcNVARlPqvLbA7PBAKIH/view?usp=sharing) | 🎥 [**Watch V1**](https://drive.google.com/file/d/1s1Cir_J_sommAQWMEaIXUlTXQYM29-Fj/view?usp=sharing) | 🎥 [**Watch V2**](https://drive.google.com/file/d/1neINE83qJeY4Sc_N3AW9jvE_SI8vtXZi/view?usp=sharing) | 🎥 [**Watch V4**](https://drive.google.com/file/d/1ALhd1qUhMGspCIEmxpIiORcFw-T5unA7/view?usp=sharing) |
| 86MB | 62MB | 64MB | 54MB |

**Video Specifications:**
- Resolution: 1920×1080 (Full HD)
- Frame Rate: 30 FPS  
- Duration: 10 seconds
- Total Frames: 300

### Comparison

| Metric | V1 (ORB) | V2 (CNN) | V4 (YOLO) ⭐ | Best |
|--------|----------|----------|--------------|------|
| Average Similarity | 445/500 (89.0%) | 995.68/1000 (99.57%) | - | V2 |
| Detection Rate | - | - | 99% (297/300) | V4 |
| Avg Step Distance | - | - | 5.7 pixels | V4 |
| File Size | 62MB | 64MB | 54MB | V4 |
| Speed | ~4 min | ~3 min | ~2 min | V4 |
| Visual Quality | Good | Excellent | **Excellent** ⭐ | **V4** |

---

## 🎯 Project Overview

**Problem:** Given a video with randomly shuffled frames, reconstruct the original sequence.

### V1: ORB Features + Hamming Distance
1. Extract ORB (Oriented FAST and Rotated BRIEF) keypoints from each frame
2. Build similarity matrix using Brute-Force Matcher (Hamming distance)
3. Use graph-based optimization (Hamiltonian path) to find optimal ordering
4. Reconstruct video with correctly ordered frames

**Results:** 89% average consecutive frame similarity

### V2: CNN Features + Cosine Similarity
1. Extract deep features using pre-trained ResNet50 (2048-dim vectors)
2. Build similarity matrix using cosine similarity (vectorized, fast)
3. Use same graph-based optimization (proven to work well)
4. Reconstruct video with highly accurate ordering

**Results:** 99.6% average consecutive frame similarity (+10.57% improvement)

### V4: YOLO Object Detection + Nearest Neighbor ⭐ **RECOMMENDED**
1. Detect person in each frame using YOLOv8 (99% detection rate)
2. Track person centroid position (x, y) across frames
3. Start from bottom-right corner, greedily select nearest unvisited frame
4. Build smooth path with average 5.7 pixel displacement between frames

**Results:** Excellent visual quality with smooth transitions, fastest execution (~2 min)

**Why V4 is Best:**
- ✅ Direct spatial tracking (uses actual person position, not image features)
- ✅ Fastest execution time (~2 minutes vs 3-4 minutes)
- ✅ Smallest file size (54MB vs 62-64MB)
- ✅ Very smooth motion (5.7px average step)
- ✅ Simple and interpretable algorithm
- ✅ Robust to lighting and background changes

---
- ✅ 62MB reconstructed video (1920×1080, 30 FPS)
- ✅ Complete pipeline executes in ~4 minutes

## 📁 Project Structure

```
JumbledFramesProject/
 ├── src/
 │   ├── v1_orb/                          # V1: ORB Features Approach
 │   │   ├── extract_frames.py            # Extract frames from video
 │   │   ├── extract_features.py          # ORB feature extraction
 │   │   ├── build_similarity_matrix.py   # Hamming distance matching
 │   │   ├── order_frames.py              # Graph-based ordering
 │   │   ├── reconstruct_video.py         # Rebuild video
 │   │   ├── logger.py                    # Execution time logging
 │   │   └── run_pipeline.py              # Run complete V1 pipeline
 │   │
 │   ├── v2_deeplearning/                 # V2: CNN Features Approach
 │   │   ├── extract_frames.py            # Reused from V1
 │   │   ├── extract_features_cnn.py      # ResNet50 feature extraction
 │   │   ├── build_similarity_matrix_cnn.py # Cosine similarity
 │   │   ├── order_frames_improved.py     # Same graph algorithm
 │   │   ├── reconstruct_video.py         # Rebuild video
 │   │   └── run_pipeline.py              # Run complete V2 pipeline
 │   │
 │   ├── v4_yolo_tracking/                # V4: YOLO + Nearest Neighbor ⭐ BEST
 │   │   ├── 1_extract_tracking_data.py   # YOLOv8 person detection
 │   │   ├── 2_compute_frame_order.py     # Nearest neighbor ordering
 │   │   ├── 3_reconstruct_video.py       # Video reconstruction
 │   │   ├── run_pipeline.py              # Run complete V4 pipeline ⭐
 │   │   └── README.md                    # V4 documentation
 │   │
 │   └── comparison/                      # Compare approaches
 │       └── compare_results.py
 │
 ├── frames/                              # Extracted frames (300 .jpg)
 ├── output/
 │   ├── reconstructed_video.mp4          # V1 output (62MB, 89%)
 │   ├── reconstructed_video_cnn.mp4      # V2 output (64MB, 99.6%)
 │   └── reconstructed_video_v4.mp4       # V4 output (54MB) ⭐ BEST
 │
 ├── Algorithm_Description.md             # V1 algorithm documentation
 ├── V2_Algorithm_Description.md          # V2 algorithm documentation
 ├── UTILITIES.md                         # Utility files guide
 ├── README.md                            # This file
 └── requirements.txt                     # Dependencies
```

## 🧰 Setup Instructions

### 1. Set Up Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # (on Mac/Linux)
venv\Scripts\activate         # (on Windows)
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

## 📦 Dependencies

### V1 (ORB Approach)
```
opencv-python
numpy
tqdm
```

### V2 (CNN Approach) - Additional
```
tensorflow
scikit-learn
matplotlib
```

### V4 (YOLO Approach) - Additional ⭐
```
ultralytics
pandas
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Run V4 (YOLO) - Best Overall ⭐ **RECOMMENDED**

```bash
source venv/bin/activate
cd src/v4_yolo_tracking

# Run complete V4 pipeline (one command)
python run_pipeline.py                   # ~2 minutes total

# Output: output/reconstructed_video_v4.mp4
```

**Or run individual steps:**
```bash
python 1_extract_tracking_data.py       # ~60 seconds - YOLO detection
python 2_compute_frame_order.py         # ~5 seconds - Nearest neighbor
python 3_reconstruct_video.py           # ~10 seconds - Video creation
```

**Why V4?**
- ⚡ Fastest execution (~2 minutes)
- 🎯 Most accurate person tracking (99% detection)
- 📹 Excellent visual quality (5.7px avg motion)
- 💾 Smallest file size (54MB)

### Option 2: Run V2 (CNN) - Best Similarity Score

```bash
source venv/bin/activate
cd src/v2_deeplearning

# Run complete V2 pipeline (one command)
python run_pipeline.py                   # ~2-3 minutes total

# Output: output/reconstructed_video_cnn.mp4 (99.6% similarity)
```

**Or run individual steps:**
```bash
python extract_features_cnn.py          # ~60 seconds
python build_similarity_matrix_cnn.py   # ~2 seconds
python order_frames_improved.py         # ~5 seconds
python reconstruct_video.py             # ~5 seconds
```

### Option 3: Run V1 (ORB) - Baseline Approach

```bash
source venv/bin/activate
cd src/v1_orb

# Run complete V1 pipeline
python run_pipeline.py                  # ~4 minutes total

# Output: output/reconstructed_video.mp4 (89% similarity)
```

#### Phase 1: Setup ✅
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Phase 2: Extract Frames ✅
```bash
python src/extract_frames.py
```
- Extracts all frames from `jumbled_video.mp4`
- Saves frames to `frames/` directory

### Phase 3: Extract ORB Features ✅
```bash
python src/extract_features.py
```
- Reads all extracted frames from `frames/`
- Converts each to grayscale for optimal ORB processing
- Extracts keypoints and descriptors using OpenCV's ORB_create()
- Saves features to `frames_features.pkl` for next step

### Phase 4: Build Similarity Matrix ✅
```bash
python src/build_similarity_matrix.py
```
- Loads saved ORB descriptors from `frames_features.pkl`
- Creates Brute-Force Matcher with Hamming distance (optimal for ORB)
- Compares every frame pair (i ≠ j) - 44,850 comparisons for 300 frames
- Counts matches and stores in 300x300 similarity matrix
- Saves matrix to `similarity_matrix.npy` for next phase

### Phase 5A: Determine Optimal Frame Order ✅
```bash
python src/order_frames.py
```
- Loads similarity matrix from Phase 4
- Uses graph-based approach with hybrid optimization
- Finds best starting pair (highest similarity frames)
- Builds initial path using nearest neighbor algorithm
- Applies 2-opt optimization to improve the path
- Saves optimal frame order to `frame_order.pkl`

**Algorithm Details:**
- **Step 1**: Find the two frames with highest similarity (likely consecutive)
- **Step 2**: Build path from both directions using greedy nearest neighbor
- **Step 3**: Apply 2-opt local optimization to improve path quality
- **Result**: Near-optimal frame sequence with high consecutive similarities

### Phase 5B: Video Reconstruction ✅
```bash
python src/reconstruct_video.py
```
- Loads optimal frame order from `frame_order.pkl`
- Reads frames in the determined sequence
- Writes frames to video file using cv2.VideoWriter
- Outputs reconstructed video to `output/reconstructed_video.mp4`

**Results:**
- Successfully reconstructed 300 frames into coherent video
- Output: 62MB MP4 file at 30 FPS
- Duration: 10 seconds
- Resolution: 1920x1080 (Full HD)

---

## 📊 Performance Metrics

### Quality Metrics
```
Average Consecutive Similarity:  445/500 (89%)
Minimum Similarity:              363/500 (73%)
Maximum Similarity:              478/500 (96%)
Standard Deviation:              8.62 (very consistent)
Low Similarity Pairs (<100):     0 (perfect!)
```

### Execution Time (approximate)
```
Phase 1: Setup                   < 1 second
Phase 2: Frame Extraction        ~ 5 seconds
Phase 3: Feature Extraction      ~ 30 seconds
Phase 4: Similarity Matrix       ~ 180 seconds
Phase 5A: Frame Ordering         ~ 2 seconds
Phase 5B: Video Reconstruction   ~ 5 seconds
─────────────────────────────────────────────
Total Pipeline:                  ~ 223 seconds (~3.7 minutes)
```

### Storage Requirements
```
Input Video:                     90MB
Extracted Frames (300):          ~300MB
ORB Features:                    4.6MB
Similarity Matrix:               352KB
Frame Order:                     6KB
Output Video:                    62MB
─────────────────────────────────────────────
Total:                           ~457MB
```

---

## 🧠 Algorithm Overview

The reconstruction uses a **hybrid graph-based approach**:

1. **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF)
   - 500 keypoints per frame
   - Binary descriptors (fast Hamming distance matching)
   - Rotation and scale invariant

2. **Similarity Computation**: Brute-Force Matcher
   - Hamming distance for binary descriptors
   - 44,850 frame pair comparisons
   - Results in 300×300 similarity matrix

3. **Frame Ordering**: Graph Optimization
   - Treat frames as graph nodes, similarities as edge weights
   - Find Hamiltonian path (visit each frame once)
   - Hybrid algorithm:
     - Find best starting pair (max similarity)
     - Greedy nearest neighbor path construction
     - 2-opt optimization for refinement

4. **Video Reconstruction**: Sequential Write
   - Read frames in optimal order
   - Write to MP4 using cv2.VideoWriter

For detailed algorithm explanation, see **[Algorithm_Description.md](Algorithm_Description.md)**

---

## 📂 Expected Output

After running the complete pipeline:

1. **frames/** - 300 extracted JPG frames (numbered sequentially from original)
2. **frames_features.pkl** - Serialized ORB features (4.6MB)
3. **similarity_matrix.npy** - NumPy array (300×300, 352KB)
4. **frame_order.pkl** - Optimal frame sequence (6KB)
5. **output/reconstructed_video.mp4** - ⭐ **Final reconstructed video (62MB)**
6. **execution_log.txt** - Timing log for all phases

### File Structure After Execution
```
output/
└── reconstructed_video.mp4    # The reconstructed video - ready to play!
```

**To view the result:** Open `output/reconstructed_video.mp4` in any video player (VLC, QuickTime, Windows Media Player, etc.)

---

## 🎥 How to View/Share Videos

### Option 1: Local Playback
```bash
# Play the reconstructed video
open output/reconstructed_video.mp4           # macOS
start output\reconstructed_video.mp4          # Windows  
xdg-open output/reconstructed_video.mp4       # Linux
```

### Option 2: Upload to Video Platform (Recommended for Sharing)
Upload both videos to showcase your results:
- **YouTube** - Create unlisted or public video
- **Google Drive** - Generate shareable link
- **Vimeo** - Upload and embed
- **GitHub Release** - Attach files to release (< 2GB)

Then add links to this README:
```markdown
### 📺 Video Links
- [Jumbled Video (Before)](your-youtube-link)
- [Reconstructed Video (After)](your-youtube-link)
```

### Option 3: Create Comparison GIF/Video
```bash
# Install ffmpeg if not already installed
brew install ffmpeg           # macOS
sudo apt install ffmpeg       # Linux
choco install ffmpeg          # Windows

# Create side-by-side comparison (first 5 seconds)
ffmpeg -i jumbled_video.mp4 -i output/reconstructed_video.mp4 \
  -filter_complex "[0:v]scale=480:-1[v0];[1:v]scale=480:-1[v1];[v0][v1]hstack" \
  -t 5 -r 15 comparison.gif

# Or create comparison video
ffmpeg -i jumbled_video.mp4 -i output/reconstructed_video.mp4 \
  -filter_complex "[0:v]scale=960:-1,drawtext=text='Jumbled':fontsize=30:x=10:y=10[v0]; \
                   [1:v]scale=960:-1,drawtext=text='Reconstructed':fontsize=30:x=10:y=10[v1]; \
                   [v0][v1]hstack" \
  comparison_video.mp4
```

---

## 🔬 Technical Details

### Why ORB?
- **Speed**: 100x faster than SIFT
- **Free**: No patent restrictions (unlike SURF)
- **Effective**: Binary descriptors work well for frame matching
- **Robust**: Handles rotation and scale changes

### Why Graph-Based Ordering?
- **Better than greedy**: Considers global structure
- **Faster than exact**: Avoids factorial complexity
- **Near-optimal**: 2-opt refinement improves quality
- **Proven**: Adapted from TSP solvers

### Future Optimizations
- Parallel similarity computation (4-8x speedup)
- Adaptive feature counts based on scene complexity
- Multi-scale matching for better accuracy
- GPU acceleration for feature extraction

---

## 📖 Documentation

- **[README.md](README.md)** - Main project documentation with all approaches
- **[Algorithm_Description.md](Algorithm_Description.md)** - V1 ORB approach details
- **[V2_Algorithm_Description.md](V2_Algorithm_Description.md)** - V2 CNN approach with trade-offs analysis
- **[src/v4_yolo_tracking/README.md](src/v4_yolo_tracking/README.md)** - V4 YOLO approach documentation
- **[UTILITIES.md](UTILITIES.md)** - Utility files explained

## 🔬 Why Multiple Approaches?

This project demonstrates **iterative improvement** in machine learning projects:

1. **V1 (ORB):** Fast to implement, good baseline (89% similarity)
   - Limitation: Only captures local features, misses semantic context
   
2. **V2 (CNN):** Semantic understanding, excellent similarity (99.6%)
   - Limitation: Computationally expensive, slower execution
   
3. **V4 (YOLO):** Direct object tracking, best overall results ⭐
   - Advantage: Tracks actual person position, not image features
   - Fastest execution, smallest file size, excellent visual quality

This is how real-world ML projects evolve! All approaches are preserved to show:
- The problem-solving process
- Trade-offs between approaches (speed vs accuracy vs complexity)
- When to use which method

**Recommendation:** Use **V4** for best overall results (speed + quality + simplicity)

---

## 🎉 Project Status

**✅ ALL APPROACHES COMPLETE**

### V1 (ORB - Baseline)
- ✅ ORB feature extraction (150,000 keypoints)
- ✅ Hamming distance similarity matrix
- ✅ Graph-based ordering
- ✅ Video reconstruction (62MB, 89% similarity)

### V2 (CNN - High Similarity)
- ✅ ResNet50 deep feature extraction
- ✅ Cosine similarity matrix
- ✅ Graph-based ordering
- ✅ Video reconstruction (64MB, 99.6% similarity)

### V4 (YOLO - Best Overall) ⭐
- ✅ YOLOv8 person detection (99% detection rate)
- ✅ Nearest neighbor greedy ordering
- ✅ Video reconstruction (54MB, 5.7px avg step)
- ✅ **RECOMMENDED** for best results

**Results:** Successfully reconstructed jumbled video using three different approaches. **V4 (YOLO)** provides the best balance of speed, quality, and simplicity.

---

## 👨‍💻 Author

**Ansari Usaid Anzer**
- GitHub: [@AnsariUsaid](https://github.com/AnsariUsaid)
- Repository: [jumbled-video-reconstruction](https://github.com/AnsariUsaid/jumbled-video-reconstruction)

---

## 📄 License

This project is open source and available for educational purposes.
