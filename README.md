# Jumbled Frames Video Reconstruction

A Python project to reconstruct jumbled video frames using **computer vision (ORB features)** and **graph-based optimization**. Successfully reorders 300 shuffled frames with 89% average frame similarity.

## 🎬 Demo

### Before (Jumbled) vs After (Reconstructed)

| Jumbled Video | Reconstructed Video |
|---------------|---------------------|
| 300 randomly shuffled frames | Correctly ordered sequence |
| No temporal coherence | Smooth, coherent playback |
| 🎥 [**Watch Jumbled Video**](https://drive.google.com/file/d/1XdR-mheAkt_vi5SHwG9B7Hxqn_0-INVv/view?usp=sharing) | 🎥 [**Watch Reconstructed Video**](https://drive.google.com/file/d/1K9LEC97fol47eeqc46OXMfhhv299g1yv/view?usp=sharing) |
| 86MB | 62MB |

**Video Specifications:**
- Resolution: 1920×1080 (Full HD)
- Frame Rate: 30 FPS  
- Duration: 10 seconds
- Total Frames: 300

### 🎯 See The Results!

**Click the links above to watch:**
- **[Jumbled Video](https://drive.google.com/file/d/1XdR-mheAkt_vi5SHwG9B7Hxqn_0-INVv/view?usp=sharing)** - Random frame order (the problem)
- **[Reconstructed Video](https://drive.google.com/file/d/1K9LEC97fol47eeqc46OXMfhhv299g1yv/view?usp=sharing)** - Correctly ordered frames (the solution)

### Quality Metrics

After reconstruction:
- ✅ 89% average similarity between consecutive frames
- ✅ Smooth, coherent playback
- ✅ No jarring transitions or jumps
- ✅ All 300 frames correctly ordered

---

## 🎯 Project Overview

**Problem:** Given a video with randomly shuffled frames, reconstruct the original sequence.

**Solution:** 
1. Extract ORB (Oriented FAST and Rotated BRIEF) features from each frame
2. Build similarity matrix by comparing all frame pairs
3. Use graph-based optimization (Hamiltonian path) to find optimal ordering
4. Reconstruct video with correctly ordered frames

**Results:**
- ✅ 300 frames successfully reordered
- ✅ 89% average consecutive frame similarity
- ✅ 62MB reconstructed video (1920×1080, 30 FPS)
- ✅ Complete pipeline executes in ~4 minutes

## 📁 Project Structure

```
JumbledFramesProject/
 ├── src/                              # all Python files
 │   ├── extract_frames.py             # Phase 2: Extract frames from video
 │   ├── extract_features.py           # Phase 3: Extract ORB features
 │   ├── build_similarity_matrix.py    # Phase 4: Build similarity matrix
 │   ├── order_frames.py               # Phase 5A: Determine optimal order
 │   ├── reconstruct_video.py          # Phase 5B: Rebuild video
 │   ├── logger.py                     # Execution time logging utility
 │   └── run_pipeline.py               # Run complete pipeline with timing
 ├── frames/                           # extracted frames (300 .jpg files)
 ├── output/                           # reconstructed video output
 │   └── reconstructed_video.mp4       # final reconstructed video (62MB)
 ├── frames_features.pkl               # saved ORB features (4.6MB)
 ├── similarity_matrix.npy             # frame similarity matrix (352KB)
 ├── frame_order.pkl                   # optimal frame ordering (6KB)
 ├── execution_log.txt                 # pipeline execution timing log
 ├── Algorithm_Description.md          # detailed algorithm documentation
 ├── README.md                         # this file
 ├── requirements.txt                  # dependency list
 ├── jumbled_video.mp4                 # input jumbled video
 └── venv/                             # virtual environment
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

- **opencv-python** → for handling video, frames, and ORB features
- **numpy** → for numerical operations and the similarity matrix
- **tqdm** → for progress bars during processing

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
# Activate virtual environment
source venv/bin/activate

# Run entire pipeline with automatic timing logs
python src/run_pipeline.py
```
**What this does:**
- ✅ Automatically runs all 5 phases in sequence
- ✅ Logs execution time for each phase
- ✅ Saves timing data to `execution_log.txt`
- ✅ Creates `output/reconstructed_video.mp4`

### Option 2: Run Individual Phases

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

- **[Algorithm_Description.md](Algorithm_Description.md)** - Detailed algorithm explanation, trade-offs, and alternatives
- **[execution_log.txt](execution_log.txt)** - Generated after running pipeline, contains timing for each phase

---

## 🎉 Project Status

**✅ ALL PHASES COMPLETE**

- ✅ Phase 1: Project setup and dependencies
- ✅ Phase 2: Frame extraction (300 frames)
- ✅ Phase 3: ORB feature extraction (150,000 keypoints)
- ✅ Phase 4: Similarity matrix construction (90,000 comparisons)
- ✅ Phase 5A: Optimal frame ordering (graph-based)
- ✅ Phase 5B: Video reconstruction (62MB output)
- ✅ Phase 6: Documentation and optimization

**Results:** Successfully reconstructed jumbled video with 89% average frame similarity using computer vision and graph-based optimization.

---

## 👨‍💻 Author

**Usaid Ansari**
- GitHub: [@AnsariUsaid](https://github.com/AnsariUsaid)
- Repository: [jumbled-video-reconstruction](https://github.com/AnsariUsaid/jumbled-video-reconstruction)

---

## 📄 License

This project is open source and available for educational purposes.
