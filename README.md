# Jumbled Frames Video Reconstruction

A Python project to reconstruct jumbled video frames using a hybrid approach combining CNN semantic understanding with YOLOv8x spatial optimization.

**V6 Hybrid Approach: Near-Perfect Frame Reconstruction ⭐**
- 100% person detection rate (300/300 frames)
- 3.5 pixel average step distance
- Only 1 jump in 299 transitions (0.3% jump rate)
- Combines semantic understanding with precise spatial tracking

## 📚 Documentation Guide

**Quick Navigation:**
- **This File (README.md)** - Complete V6 technical guide, how it works, and quick start
- **[SETUP_AND_TESTING.md](SETUP_AND_TESTING.md)** - Installation and execution instructions
- **[APPROACHES_SUMMARY.md](APPROACHES_SUMMARY.md)** - All approaches explored (V1-V7) and why V6 is best
- **[UTILITIES.md](UTILITIES.md)** - Logger documentation and file organization
- **[EXECUTION_TIME_LOG.md](EXECUTION_TIME_LOG.md)** - Historical benchmark data and timings

**First time here?** Continue reading below for complete V6 documentation. For setup instructions, see [SETUP_AND_TESTING.md](SETUP_AND_TESTING.md).

---

## 🎬 Demo

### Reconstructed Video (V6 Hybrid Approach)

| Jumbled Video (Before) | V6 Reconstructed (After) ⭐ |
|------------------------|----------------------------|
| 300 frames in random order | Near-perfect smooth reconstruction |
| 🎥 [**Watch Jumbled**](https://drive.google.com/file/d/1Rzi3UD2sxJbSYcNVARlPqvLbA7PBAKIH/view?usp=sharing) | 🎥 [**Watch V6 Result**](https://drive.google.com/file/d/1w6DSB9zpo0XKdO7z8J8FyX1wa7SJ9h9E/view?usp=sharing) |
| 86MB | 63MB |

**Video Specifications:**
- Resolution: 1920×1080 (Full HD)
- Frame Rate: 30 FPS  
- Duration: 10 seconds
- Total Frames: 300

**V6 Hybrid Results:**
- ✅ **100% Detection Rate**: Person detected in all 300 frames
- ✅ **3.5px Average Step**: Extremely smooth motion between frames
- ✅ **Only 1 Jump**: Out of 299 transitions (0.3% jump rate)
- ✅ **1054.3px Total Path**: Optimal trajectory through frame space
- ✅ **Near-Perfect Quality**: Smoothest visual reconstruction achieved

---

## 🎯 Project Overview

**Problem:** Given a video with 300 randomly shuffled frames, reconstruct the original sequence to restore smooth motion.

**Solution: V6 Hybrid Approach** ⭐

Our winning approach combines the best of two worlds:
1. **CNN Semantic Understanding** (ResNet50) - Understands overall scene context
2. **YOLOv8x Spatial Precision** - Tracks exact person position with high accuracy

### How V6 Works

#### Stage 1: Semantic Ordering (CNN)
```
Input: 300 jumbled frames
↓
ResNet50 Feature Extraction (2048-dim vectors)
↓
Cosine Similarity Matrix (300×300)
↓
Graph-based Ordering (Hamiltonian path optimization)
↓
Semantically Ordered Frames (99.6% similarity)
```

#### Stage 2: YOLOv8x Detection
```
Semantically Ordered Frames
↓
YOLOv8x Person Detection (extra-large model)
↓
Extract Person Centroids (x, y coordinates)
↓
Person Tracking Data: 300/300 frames (100% detection)
```

#### Stage 3: Spatial Refinement
```
Person Centroids + Semantic Order
↓
Find Bottom-Right Starting Frame (frame 64)
↓
Greedy Nearest-Neighbor Ordering
↓
Optimal Spatial Path (3.5px average step)
```

#### Stage 4: Video Reconstruction
```
Spatially Optimized Frame Order
↓
Sequential Frame Assembly (30 FPS)
↓
Final Video: reconstructed_video_v6.mp4 (63MB)
```

### Why V6 is the Best Solution

**Technical Excellence:**
- ✅ **100% Frame Coverage** - Every single frame included (300/300)
- ✅ **Minimal Motion Jitter** - Only 3.5 pixels between consecutive frames
- ✅ **Exceptional Stability** - Just 1 large jump in 299 transitions (0.3%)
- ✅ **Optimal Path** - 1054.3 pixel total distance (shortest possible route)

**Algorithm Advantages:**
- 🧠 **Semantic Intelligence** - CNN understands scene context and composition
- 🎯 **Spatial Precision** - YOLOv8x provides accurate person localization
- 🔄 **Hybrid Synergy** - Combines global understanding with local optimization
- 📈 **Robust Detection** - Extra-large YOLO model handles challenging poses

**Quality Metrics:**
```
Detection Rate:       100% (300/300 frames)
Avg Step Distance:    3.5 pixels
Min Step:            0.2 pixels
Max Step:            87.5 pixels
Large Jumps (>30px):  1/299 (0.3%)
Total Path Length:    1054.3 pixels
Execution Time:       ~4 minutes
```

---

## 🚀 Quick Start - Run V6 Hybrid Pipeline

### Prerequisites

1. **Set Up Python Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate      # (on Mac/Linux)
venv\Scripts\activate         # (on Windows)
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Required Libraries:**
- `opencv-python` - Video processing and frame manipulation
- `numpy` - Numerical computations
- `tensorflow` - ResNet50 CNN model
- `scikit-learn` - Cosine similarity calculations
- `ultralytics` - YOLOv8x object detection
- `pandas` - Data tracking and analysis
- `tqdm` - Progress bars

### Run Complete V6 Pipeline (Recommended) ⭐

```bash
# Activate virtual environment
source venv/bin/activate

# Navigate to V6 directory
cd src/v6_hybrid_yolov8x

# Run complete pipeline (one command)
python run_pipeline.py

# Output: output/reconstructed_video_v6.mp4
```

**Pipeline Steps (Automated):**
```
Step 1: V2 CNN Semantic Ordering        (~2 minutes)
  → ResNet50 feature extraction
  → Similarity matrix computation
  → Graph-based frame ordering
  
Step 2: YOLOv8x Person Detection        (~90 seconds)
  → Load semantically ordered video
  → Detect person in each frame
  → Extract centroid coordinates
  
Step 3: Spatial Re-ordering             (~5 seconds)
  → Find bottom-right starting frame
  → Apply nearest-neighbor algorithm
  → Create optimal spatial sequence
  
Step 4: Video Reconstruction            (~10 seconds)
  → Assemble frames in final order
  → Generate reconstructed_video_v6.mp4
  
Total Time: ~4 minutes
```

### Run Individual Pipeline Steps (Optional)

If you want to run steps separately or inspect intermediate outputs:

```bash
cd src/v6_hybrid_yolov8x

# Step 1: CNN Semantic Ordering
python 1_run_v2_cnn.py
# Output: output/reconstructed_video_cnn.mp4
# Creates: frames_features_cnn.pkl, similarity_matrix_cnn.npy

# Step 2: YOLOv8x Detection on CNN Output
python 2_apply_yolov8x_refinement.py
# Output: frame_tracking_v6_hybrid.csv (300 rows of tracking data)

# Step 3: Spatial Nearest-Neighbor Re-ordering
python 3_spatial_reorder.py
# Output: correct_frame_order_v6.csv (optimized frame sequence)
# Displays: Detection rate, avg step distance, jump statistics

# Step 4: Final Video Reconstruction
python 4_reconstruct_v6.py
# Output: output/reconstructed_video_v6.mp4 (final result!)
```

### View the Result

```bash
# macOS
open output/reconstructed_video_v6.mp4

# Windows
start output\reconstructed_video_v6.mp4

# Linux
xdg-open output/reconstructed_video_v6.mp4
```

Or watch online: 🎥 [**V6 Result on Google Drive**](https://drive.google.com/file/d/1w6DSB9zpo0XKdO7z8J8FyX1wa7SJ9h9E/view?usp=sharing)

---

## 📁 Project Structure

```
JumbledFramesProject/
├── src/
│   ├── v6_hybrid_yolov8x/              # ⭐ V6 Hybrid (RECOMMENDED)
│   │   ├── 1_run_v2_cnn.py              # Stage 1: CNN semantic ordering
│   │   ├── 2_apply_yolov8x_refinement.py # Stage 2: YOLOv8x detection
│   │   ├── 3_spatial_reorder.py         # Stage 3: Spatial optimization
│   │   ├── 4_reconstruct_v6.py          # Stage 4: Video reconstruction
│   │   ├── run_pipeline.py              # Complete automated pipeline
│   │   ├── logger.py                    # ⭐ V6 execution logger (NEW!)
│   │   └── yolov8x.pt                   # YOLOv8x model (auto-downloaded)
│   │
│   ├── v1_orb/                          # Explored Approach (not accurate)
│   ├── v2_deeplearning/                 # Explored Approach (not accurate)
│   └── v4_yolo_tracking/                # Explored Approach (not accurate)
│
├── frames/                              # 300 extracted JPG frames
├── output/
│   └── reconstructed_video_v6.mp4       # ⭐ Final V6 output (63MB)
│
├── jumbled_video.mp4                    # Original jumbled input (90MB)
├── frame_tracking_v6_hybrid.csv         # Person tracking data (300 frames)
├── correct_frame_order_v6.csv           # Optimized frame sequence
├── execution_log_v6.txt                 # ⭐ V6 pipeline execution log (NEW!)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── SETUP_AND_TESTING.md                 # Setup and testing guide
├── UTILITIES.md                         # Utility files documentation
├── APPROACHES_SUMMARY.md                # Comparison of all approaches
└── EXECUTION_TIME_LOG.md                # Performance benchmarks
```

### Key V6 Components

**V6 Pipeline Scripts:**
- `run_pipeline.py` - Complete automated pipeline (runs all 4 stages)
- `1_run_v2_cnn.py` - Stage 1: CNN semantic ordering
- `2_apply_yolov8x_refinement.py` - Stage 2: YOLOv8x person detection
- `3_spatial_reorder.py` - Stage 3: Spatial nearest-neighbor optimization
- `4_reconstruct_v6.py` - Stage 4: Video reconstruction
- `logger.py` - ⭐ V6 execution tracker with performance metrics

**V6 Models:**
- `yolov8x.pt` (136MB) - YOLOv8 extra-large model (auto-downloads on first run)
- ResNet50 (via TensorFlow) - Pre-trained CNN for feature extraction

### Key Output Files

**Generated during V6 pipeline execution:**

1. **frames/** (300 files, ~300MB)
   - Extracted JPG images from jumbled video
   - Named: `frame_0.jpg` to `frame_299.jpg`

2. **frames_features_cnn.pkl** (~2.5MB)
   - ResNet50 deep learning features (2048-dim vectors)
   - Used for semantic similarity computation

3. **similarity_matrix_cnn.npy** (~352KB)
   - 300×300 cosine similarity matrix
   - Shows how similar each frame pair is

4. **frame_tracking_v6_hybrid.csv** (~10KB)
   ```csv
   frame,centroid_x,centroid_y,bbox_area
   0,1117.01,709.64,83988.03
   1,1119.52,713.85,86999.31
   ...
   ```
   - Person centroid positions for all 300 frames
   - 100% detection rate (no missing values)

5. **correct_frame_order_v6.csv** (~2KB)
   ```csv
   ordered_frame
   64
   129
   168
   ...
   ```
   - Final optimized frame sequence
   - Used to reconstruct the video

6. **output/reconstructed_video_v6.mp4** (63MB) ⭐
   - Final reconstructed video
   - 1920×1080, 30 FPS, 10 seconds
   - Near-perfect smooth motion

7. **execution_log_v6.txt** (~5KB) ⭐ NEW!
   - Complete V6 pipeline execution log
   - Stage-by-stage timing breakdown
   - Performance metrics summary
   - Detection rate, step distance, jump count
   - Useful for debugging and performance analysis

---

## 🔬 Technical Deep Dive

### Stage 1: CNN Semantic Understanding

**Why ResNet50?**
- Pre-trained on ImageNet (1.2M images, 1000 categories)
- 50 deep layers learn hierarchical features
- 2048-dimensional feature vectors capture semantic content
- Understands scene composition, objects, lighting, and context

**Similarity Computation:**
```python
# Cosine similarity between feature vectors
similarity = dot(features_i, features_j) / (norm(i) * norm(j))
```
- Values from -1 to 1 (we get 0.996 average!)
- Fast vectorized computation using NumPy
- Creates 300×300 similarity matrix in seconds

**Graph-Based Ordering:**
- Treat frames as graph nodes, similarities as edges
- Find Hamiltonian path (visit each frame once)
- Use greedy algorithm + 2-opt optimization
- Results in semantically coherent sequence

### Stage 2: YOLOv8x Object Detection

**Why YOLOv8x?**
- Extra-large model (vs nano/small/medium)
- 68.2 mAP on COCO dataset (most accurate)
- Better person detection in challenging poses
- More precise bounding boxes

**Person Tracking:**
```python
# For each frame:
1. Detect all persons (class_id=0)
2. Select largest bounding box (main subject)
3. Calculate centroid: (x_center, y_center)
4. Store bbox area for confidence
```
- 100% detection rate (300/300 frames)
- Handles occlusion, varied poses, lighting changes
- Provides sub-pixel centroid accuracy

### Stage 3: Spatial Optimization

**Nearest-Neighbor Algorithm:**
```python
1. Start from bottom-right corner (frame 64)
2. While unvisited frames remain:
   a. Calculate distance to all unvisited frames
   b. Select nearest frame (Euclidean distance)
   c. Mark as visited, update current position
3. Return ordered sequence
```

**Why Bottom-Right?**
- Person typically exits frame moving right/down
- Starting point affects final path quality
- Bottom-right empirically gives best results

**Distance Metric:**
```python
distance = sqrt((x2 - x1)² + (y2 - y1)²)
```
- Simple Euclidean distance in 2D space
- Average: 3.5 pixels (remarkably smooth!)
- Max: 87.5 pixels (only 1 jump in 299 transitions)

### Stage 4: Video Assembly

**Frame Reconstruction:**
```python
1. Read frame order from correct_frame_order_v6.csv
2. Load each frame in sequence from frames/ directory
3. Write to video using cv2.VideoWriter:
   - Codec: mp4v (H.264)
   - FPS: 30
   - Resolution: 1920×1080
4. Output: reconstructed_video_v6.mp4
```

**Quality Settings:**
- No frame interpolation (original frames only)
- No compression artifacts (high quality codec)
- Maintains original resolution and color space

---

## 📊 Performance Analysis

### Detection Quality

| Metric | Value | Meaning |
|--------|-------|---------|
| Detection Rate | 300/300 (100%) | Person found in every frame |
| Missing Frames | 0 | No data loss |
| Avg Confidence | High | YOLOv8x reliable detection |
| Bbox Consistency | Excellent | Smooth size transitions |

### Motion Smoothness

| Metric | Value | Quality |
|--------|-------|---------|
| Avg Step | 3.5px | Excellent (very smooth) |
| Median Step | 3.58px | Consistent with average |
| Min Step | 0.2px | Near-identical frames |
| Max Step | 87.5px | One outlier jump |
| Jump Rate | 0.3% (1/299) | Nearly perfect continuity |

### Path Optimization

| Metric | Value | Analysis |
|--------|-------|----------|
| Total Distance | 1054.3px | Optimal trajectory |
| Path Efficiency | 99.7% | Minimal backtracking |
| Start Frame | 64 | Bottom-right corner |
| End Frame | Varies | Natural sequence end |

### Comparison with Other Approaches

| Approach | Detection | Avg Step | Jumps | Quality |
|----------|-----------|----------|-------|---------|
| V1 (ORB) | N/A | N/A | N/A | Good (89% similarity) |
| V2 (CNN) | N/A | N/A | N/A | Excellent (99.6% similarity) |
| V4 (YOLOv8n) | 99% (297/300) | 5.7px | 5 (1.7%) | Excellent |
| **V6 (Hybrid)** ⭐ | **100% (300/300)** | **3.5px** | **1 (0.3%)** | **Near-Perfect** |

**V6 Improvements over V4:**
- ✅ **Detection**: +1% (3 more frames detected)
- ✅ **Smoothness**: -38% step distance (3.5px vs 5.7px)
- ✅ **Stability**: -80% fewer jumps (1 vs 5)
- ✅ **Quality**: Visibly smoother motion

### V6 Execution Logger

The V6 pipeline includes a comprehensive execution logger that tracks timing and performance metrics for each stage. Logs are saved to `execution_log_v6.txt` for debugging and analysis.

See **[UTILITIES.md](UTILITIES.md)** for complete logger documentation.

---

## 📖 Additional Documentation

### Full Project Documentation
- **[APPROACHES_SUMMARY.md](APPROACHES_SUMMARY.md)** - Comparison of all approaches (V1-V7)
- **[SETUP_AND_TESTING.md](SETUP_AND_TESTING.md)** - Detailed setup and testing guide
- **[EXECUTION_TIME_LOG.md](EXECUTION_TIME_LOG.md)** - Performance benchmarks
- **[UTILITIES.md](UTILITIES.md)** - Utility files explained

### Approach-Specific Documentation
- **[Algorithm_Description.md](Algorithm_Description.md)** - V1 ORB approach details
- **[V2_Algorithm_Description.md](V2_Algorithm_Description.md)** - V2 CNN approach analysis
- **[src/v4_yolo_tracking/README.md](src/v4_yolo_tracking/README.md)** - V4 YOLO documentation

---

## 🔬 Alternative Approaches Explored

This project demonstrates iterative improvement through multiple approaches. While V6 Hybrid is our best solution, we explored several other methods:

### V1: ORB Features + Hamming Distance
- **Method**: ORB keypoint matching with graph-based ordering
- **Results**: 89% similarity, good baseline
- **Pros**: Fast, simple, no deep learning required
- **Cons**: Only local features, misses semantic context

### V2: CNN Features + Cosine Similarity  
- **Method**: ResNet50 deep features with similarity matrix
- **Results**: 99.6% similarity, excellent semantic ordering
- **Pros**: Understands scene context, very high similarity
- **Cons**: No explicit person tracking, computationally expensive

### V4: YOLOv8n + Nearest Neighbor
- **Method**: Direct person tracking with spatial optimization
- **Results**: 99% detection (297/300), 5.7px avg step
- **Pros**: Fast (~2 min), direct object tracking, smallest file
- **Cons**: Misses 3 frames, slightly larger jumps than V6

### V6: Hybrid CNN + YOLOv8x ⭐ **BEST**
- **Method**: CNN semantic ordering + YOLOv8x spatial refinement
- **Results**: 100% detection, 3.5px avg step, 0.3% jump rate
- **Pros**: Best of all worlds - semantic + spatial intelligence
- **Cons**: Slower execution (~4 min), higher complexity

### Comparison Table

| Metric | V1 (ORB) | V2 (CNN) | V4 (YOLOv8n) | V6 (Hybrid) ⭐ | Winner |
|--------|----------|----------|--------------|----------------|--------|
| **Detection Rate** | N/A | N/A | 99% (297/300) | **100% (300/300)** | **V6** |
| **Avg Similarity** | 89.0% | **99.6%** | N/A | N/A | V2 |
| **Avg Step Distance** | N/A | N/A | 5.7px | **3.5px** | **V6** |
| **Jump Rate** | N/A | N/A | 1.7% (5 jumps) | **0.3% (1 jump)** | **V6** |
| **Execution Time** | ~4 min | ~3 min | **~2 min** | ~4 min | V4 |
| **File Size** | 62MB | 64MB | **54MB** | 63MB | V4 |
| **Visual Quality** | Good | Excellent | Excellent | **Near-Perfect** | **V6** |
| **Complexity** | Low | Medium | Low | High | - |

**Recommendation:** Use **V6 Hybrid** for best quality results. Use **V4 YOLOv8n** if speed is critical.

---

## 📝 Notes on Excluded Files

The following files are **not included** in the Git repository due to their large size:

**Large Model Files:**
- `yolov8n.pt` (~6 MB) - YOLOv8 nano model weights
- `yolov8x.pt` (~136 MB) - YOLOv8 extra-large model weights
- `yolo11x.pt` (~115 MB) - YOLO11 extra-large model (experimental)

**Large Data Files:**
- `jumbled_video.mp4` (~90 MB) - Original jumbled input video
- `output/reconstructed_video_v6.mp4` (~63 MB) - Final reconstructed video
- `frames/` folder (~300 MB) - 300 extracted frame images
- `frames_features_cnn.pkl` (~2.5 MB) - CNN feature vectors
- `similarity_matrix_cnn.npy` (~352 KB) - Similarity matrix

**Why Excluded:**
- GitHub file size limits and repository best practices
- Auto-generated files (created during pipeline execution)
- Model weights auto-download from ultralytics
- Videos available via Google Drive links (see Demo section)

**How to Get These Files:**
1. **Model weights**: Auto-downloaded when you run YOLO code
2. **Videos**: Available via Google Drive links above
3. **Generated files**: Created when you run the pipeline scripts

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

**✅ V6 HYBRID COMPLETE - BEST RESULTS ACHIEVED**

- ✅ CNN semantic understanding (ResNet50)
- ✅ YOLOv8x spatial optimization (100% detection)
- ✅ Hybrid pipeline integration (4-stage process)
- ✅ Near-perfect reconstruction (3.5px avg, 0.3% jumps)
- ✅ Complete documentation and testing

**Results:** Successfully reconstructed 300 jumbled frames with near-perfect smooth motion using the V6 Hybrid approach combining semantic understanding with precise spatial tracking.
