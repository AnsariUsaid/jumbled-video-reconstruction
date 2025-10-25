# Jumbled Frames Project

A Python project to extract, analyze, and reconstruct jumbled video frames using computer vision techniques.

## 📁 Project Structure

```
JumbledFramesProject/
 ├── src/                              # all Python files
 │   ├── extract_frames.py             # Phase 2: Extract frames from video
 │   ├── extract_features.py           # Phase 3: Extract ORB features
 │   ├── build_similarity_matrix.py    # Phase 4: Build similarity matrix
 │   ├── order_frames.py               # Phase 5A: Determine optimal order
 │   └── reconstruct_video.py          # Phase 5B: Rebuild video
 ├── frames/                           # extracted frames (300 .jpg files)
 ├── output/                           # reconstructed video output
 │   └── reconstructed_video.mp4       # final reconstructed video (62MB)
 ├── frames_features.pkl               # saved ORB features (4.6MB)
 ├── similarity_matrix.npy             # frame similarity matrix (352KB)
 ├── frame_order.pkl                   # optimal frame ordering (6KB)
 ├── README.md                         # documentation
 ├── requirements.txt                  # dependency list
 ├── jumbled_video.mp4                 # provided jumbled video
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

## 🚀 Usage

### Phase 1: Setup ✅
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
- Reconstruct video in correct order
- Save to `output/reconstructed_video.mp4`

## 📝 Notes

This project uses ORB (Oriented FAST and Rotated BRIEF) features for frame similarity detection and ordering.

**Current Progress:**
- ✅ Phase 1: Project setup complete
- ✅ Phase 2: 300 frames extracted from jumbled video
- ✅ Phase 3: ORB features extracted and saved (4.6MB features file)
- ✅ Phase 4: Similarity matrix built (300x300, 352KB)
- ✅ Phase 5A: Optimal frame order determined (avg similarity: 445/500)
- ✅ Phase 5B: Video successfully reconstructed (62MB, 10 seconds, 1920x1080)

## 🎉 Project Complete!

The jumbled video has been successfully reconstructed using computer vision and graph-based optimization techniques.
