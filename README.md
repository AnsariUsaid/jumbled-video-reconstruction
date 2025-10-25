# Jumbled Frames Project

A Python project to extract, analyze, and reconstruct jumbled video frames using computer vision techniques.

## 📁 Project Structure

```
JumbledFramesProject/
 ├── src/                    # all Python files
 │   ├── extract_frames.py   # Phase 2: Extract frames from video
 │   └── extract_features.py # Phase 3: Extract ORB features
 ├── frames/                 # extracted frames (300 .jpg files)
 ├── output/                 # reconstructed video will be saved here
 ├── frames_features.pkl     # saved ORB features (4.6MB)
 ├── README.md               # documentation
 ├── requirements.txt        # dependency list
 ├── jumbled_video.mp4       # provided jumbled video
 └── venv/                   # virtual environment
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

### Phase 4: Frame Comparison & Ordering (Coming Next)
- Compare frames using extracted ORB descriptors
- Build similarity matrix
- Determine correct frame order

### Phase 5: Video Reconstruction (Coming Next)
- Reconstruct video in correct order
- Save to `output/reconstructed_video.mp4`

## 📝 Notes

This project uses ORB (Oriented FAST and Rotated BRIEF) features for frame similarity detection and ordering.

**Current Progress:**
- ✅ Phase 1: Project setup complete
- ✅ Phase 2: 300 frames extracted from jumbled video
- ✅ Phase 3: ORB features extracted and saved (4.6MB features file)
- ⏳ Phase 4: Frame comparison (next step)
- ⏳ Phase 5: Video reconstruction (final step)
