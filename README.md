# Jumbled Frames Project

A Python project to extract, analyze, and reconstruct jumbled video frames using computer vision techniques.

## 📁 Project Structure

```
JumbledFramesProject/
 ├── src/                # all Python files
 ├── frames/             # extracted frames will be saved here
 ├── output/             # reconstructed video will be saved here
 ├── README.md           # documentation
 ├── requirements.txt    # dependency list
 └── jumbled_video.mp4   # place your provided video here
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

1. Place your `jumbled_video.mp4` file in the root project directory
2. Run the frame extraction and reconstruction script from the `src/` directory
3. Processed frames will be saved in `frames/`
4. Reconstructed video will be saved in `output/`

## 📝 Notes

This project uses ORB (Oriented FAST and Rotated BRIEF) features for frame similarity detection and ordering.
