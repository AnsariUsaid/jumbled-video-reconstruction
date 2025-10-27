#!/usr/bin/env python3
"""
V4 Approach - Step 1: Extract Tracking Data using YOLOv8
Detects person in each frame and records centroid position and bounding box area.
"""

import cv2
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)

def extract_tracking_data(video_path, output_csv="frame_tracking_data.csv"):
    """
    Extract person tracking data from video using YOLOv8.
    
    Args:
        video_path: Path to input video
        output_csv: Path to save tracking data CSV
    """
    print("="*80)
    print("V4 APPROACH - STEP 1: YOLO TRACKING DATA EXTRACTION")
    print("="*80)
    print()
    
    # Load YOLOv8 model
    print("Loading YOLOv8n model...")
    try:
        model = YOLO("yolov8n.pt")  # Small model for speed
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  The model will be downloaded on first use.")
        model = YOLO("yolov8n.pt")
    
    print()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Error: Could not open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Information:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print()
    
    print("Detecting person in each frame...")
    print("(Using largest detection as the person)")
    print()
    
    frames_data = []
    frame_id = 0
    detected_count = 0
    
    from tqdm import tqdm
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        
        if len(boxes) > 0:
            # Choose largest detection (assume it's the person)
            areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = boxes[largest_idx]
            
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            bbox_area = (x2 - x1) * (y2 - y1)
            
            frames_data.append([frame_id, centroid_x, centroid_y, bbox_area])
            detected_count += 1
        else:
            # No detection in this frame
            frames_data.append([frame_id, np.nan, np.nan, np.nan])
        
        frame_id += 1
    
    cap.release()
    
    # Save to CSV
    df = pd.DataFrame(frames_data, columns=["frame", "centroid_x", "centroid_y", "bbox_area"])
    output_path = project_root / output_csv
    df.to_csv(output_path, index=False)
    
    print()
    print("="*80)
    print("TRACKING DATA EXTRACTION COMPLETE")
    print("="*80)
    print(f"✓ Processed {frame_id} frames")
    print(f"✓ Detected person in {detected_count}/{frame_id} frames ({detected_count/frame_id*100:.1f}%)")
    print(f"✓ Missing detections: {frame_id - detected_count} frames")
    print(f"✓ Saved tracking data to: {output_path}")
    print()
    
    return True

def main():
    project_root = Path(__file__).parent.parent.parent
    video_path = project_root / "jumbled_video.mp4"
    
    if not video_path.exists():
        print(f"✗ Error: Video not found: {video_path}")
        return
    
    success = extract_tracking_data(video_path)
    
    if success:
        print("Next step: Run 2_compute_frame_order.py")

if __name__ == "__main__":
    main()
