#!/usr/bin/env python3
"""
V6 Hybrid Approach - Step 2: Apply YOLOv8x (extra-large) refinement on V2 output
Uses YOLOv8x for more accurate detection than YOLOv8n.
"""

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def analyze_and_refine_v2(v2_video_path, output_csv="frame_tracking_v6_hybrid.csv"):
    """
    Analyze V2's CNN-ordered video with YOLOv8x and refine if needed.
    YOLOv8x is the extra-large model with better accuracy than YOLOv8n.
    
    Args:
        v2_video_path: Path to V2 CNN reconstructed video
        output_csv: Path to save tracking data
    """
    print("="*80)
    print("V6 HYBRID - STEP 2: YOLOv8x REFINEMENT ON V2 OUTPUT")
    print("="*80)
    print()
    print("Approach:")
    print("  1. V2 (CNN) provided semantic ordering (99.6% similarity)")
    print("  2. Now apply YOLOv8x (extra-large) for better spatial analysis")
    print("  3. YOLOv8x is more accurate than YOLOv8n (nano)")
    print("  4. Check person position continuity")
    print("  5. Re-order sections with large spatial jumps")
    print()
    
    # Load YOLOv8x model (extra-large, more accurate)
    print("Loading YOLOv8x model (extra-large)...")
    print("Note: First run will download ~131 MB model")
    model = YOLO("yolov8x.pt")
    print("✓ YOLOv8x model loaded")
    print()
    
    # Open V2 video
    print(f"Opening V2 CNN video: {v2_video_path}")
    cap = cv2.VideoCapture(str(v2_video_path))
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open video: {v2_video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"V2 Video Properties:")
    print(f"  Frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print()
    
    # Extract person positions with YOLOv8x
    print("Extracting person positions with YOLOv8x...")
    print("(More accurate detection than YOLOv8n)")
    frames_data = []
    frame_id = 0
    
    pbar = tqdm(total=total_frames, desc="Detecting person (YOLOv8x)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use YOLOv8x with person class only
        results = model(frame, verbose=False, classes=[0])
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        
        if len(boxes) > 0:
            # Use largest detection (assume it's the person)
            x1, y1, x2, y2 = boxes[0]
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            frames_data.append([frame_id, centroid_x, centroid_y, area])
        else:
            frames_data.append([frame_id, np.nan, np.nan, np.nan])
        
        frame_id += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    print()
    
    # Analyze spatial continuity
    df = pd.DataFrame(frames_data, columns=["frame", "centroid_x", "centroid_y", "bbox_area"])
    
    print("Analyzing V2's spatial continuity with YOLOv8x detections...")
    distances = []
    large_jumps = []
    
    for i in range(len(df) - 1):
        if not pd.isna(df.loc[i, "centroid_x"]) and not pd.isna(df.loc[i+1, "centroid_x"]):
            x1, y1 = df.loc[i, "centroid_x"], df.loc[i, "centroid_y"]
            x2, y2 = df.loc[i+1, "centroid_x"], df.loc[i+1, "centroid_y"]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distances.append(dist)
            
            if dist > 30:
                large_jumps.append((i, i+1, dist))
    
    avg_dist = np.mean(distances) if distances else 0
    max_dist = max(distances) if distances else 0
    
    detected = df['centroid_x'].notna().sum()
    detection_rate = (detected / len(df)) * 100
    
    print()
    print("V2 CNN Video Analysis (with YOLOv8x):")
    print(f"  Detection rate: {detected}/{len(df)} frames ({detection_rate:.1f}%)")
    print(f"  Average consecutive distance: {avg_dist:.1f} pixels")
    print(f"  Max distance: {max_dist:.1f} pixels")
    print(f"  Large spatial jumps (>30px): {len(large_jumps)}")
    
    if large_jumps:
        print()
        print("  Large jumps at positions:")
        for pos1, pos2, dist in large_jumps[:5]:
            print(f"    Frame {pos1} → {pos2}: {dist:.1f} pixels")
        if len(large_jumps) > 5:
            print(f"    ... and {len(large_jumps)-5} more")
    print()
    
    # Save tracking data
    output_path = project_root / output_csv
    df.to_csv(output_path, index=False)
    
    print("="*80)
    print("V2 ANALYSIS WITH YOLOv8x COMPLETE")
    print("="*80)
    print(f"✓ Saved tracking data to: {output_path}")
    print()
    
    if len(large_jumps) > 0:
        print(f"💡 V2 has {len(large_jumps)} spatial jumps")
        print("   → Will apply spatial re-ordering in next step")
    else:
        print("✓ V2 has excellent spatial continuity!")
        print("   → May not need refinement")
    
    print()
    print("Next step: Run 3_spatial_reorder.py")
    print()
    
    return True

def main():
    project_root = Path(__file__).parent.parent.parent
    v2_video = project_root / "output" / "reconstructed_video_cnn.mp4"
    
    if not v2_video.exists():
        print(f"✗ Error: V2 video not found: {v2_video}")
        print("  Please run 1_run_v2_cnn.py first")
        return
    
    analyze_and_refine_v2(v2_video)

if __name__ == "__main__":
    main()
