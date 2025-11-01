#!/usr/bin/env python3
"""
V4 Approach - Step 3: Reconstruct Video
Rebuilds video using the computed frame order.
"""

import cv2
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def reconstruct_video(video_path, order_csv, output_video="output/reconstructed_video_v4.mp4"):
    """
    Reconstruct video using ordered frames.
    
    Args:
        video_path: Path to original jumbled video
        order_csv: Path to frame order CSV
        output_video: Path to save reconstructed video
    """
    print("="*80)
    print("V4 APPROACH - STEP 3: VIDEO RECONSTRUCTION")
    print("="*80)
    print()
    
    # Load frame order
    print(f"Loading frame order from: {order_csv}")
    order_df = pd.read_csv(order_csv)
    order = order_df["ordered_frame"].tolist()
    print(f"✓ Loaded {len(order)} frames in sequence")
    print()
    
    # Open video and read all frames
    print(f"Reading frames from: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video specs: {width}x{height} @ {fps} FPS")
    print(f"Total frames in video: {total_frames}")
    print()
    
    print("Loading all frames into memory...")
    frames = []
    for _ in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"✓ Loaded {len(frames)} frames")
    print()
    
    # Reorder frames
    print("Reordering frames...")
    ordered_frames = []
    missing_frames = 0
    
    for i in tqdm(order, desc="Reordering"):
        if i < len(frames):
            ordered_frames.append(frames[i])
        else:
            missing_frames += 1
            print(f"⚠ Warning: Frame {i} out of range (max: {len(frames)-1})")
    
    if missing_frames > 0:
        print(f"⚠ {missing_frames} frames were out of range and skipped")
    print()
    
    # Create output directory
    output_path = project_root / output_video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write reconstructed video
    print(f"Writing reconstructed video...")
    print(f"Output: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("✗ Error: Could not create video writer")
        return False
    
    for frame in tqdm(ordered_frames, desc="Writing frames"):
        out.write(frame)
    
    out.release()
    
    # Get file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    duration = len(ordered_frames) / fps
    
    print()
    print("="*80)
    print("VIDEO RECONSTRUCTION COMPLETE")
    print("="*80)
    print(f"✓ Reconstructed {len(ordered_frames)} frames")
    print(f"✓ Duration: {duration:.1f} seconds")
    print(f"✓ File size: {file_size:.1f} MB")
    print(f"✓ Output: {output_path}")
    print()
    print("🎥 View the video:")
    print(f"   open {output_path}")
    print()
    
    return True

def main():
    project_root = Path(__file__).parent.parent.parent
    video_path = project_root / "jumbled_video.mp4"
    order_csv = project_root / "correct_frame_order.csv"
    
    if not video_path.exists():
        print(f"✗ Error: Video not found: {video_path}")
        return
    
    if not order_csv.exists():
        print(f"✗ Error: Frame order not found: {order_csv}")
        print("  Please run 2_compute_frame_order.py first")
        return
    
    success = reconstruct_video(video_path, order_csv)
    
    if success:
        print("="*80)
        print("V4 PIPELINE COMPLETE!")
        print("="*80)

if __name__ == "__main__":
    main()
