#!/usr/bin/env python3
"""
V5 Hybrid Approach - Step 4: Reconstruct final V5 video
Uses spatial ordering on V2's frames to create V5.
"""

import cv2
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def reconstruct_v6_hybrid(v2_video_path, order_csv, output_video="output/reconstructed_video_v6.mp4"):
    """
    Reconstruct V6 by reordering V2's CNN frames using YOLOv8x spatial order.
    """
    print("="*80)
    print("V6 HYBRID - STEP 4: RECONSTRUCT FINAL VIDEO")
    print("="*80)
    print()
    
    # Load frame order
    print(f"Loading spatial frame order: {order_csv}")
    order_df = pd.read_csv(order_csv)
    order = order_df["ordered_frame"].tolist()
    print(f"✓ Loaded {len(order)} frames in spatial order")
    print()
    
    # Open V2 video
    print(f"Reading V2 CNN video: {v2_video_path}")
    cap = cv2.VideoCapture(str(v2_video_path))
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open video")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video specs: {width}x{height} @ {fps} FPS")
    print()
    
    # Load all V2 frames
    print("Loading V2 frames...")
    frames = []
    for _ in tqdm(range(total_frames), desc="Reading"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"✓ Loaded {len(frames)} frames")
    print()
    
    # Reorder using spatial ordering
    print("Reordering with spatial optimization...")
    ordered_frames = []
    for i in tqdm(order, desc="Reordering"):
        if i < len(frames):
            ordered_frames.append(frames[i])
    
    print(f"✓ Reordered {len(ordered_frames)} frames")
    print()
    
    # Write V5 video
    output_path = project_root / output_video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing V5 hybrid video...")
    print(f"Output: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in tqdm(ordered_frames, desc="Writing"):
        out.write(frame)
    
    out.release()
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    duration = len(ordered_frames) / fps
    
    print()
    print("="*80)
    print("V6 HYBRID RECONSTRUCTION COMPLETE")
    print("="*80)
    print(f"✓ Frames: {len(ordered_frames)}")
    print(f"✓ Duration: {duration:.1f} seconds")
    print(f"✓ File size: {file_size:.1f} MB")
    print(f"✓ Output: {output_path}")
    print()
    print("V6 Hybrid combines:")
    print("  • V2's CNN semantic features (99.6% similarity)")
    print("  • YOLOv8x spatial optimization (more accurate than YOLOv8n)")
    print()
    print("🎥 View the video:")
    print(f"   open {output_path}")
    print()
    print("Compare all approaches:")
    print("  V1 (ORB): output/reconstructed_video.mp4")
    print("  V2 (CNN): output/reconstructed_video_cnn.mp4")
    print("  V4 (YOLOv8n): output/reconstructed_video_v4.mp4")
    print("  V5 (Hybrid v8n): output/reconstructed_video_v5.mp4")
    print("  V6 (Hybrid v8x): output/reconstructed_video_v6.mp4")
    print()
    
    return True

def main():
    project_root = Path(__file__).parent.parent.parent
    v2_video = project_root / "output" / "reconstructed_video_cnn.mp4"
    order_csv = project_root / "correct_frame_order_v6.csv"
    
    if not v2_video.exists():
        print(f"✗ Error: V2 video not found")
        return
    
    if not order_csv.exists():
        print(f"✗ Error: Frame order not found")
        print("  Please run 3_spatial_reorder.py first")
        return
    
    reconstruct_v6_hybrid(v2_video, order_csv)

if __name__ == "__main__":
    main()
