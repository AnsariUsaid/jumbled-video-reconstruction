#!/usr/bin/env python3
"""
V4 Approach - Step 2: Compute Frame Order
Uses nearest-neighbor greedy algorithm starting from bottom-right position.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def compute_frame_order(csv_path, output_csv="correct_frame_order.csv"):
    """
    Compute optimal frame order using nearest-neighbor greedy search.
    Starts from frame closest to bottom-right corner.
    
    Args:
        csv_path: Path to tracking data CSV
        output_csv: Path to save ordered frames CSV
    """
    print("="*80)
    print("V4 APPROACH - STEP 2: COMPUTE FRAME ORDER")
    print("="*80)
    print()
    
    # Load tracking data
    print(f"Loading tracking data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} frames")
    print()
    
    # Remove frames without detections
    original_count = len(df)
    df = df.dropna(subset=["centroid_x", "centroid_y"])
    removed_count = original_count - len(df)
    
    if removed_count > 0:
        print(f"⚠ Removed {removed_count} frames with missing detections")
        print(f"  Working with {len(df)} frames")
    print()
    
    if len(df) == 0:
        print("✗ Error: No valid frames with detections!")
        return False
    
    # Calculate bottom-right distance
    print("Finding starting frame (closest to bottom-right corner)...")
    max_x = df["centroid_x"].max()
    max_y = df["centroid_y"].max()
    print(f"  Bottom-right corner: ({max_x:.1f}, {max_y:.1f})")
    
    df["distance_from_bottom_right"] = np.sqrt(
        (max_x - df["centroid_x"])**2 + (max_y - df["centroid_y"])**2
    )
    
    # Start from frame closest to bottom-right
    start_idx = df["distance_from_bottom_right"].idxmin()
    start_frame = int(df.loc[start_idx, "frame"])
    start_x, start_y = df.loc[start_idx, ["centroid_x", "centroid_y"]]
    start_dist = df.loc[start_idx, "distance_from_bottom_right"]
    
    print(f"✓ Starting frame: {start_frame}")
    print(f"  Position: ({start_x:.1f}, {start_y:.1f})")
    print(f"  Distance from corner: {start_dist:.1f} pixels")
    print()
    
    # Greedy nearest-neighbor ordering
    print("Building frame sequence using nearest-neighbor greedy search...")
    ordered_frames = [start_frame]
    remaining = df.drop(index=start_idx).copy()
    current_x, current_y = start_x, start_y
    
    from tqdm import tqdm
    pbar = tqdm(total=len(remaining), desc="Ordering frames")
    
    total_distance = 0
    distances_list = []
    
    while not remaining.empty:
        # Find nearest frame to current position
        distances = np.sqrt(
            (remaining["centroid_x"] - current_x)**2 + 
            (remaining["centroid_y"] - current_y)**2
        )
        next_idx = distances.idxmin()
        min_distance = distances.min()
        
        ordered_frames.append(int(remaining.loc[next_idx, "frame"]))
        current_x, current_y = remaining.loc[next_idx, ["centroid_x", "centroid_y"]]
        remaining = remaining.drop(index=next_idx)
        
        total_distance += min_distance
        distances_list.append(min_distance)
        pbar.update(1)
    
    pbar.close()
    
    # Save ordered frames
    order_df = pd.DataFrame({"ordered_frame": ordered_frames})
    output_path = project_root / output_csv
    order_df.to_csv(output_path, index=False)
    
    # Statistics
    avg_distance = total_distance / len(distances_list) if distances_list else 0
    max_distance = max(distances_list) if distances_list else 0
    min_distance = min(distances_list) if distances_list else 0
    
    print()
    print("="*80)
    print("FRAME ORDERING COMPLETE")
    print("="*80)
    print(f"✓ Ordered {len(ordered_frames)} frames")
    print(f"✓ Total path distance: {total_distance:.1f} pixels")
    print(f"✓ Average step distance: {avg_distance:.1f} pixels")
    print(f"✓ Min step distance: {min_distance:.1f} pixels")
    print(f"✓ Max step distance: {max_distance:.1f} pixels")
    print(f"✓ Saved frame order to: {output_path}")
    print()
    
    return True

def main():
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "frame_tracking_data.csv"
    
    if not csv_path.exists():
        print(f"✗ Error: Tracking data not found: {csv_path}")
        print("  Please run 1_extract_tracking_data.py first")
        return
    
    success = compute_frame_order(csv_path)
    
    if success:
        print("Next step: Run 3_reconstruct_video.py")

if __name__ == "__main__":
    main()
