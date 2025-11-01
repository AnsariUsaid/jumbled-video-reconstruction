#!/usr/bin/env python3
"""
V5 Hybrid Approach - Step 3: Spatial re-ordering using YOLO nearest-neighbor
Applies V4's spatial ordering algorithm to V2's frames.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def spatial_reorder(csv_path, output_csv="correct_frame_order_v6.csv"):
    """
    Apply V4's spatial nearest-neighbor ordering to V2's frames.
    """
    print("="*80)
    print("V6 HYBRID - STEP 3: SPATIAL RE-ORDERING")
    print("="*80)
    print()
    print("Applying V4's spatial nearest-neighbor algorithm to V2's frames...")
    print("This combines V2's semantic ordering with YOLOv8x spatial optimization.")
    print()
    
    # Load tracking data from V2 (with YOLOv8x)
    print(f"Loading V2 tracking data (YOLOv8x): {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} frames")
    print()
    
    # Remove frames without detections
    valid = df.dropna(subset=["centroid_x", "centroid_y"])
    removed = len(df) - len(valid)
    
    if removed > 0:
        print(f"⚠ Removed {removed} frames without detections")
        print(f"  Working with {len(valid)} frames")
    print()
    
    # Apply V4's spatial ordering: start from bottom-right, nearest-neighbor
    print("Finding starting frame (closest to bottom-right)...")
    max_x = valid["centroid_x"].max()
    max_y = valid["centroid_y"].max()
    valid["dist_br"] = np.sqrt((max_x - valid["centroid_x"])**2 + (max_y - valid["centroid_y"])**2)
    
    start_idx = valid["dist_br"].idxmin()
    start_frame = int(valid.loc[start_idx, "frame"])
    
    print(f"✓ Starting frame: {start_frame}")
    print(f"  Position: ({valid.loc[start_idx, 'centroid_x']:.1f}, {valid.loc[start_idx, 'centroid_y']:.1f})")
    print()
    
    # Greedy nearest-neighbor ordering (V4 algorithm)
    print("Applying spatial nearest-neighbor ordering...")
    ordered_frames = [start_frame]
    remaining = valid.drop(index=start_idx).copy()
    current_x = valid.loc[start_idx, "centroid_x"]
    current_y = valid.loc[start_idx, "centroid_y"]
    
    total_distance = 0
    distances_list = []
    
    pbar = tqdm(total=len(remaining), desc="Ordering frames")
    
    while not remaining.empty:
        # Find nearest frame
        distances = np.sqrt(
            (remaining["centroid_x"] - current_x)**2 + 
            (remaining["centroid_y"] - current_y)**2
        )
        next_idx = distances.idxmin()
        min_dist = distances.min()
        
        ordered_frames.append(int(remaining.loc[next_idx, "frame"]))
        current_x = remaining.loc[next_idx, "centroid_x"]
        current_y = remaining.loc[next_idx, "centroid_y"]
        remaining = remaining.drop(index=next_idx)
        
        total_distance += min_dist
        distances_list.append(min_dist)
        pbar.update(1)
    
    pbar.close()
    
    # Calculate statistics
    avg_dist = total_distance / len(distances_list) if distances_list else 0
    max_dist = max(distances_list) if distances_list else 0
    min_dist = min(distances_list) if distances_list else 0
    large_jumps = sum(1 for d in distances_list if d > 30)
    
    # Save ordered frames
    order_df = pd.DataFrame({"ordered_frame": ordered_frames})
    output_path = project_root / output_csv
    order_df.to_csv(output_path, index=False)
    
    print()
    print("="*80)
    print("V6 HYBRID SPATIAL ORDERING COMPLETE")
    print("="*80)
    print(f"✓ Ordered {len(ordered_frames)} frames")
    print(f"✓ Total path distance: {total_distance:.1f} pixels")
    print(f"✓ Average step distance: {avg_dist:.1f} pixels")
    print(f"✓ Min step: {min_dist:.1f} pixels")
    print(f"✓ Max step: {max_dist:.1f} pixels")
    print(f"✓ Large jumps (>30px): {large_jumps}/{len(distances_list)} ({large_jumps/len(distances_list)*100:.1f}%)")
    print(f"✓ Saved frame order to: {output_path}")
    print()
    print("This order combines:")
    print("  • V2's semantic understanding (CNN features)")
    print("  • YOLOv8x spatial optimization (nearest-neighbor)")
    print()
    print("Next step: Run 4_reconstruct_v6.py")
    print()
    
    return True

def main():
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "frame_tracking_v6_hybrid.csv"
    
    if not csv_path.exists():
        print(f"✗ Error: Tracking data not found: {csv_path}")
        print("  Please run 2_apply_yolov8x_refinement.py first")
        return
    
    spatial_reorder(csv_path)

if __name__ == "__main__":
    main()
