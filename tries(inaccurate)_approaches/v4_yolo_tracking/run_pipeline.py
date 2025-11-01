#!/usr/bin/env python3
"""
Complete pipeline for V4 (YOLO + Nearest Neighbor) approach.
Runs all three steps: tracking → ordering → reconstruction.
"""

import sys
import time
from pathlib import Path

def main():
    print("="*80)
    print("V4 (YOLO + NEAREST NEIGHBOR) COMPLETE PIPELINE")
    print("="*80)
    print()
    print("This pipeline uses:")
    print("  1. YOLOv8 for person detection and tracking")
    print("  2. Nearest-neighbor greedy algorithm for frame ordering")
    print("  3. Standard video reconstruction")
    print()
    print("="*80)
    print()
    
    overall_start = time.time()
    
    # Step 1: Extract Tracking Data
    print("\n" + "="*80)
    print("STEP 1: EXTRACT TRACKING DATA WITH YOLO")
    print("="*80)
    step_start = time.time()
    
    try:
        from src.v4_yolo_tracking import extract_tracking_data as step1
        # Import and run will execute the main() function
        import importlib
        module1 = importlib.import_module('src.v4_yolo_tracking.1_extract_tracking_data')
        module1.main()
    except Exception as e:
        print(f"✗ Error in Step 1: {e}")
        import traceback
        traceback.print_exc()
        return
    
    step1_time = time.time() - step_start
    print(f"Step 1 completed in {step1_time:.2f} seconds")
    
    # Step 2: Compute Frame Order
    print("\n" + "="*80)
    print("STEP 2: COMPUTE FRAME ORDER")
    print("="*80)
    step_start = time.time()
    
    try:
        module2 = importlib.import_module('src.v4_yolo_tracking.2_compute_frame_order')
        module2.main()
    except Exception as e:
        print(f"✗ Error in Step 2: {e}")
        import traceback
        traceback.print_exc()
        return
    
    step2_time = time.time() - step_start
    print(f"Step 2 completed in {step2_time:.2f} seconds")
    
    # Step 3: Reconstruct Video
    print("\n" + "="*80)
    print("STEP 3: RECONSTRUCT VIDEO")
    print("="*80)
    step_start = time.time()
    
    try:
        module3 = importlib.import_module('src.v4_yolo_tracking.3_reconstruct_video')
        module3.main()
    except Exception as e:
        print(f"✗ Error in Step 3: {e}")
        import traceback
        traceback.print_exc()
        return
    
    step3_time = time.time() - step_start
    print(f"Step 3 completed in {step3_time:.2f} seconds")
    
    # Summary
    total_time = time.time() - overall_start
    print("\n" + "="*80)
    print("V4 PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"\n✓ Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  - Step 1 (YOLO Tracking): {step1_time:.2f}s")
    print(f"  - Step 2 (Frame Ordering): {step2_time:.2f}s")
    print(f"  - Step 3 (Video Reconstruction): {step3_time:.2f}s")
    print()
    print("🎥 Output: output/reconstructed_video_v4.mp4")
    print()
    print("Compare with other approaches:")
    print("  - V1 (ORB): output/reconstructed_video.mp4")
    print("  - V2 (CNN): output/reconstructed_video_cnn.mp4")
    print("  - V4 (YOLO): output/reconstructed_video_v4.mp4")
    print()
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
