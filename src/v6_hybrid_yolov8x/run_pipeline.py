#!/usr/bin/env python3
"""
Complete pipeline for V6 Hybrid (CNN + YOLOv8x) approach.
Combines V2's semantic understanding with YOLOv8x's more accurate spatial optimization.
"""

import sys
import time
import subprocess
from pathlib import Path

def main():
    print("="*80)
    print("V6 HYBRID (CNN + YOLOv8x) COMPLETE PIPELINE")
    print("="*80)
    print()
    print("This hybrid approach uses YOLOv8x (extra-large) instead of YOLOv8n:")
    print()
    print("  V2 (CNN ResNet50):")
    print("    ✓ Semantic image understanding")
    print("    ✓ 99.6% similarity score")
    print("    ✓ Good global ordering")
    print()
    print("  YOLOv8x (Extra-Large):")
    print("    ✓ More accurate than YOLOv8n (nano)")
    print("    ✓ Better person detection")
    print("    ✓ More precise bounding boxes")
    print("    ⚠ Slower but more accurate")
    print()
    print("  V6 Hybrid Strategy:")
    print("    1. Use V2's CNN-ordered frames as input")
    print("    2. Extract YOLOv8x positions from V2 output")
    print("    3. Apply spatial nearest-neighbor with better detections")
    print("    4. Create refined V6 video")
    print()
    print("="*80)
    print()
    
    overall_start = time.time()
    script_dir = Path(__file__).parent
    
    # Step 1: Run V2 CNN (if not already done)
    print("\n" + "="*80)
    print("STEP 1: RUN V2 (CNN) FOR SEMANTIC ORDERING")
    print("="*80)
    
    project_root = Path(__file__).parent.parent.parent
    v2_output = project_root / "output" / "reconstructed_video_cnn.mp4"
    
    if v2_output.exists():
        file_size = v2_output.stat().st_size / (1024 * 1024)
        print(f"✓ V2 output already exists: {v2_output}")
        print(f"  File size: {file_size:.1f} MB")
        print("  Skipping V2 pipeline (already complete)")
        step1_time = 0
    else:
        step_start = time.time()
        result = subprocess.run([
            sys.executable,
            str(script_dir / "1_run_v2_cnn.py")
        ])
        
        if result.returncode != 0:
            print("✗ Step 1 failed")
            return
        
        step1_time = time.time() - step_start
    
    print(f"Step 1 completed in {step1_time:.2f} seconds")
    
    # Step 2: Apply YOLOv8x refinement
    print("\n" + "="*80)
    print("STEP 2: YOLOv8x SPATIAL ANALYSIS ON V2 OUTPUT")
    print("="*80)
    print("Note: This will be slower than YOLOv8n but more accurate")
    step_start = time.time()
    
    result = subprocess.run([
        sys.executable,
        str(script_dir / "2_apply_yolov8x_refinement.py")
    ])
    
    if result.returncode != 0:
        print("✗ Step 2 failed")
        return
    
    step2_time = time.time() - step_start
    print(f"Step 2 completed in {step2_time:.2f} seconds")
    
    # Step 3: Spatial re-ordering
    print("\n" + "="*80)
    print("STEP 3: SPATIAL NEAREST-NEIGHBOR ORDERING")
    print("="*80)
    step_start = time.time()
    
    result = subprocess.run([
        sys.executable,
        str(script_dir / "3_spatial_reorder.py")
    ])
    
    if result.returncode != 0:
        print("✗ Step 3 failed")
        return
    
    step3_time = time.time() - step_start
    print(f"Step 3 completed in {step3_time:.2f} seconds")
    
    # Step 4: Reconstruct V6
    print("\n" + "="*80)
    print("STEP 4: RECONSTRUCT V6 HYBRID VIDEO")
    print("="*80)
    step_start = time.time()
    
    result = subprocess.run([
        sys.executable,
        str(script_dir / "4_reconstruct_v6.py")
    ])
    
    if result.returncode != 0:
        print("✗ Step 4 failed")
        return
    
    step4_time = time.time() - step_start
    print(f"Step 4 completed in {step4_time:.2f} seconds")
    
    # Summary
    total_time = time.time() - overall_start
    print("\n" + "="*80)
    print("V6 HYBRID PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✓ Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  - Step 1 (V2 CNN): {step1_time:.2f}s")
    print(f"  - Step 2 (YOLOv8x Analysis): {step2_time:.2f}s")
    print(f"  - Step 3 (Spatial Ordering): {step3_time:.2f}s")
    print(f"  - Step 4 (Reconstruction): {step4_time:.2f}s")
    print()
    print("🎥 Output: output/reconstructed_video_v6.mp4")
    print()
    print("V6 Hybrid combines:")
    print("  ✓ V2's semantic understanding (CNN features)")
    print("  ✓ YOLOv8x more accurate spatial continuity")
    print()
    print("Compare with V5 (YOLOv8n) to see if extra accuracy helps!")
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
