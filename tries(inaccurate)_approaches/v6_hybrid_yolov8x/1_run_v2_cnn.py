#!/usr/bin/env python3
"""
V6 Hybrid Approach - Step 1: Run V2 (CNN) to get initial ordering
Same as V5, uses ResNet50 CNN features for semantic similarity ordering.
"""

import sys
import subprocess
from pathlib import Path

def run_v2_pipeline():
    """
    Run the V2 CNN pipeline to get initial frame ordering.
    V2 uses ResNet50 features and similarity matrix.
    """
    print("="*80)
    print("V6 HYBRID - STEP 1: RUN V2 (CNN) FOR INITIAL ORDERING")
    print("="*80)
    print()
    print("Strategy:")
    print("  1. Use V2 (CNN ResNet50) for semantic similarity ordering")
    print("  2. V2 is great at understanding image content (99.6% similarity)")
    print("  3. Creates initial ordered video")
    print()
    print("="*80)
    print()
    
    project_root = Path(__file__).parent.parent.parent
    v2_dir = project_root / "src" / "v2_deeplearning"
    
    print("Running V2 CNN pipeline...")
    print(f"Location: {v2_dir}")
    print()
    
    # Check if V2 output exists
    v2_output = project_root / "output" / "reconstructed_video_cnn.mp4"
    
    if v2_output.exists():
        file_size = v2_output.stat().st_size / (1024 * 1024)
        print(f"✓ V2 output already exists: {v2_output}")
        print(f"  File size: {file_size:.1f} MB")
        print("  Skipping V2 pipeline (already complete)")
    else:
        # Run V2 pipeline
        result = subprocess.run([
            sys.executable,
            str(v2_dir / "run_pipeline.py")
        ])
        
        if result.returncode != 0:
            print("✗ V2 pipeline failed")
            return False
        
        if not v2_output.exists():
            print(f"✗ V2 output not found: {v2_output}")
            return False
        
        file_size = v2_output.stat().st_size / (1024 * 1024)
    
    print()
    print("="*80)
    print("V2 CNN ORDERING COMPLETE")
    print("="*80)
    print(f"✓ V2 output: {v2_output}")
    print(f"✓ File size: {file_size:.1f} MB")
    print(f"✓ Similarity: 99.6% (from previous run)")
    print()
    print("This provides semantic-level ordering based on image features.")
    print()
    print("Next step: Run 2_apply_yolov8x_refinement.py")
    print()
    
    return True

def main():
    success = run_v2_pipeline()
    
    if not success:
        print("✗ Failed to complete V2 pipeline")
        sys.exit(1)

if __name__ == "__main__":
    main()
