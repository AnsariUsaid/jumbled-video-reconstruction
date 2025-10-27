#!/usr/bin/env python3
"""
Complete pipeline for V2 (CNN) approach.
Runs all phases from frame extraction to video reconstruction using deep learning features.
"""

import os
import time
import sys

def main():
    print("=" * 80)
    print("V2 (CNN) COMPLETE PIPELINE EXECUTION")
    print("=" * 80)
    print()
    
    overall_start = time.time()
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Phase 1: Setup Verification
    print("Phase 1: Project Setup Verification")
    print("-" * 80)
    video_path = os.path.join(project_root, "jumbled_video.mp4")
    frames_dir = os.path.join(project_root, "frames")
    output_dir = os.path.join(project_root, "output")
    
    if not os.path.exists(video_path):
        print(f"✗ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        print(f"✓ Created frames directory: {frames_dir}")
    else:
        print(f"✓ Frames directory exists: {frames_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    else:
        print(f"✓ Output directory exists: {output_dir}")
    
    print()
    
    # Phase 2: Extract Frames (if needed)
    print("Phase 2: Frame Extraction")
    print("-" * 80)
    phase_start = time.time()
    
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    if len(frame_files) > 0:
        print(f"✓ Frames already extracted: {len(frame_files)} frames found")
        print(f"  Skipping extraction...")
        frame_count = len(frame_files)
    else:
        from extract_frames import extract_frames
        frame_count, fps, resolution = extract_frames(video_path, frames_dir)
        print(f"✓ Extracted {frame_count} frames")
        print(f"  Resolution: {resolution}, FPS: {fps}")
    
    phase_time = time.time() - phase_start
    print(f"✓ Phase 2 completed in {phase_time:.2f} seconds")
    print()
    
    # Phase 3: Extract CNN Features
    print("Phase 3: CNN Feature Extraction (ResNet50)")
    print("-" * 80)
    phase_start = time.time()
    
    features_file = os.path.join(project_root, "frames_features_cnn.pkl")
    if os.path.exists(features_file):
        print(f"⚠ CNN features already exist: {features_file}")
        print(f"  To re-extract, delete the file first.")
        response = input("  Skip feature extraction? (y/n): ").lower()
        if response != 'y':
            print("  Re-extracting features...")
            os.remove(features_file)
    
    if not os.path.exists(features_file):
        print("  Loading ResNet50 model...")
        print("  Extracting deep features from frames...")
        print("  (This may take 1-2 minutes)")
        
        # Import and run extraction
        import extract_features_cnn
        # The module will extract and save features
        
    print(f"✓ CNN features saved: {features_file}")
    phase_time = time.time() - phase_start
    print(f"✓ Phase 3 completed in {phase_time:.2f} seconds")
    print()
    
    # Phase 4: Build Similarity Matrix
    print("Phase 4: Similarity Matrix Construction (Cosine Similarity)")
    print("-" * 80)
    phase_start = time.time()
    
    similarity_file = os.path.join(project_root, "similarity_matrix_cnn.npy")
    if os.path.exists(similarity_file):
        print(f"⚠ Similarity matrix already exists: {similarity_file}")
        print(f"  To rebuild, delete the file first.")
        response = input("  Skip similarity matrix? (y/n): ").lower()
        if response != 'y':
            print("  Rebuilding similarity matrix...")
            os.remove(similarity_file)
    
    if not os.path.exists(similarity_file):
        print("  Computing cosine similarities between all frame pairs...")
        
        # Import and run similarity matrix building
        import build_similarity_matrix_cnn
        # The module will build and save the matrix
    
    print(f"✓ Similarity matrix saved: {similarity_file}")
    phase_time = time.time() - phase_start
    print(f"✓ Phase 4 completed in {phase_time:.2f} seconds")
    print()
    
    # Phase 5: Order Frames
    print("Phase 5: Frame Order Optimization (Graph-based)")
    print("-" * 80)
    phase_start = time.time()
    
    order_file = os.path.join(project_root, "frame_order_cnn.pkl")
    if os.path.exists(order_file):
        print(f"⚠ Frame order already exists: {order_file}")
        print(f"  To recompute, delete the file first.")
        response = input("  Skip frame ordering? (y/n): ").lower()
        if response != 'y':
            print("  Recomputing frame order...")
            os.remove(order_file)
    
    if not os.path.exists(order_file):
        print("  Finding optimal frame sequence...")
        
        # Import and run frame ordering
        import order_frames_improved
        # The module will compute and save the order
    
    print(f"✓ Frame order saved: {order_file}")
    phase_time = time.time() - phase_start
    print(f"✓ Phase 5 completed in {phase_time:.2f} seconds")
    print()
    
    # Phase 6: Reconstruct Video
    print("Phase 6: Video Reconstruction")
    print("-" * 80)
    phase_start = time.time()
    
    output_video = os.path.join(output_dir, "reconstructed_video_cnn.mp4")
    print("  Building video from ordered frames...")
    
    # Import and run video reconstruction
    import reconstruct_video
    # The module will reconstruct the video
    
    if os.path.exists(output_video):
        video_size = os.path.getsize(output_video) / (1024 * 1024)
        print(f"✓ Video reconstructed: {output_video}")
        print(f"  File size: {video_size:.1f} MB")
    
    phase_time = time.time() - phase_start
    print(f"✓ Phase 6 completed in {phase_time:.2f} seconds")
    print()
    
    # Summary
    total_time = time.time() - overall_start
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"✓ Input: {frame_count} jumbled frames")
    print(f"✓ Output: {output_video}")
    print(f"✓ Method: CNN (ResNet50) with Cosine Similarity")
    print()
    print("🎥 Next steps:")
    print("  1. View the reconstructed video: open output/reconstructed_video_cnn.mp4")
    print("  2. Compare with V1 (ORB): open output/reconstructed_video.mp4")
    print("  3. Check quality metrics with: cd ../comparison && python compare_results.py")
    print()
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
