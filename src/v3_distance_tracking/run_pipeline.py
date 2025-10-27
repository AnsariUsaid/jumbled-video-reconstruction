import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.v3_distance_tracking.detect_and_track import detect_and_track_person
from src.v3_distance_tracking.order_by_centroid import order_frames_by_centroid
import cv2
import pickle

def run_v3_pipeline():
    print("="*60)
    print("V3: Distance-Based Person Tracking Pipeline")
    print("="*60)
    
    start_time = time.time()
    
    frames_dir = "frames"
    output_video = "output/reconstructed_v3.mp4"
    
    print("\nPhase 1: Person Detection and Tracking")
    print("-" * 60)
    phase1_start = time.time()
    centroids = detect_and_track_person(frames_dir)
    phase1_time = time.time() - phase1_start
    print(f"Phase 1 completed in {phase1_time:.2f} seconds")
    
    with open('person_centroids.pkl', 'wb') as f:
        pickle.dump(centroids, f)
    print(f"Saved {len(centroids)} centroids to person_centroids.pkl")
    
    print("\nPhase 2: Frame Ordering by Distance")
    print("-" * 60)
    phase2_start = time.time()
    frame_order = order_frames_by_centroid(centroids)
    phase2_time = time.time() - phase2_start
    print(f"Phase 2 completed in {phase2_time:.2f} seconds")
    
    with open('frame_order_v3.pkl', 'wb') as f:
        pickle.dump(frame_order, f)
    print(f"Saved frame order to frame_order_v3.pkl")
    
    print("\nPhase 3: Video Reconstruction")
    print("-" * 60)
    phase3_start = time.time()
    
    frame_files = sorted([f for f in Path(frames_dir).glob("*.jpg")])
    if not frame_files:
        print("Error: No frames found!")
        return
    
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for idx in frame_order:
        frame_path = Path(frames_dir) / f"frame_{idx:04d}.jpg"
        if frame_path.exists():
            frame = cv2.imread(str(frame_path))
            out.write(frame)
    
    out.release()
    phase3_time = time.time() - phase3_start
    print(f"Phase 3 completed in {phase3_time:.2f} seconds")
    print(f"Reconstructed video saved to: {output_video}")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("Pipeline Summary")
    print("="*60)
    print(f"Phase 1 (Detection): {phase1_time:.2f}s")
    print(f"Phase 2 (Ordering):  {phase2_time:.2f}s")
    print(f"Phase 3 (Reconstruction): {phase3_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Frames Ordered: {len(frame_order)}")
    print("="*60)

if __name__ == "__main__":
    run_v3_pipeline()
