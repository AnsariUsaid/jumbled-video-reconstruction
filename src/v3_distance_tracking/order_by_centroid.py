import numpy as np
import pickle
from pathlib import Path
import time

def compute_distance_from_bottom_right(centroids, frame_height=1080, frame_width=1920):
    distances = []
    
    bottom_right = (frame_width, frame_height)
    
    for centroid in centroids:
        if centroid is None:
            distances.append(None)
        else:
            dx = centroid[0] - bottom_right[0]
            dy = centroid[1] - bottom_right[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
    
    return distances

def order_frames_by_distance(frame_data, distances):
    frame_distance_pairs = []
    
    for i, (data, dist) in enumerate(zip(frame_data, distances)):
        if dist is not None:
            frame_distance_pairs.append((i, data['frame_file'], dist, data['interpolated_centroid']))
    
    frame_distance_pairs.sort(key=lambda x: x[2])
    
    ordered_frames = [frame_file for _, frame_file, _, _ in frame_distance_pairs]
    
    return ordered_frames, frame_distance_pairs

def main():
    start_time = time.time()
    
    input_file = "person_centroids.pkl"
    output_file = "frame_order_v3_centroid.pkl"
    
    print("Loading centroid data...")
    with open(input_file, 'rb') as f:
        frame_data = pickle.load(f)
    
    centroids = [d['interpolated_centroid'] for d in frame_data]
    
    print("Computing distances from bottom-right corner...")
    distances = compute_distance_from_bottom_right(centroids)
    
    print("Ordering frames by distance...")
    ordered_frames, frame_distance_pairs = order_frames_by_distance(frame_data, distances)
    
    print(f"\nOrdered {len(ordered_frames)} frames")
    print(f"First 10 frames in order:")
    for i in range(min(10, len(frame_distance_pairs))):
        idx, frame_file, dist, centroid = frame_distance_pairs[i]
        print(f"  {frame_file}: distance={dist:.2f}, centroid={centroid}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(ordered_frames, f)
    
    elapsed = time.time() - start_time
    print(f"\nSaved frame order to {output_file}")
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
