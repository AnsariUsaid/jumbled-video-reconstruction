import numpy as np
import pickle
import os


def load_v1_order(order_file):
    print(f"Loading V1 (ORB) ordering from: {order_file}")
    with open(order_file, 'rb') as f:
        return pickle.load(f)


def load_v2_similarity(matrix_file):
    print(f"Loading V2 (CNN) similarity matrix from: {matrix_file}")
    return np.load(matrix_file)


def refine_window_with_cnn(window_indices, cnn_similarity):
    """
    Refine a window of frames using CNN similarity.
    Use greedy nearest neighbor within the window.
    """
    if len(window_indices) <= 1:
        return window_indices
    
    refined = [window_indices[0]]
    remaining = set(window_indices[1:])
    
    while remaining:
        current = refined[-1]
        
        # Find most similar frame in remaining
        best_next = None
        best_similarity = -1
        
        for candidate in remaining:
            sim = cnn_similarity[current][candidate]
            if sim > best_similarity:
                best_similarity = sim
                best_next = candidate
        
        if best_next is not None:
            refined.append(best_next)
            remaining.remove(best_next)
        else:
            # Fallback: just add first remaining
            refined.append(remaining.pop())
    
    return refined


def sequential_refinement(v1_order, cnn_similarity, window_size=15):
    """
    Use V1 ordering as global structure, refine locally with V2 CNN similarity.
    """
    print("\n" + "=" * 60)
    print("V3: SEQUENTIAL REFINEMENT")
    print("=" * 60)
    
    print(f"\nStrategy:")
    print(f"  1. Use V1 (ORB) for global structure")
    print(f"  2. Refine within windows of {window_size} frames using V2 (CNN)")
    print(f"  3. Combine best of both approaches")
    
    indices = v1_order['frame_indices']
    total_frames = len(indices)
    
    print(f"\nV1 global order: {total_frames} frames")
    print(f"Window size: {window_size} frames")
    print(f"Number of windows: {(total_frames + window_size - 1) // window_size}")
    
    refined_order = []
    
    # Process in windows
    for start in range(0, total_frames, window_size):
        end = min(start + window_size, total_frames)
        window = indices[start:end]
        
        # Refine this window using CNN similarity
        refined_window = refine_window_with_cnn(window, cnn_similarity)
        refined_order.extend(refined_window)
    
    return refined_order


def calculate_ordering_quality(indices, similarity_matrix):
    """Calculate average similarity for consecutive frames."""
    similarities = []
    for i in range(len(indices) - 1):
        sim = similarity_matrix[indices[i]][indices[i + 1]]
        similarities.append(sim)
    
    return {
        'avg': np.mean(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'std': np.std(similarities),
        'median': np.median(similarities)
    }


def save_v3_order(indices, frames_data, output_file):
    """Save V3 ordering."""
    print(f"\nSaving V3 ordering to: {output_file}")
    
    order_data = {
        'frame_indices': indices,
        'frame_filenames': [frames_data[i]['filename'] for i in indices],
        'num_frames': len(indices)
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(order_data, f)
    
    print("V3 order saved successfully")
    return order_data


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    v1_order_file = os.path.join(project_root, "frame_order.pkl")
    v2_matrix_file = os.path.join(project_root, "similarity_matrix_cnn.npy")
    v2_features_file = os.path.join(project_root, "frames_features_cnn.pkl")
    output_file = os.path.join(project_root, "frame_order_v3.pkl")
    
    print("=" * 60)
    print("V3: HYBRID SEQUENTIAL REFINEMENT")
    print("=" * 60)
    print("\nApproach: V1 (global) + V2 (local refinement)")
    
    # Load V1 order
    v1_order = load_v1_order(v1_order_file)
    
    # Load V2 similarity matrix
    v2_similarity = load_v2_similarity(v2_matrix_file)
    
    # Load V2 features for frame info
    with open(v2_features_file, 'rb') as f:
        frames_data = pickle.load(f)
    
    # Calculate V1 quality using V2 similarity
    print("\nV1 ordering quality (using CNN similarity):")
    v1_quality = calculate_ordering_quality(v1_order['frame_indices'], v2_similarity)
    print(f"  Average: {v1_quality['avg']:.2f}")
    print(f"  Min: {v1_quality['min']:.2f}")
    print(f"  Std: {v1_quality['std']:.2f}")
    
    # Perform sequential refinement
    v3_indices = sequential_refinement(v1_order, v2_similarity, window_size=15)
    
    # Calculate V3 quality
    print("\nV3 refined ordering quality:")
    v3_quality = calculate_ordering_quality(v3_indices, v2_similarity)
    print(f"  Average: {v3_quality['avg']:.2f}")
    print(f"  Min: {v3_quality['min']:.2f}")
    print(f"  Std: {v3_quality['std']:.2f}")
    
    # Calculate improvement
    improvement = v3_quality['avg'] - v1_quality['avg']
    print(f"\nImprovement: {improvement:+.2f} ({improvement/v1_quality['avg']*100:+.2f}%)")
    
    # Save V3 order
    v3_order = save_v3_order(v3_indices, frames_data, output_file)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"V1 (ORB global):        {v1_quality['avg']:.2f}/1000")
    print(f"V2 (CNN full):          995.68/1000 (from earlier)")
    print(f"V3 (V1 + V2 refine):    {v3_quality['avg']:.2f}/1000")
    print("=" * 60)
    
    print(f"\n✓ V3 ordering complete!")
    print(f"  Output: {output_file}")
    print(f"\nNext: Run reconstruct_video_v3.py")


if __name__ == "__main__":
    main()
