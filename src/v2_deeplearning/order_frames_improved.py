import numpy as np
import pickle
import os
from tqdm import tqdm


def load_similarity_matrix(matrix_file):
    print(f"Loading similarity matrix from: {matrix_file}")
    try:
        matrix = np.load(matrix_file)
        print(f"Successfully loaded {matrix.shape[0]}x{matrix.shape[1]} similarity matrix")
        return matrix
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return None


def load_frames_data(features_file):
    print(f"Loading frame data from: {features_file}")
    try:
        with open(features_file, 'rb') as f:
            frames_data = pickle.load(f)
        print(f"Successfully loaded data for {len(frames_data)} frames")
        return frames_data
    except Exception as e:
        print(f"Error loading frame data: {e}")
        return None


def find_best_starting_pair(similarity_matrix):
    n = len(similarity_matrix)
    max_similarity = -1
    best_pair = (0, 1)
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] > max_similarity:
                max_similarity = similarity_matrix[i][j]
                best_pair = (i, j)
    
    return best_pair, max_similarity


def build_path_nearest_neighbor(similarity_matrix, start_idx):
    n = len(similarity_matrix)
    visited = [False] * n
    path = [start_idx]
    visited[start_idx] = True
    current = start_idx
    
    for _ in range(n - 1):
        max_sim = -1
        next_frame = -1
        
        for j in range(n):
            if not visited[j] and similarity_matrix[current][j] > max_sim:
                max_sim = similarity_matrix[current][j]
                next_frame = j
        
        if next_frame == -1:
            for j in range(n):
                if not visited[j]:
                    next_frame = j
                    break
        
        path.append(next_frame)
        visited[next_frame] = True
        current = next_frame
    
    return path


def calculate_path_score(path, similarity_matrix):
    score = 0
    for i in range(len(path) - 1):
        score += similarity_matrix[path[i]][path[i + 1]]
    return score


def optimize_path_2opt(path, similarity_matrix, max_iterations=1000):
    n = len(path)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                
                old_score = (similarity_matrix[path[i-1]][path[i]] + 
                           similarity_matrix[path[j-1]][path[j % n]])
                new_score = (similarity_matrix[new_path[i-1]][new_path[i]] + 
                           similarity_matrix[new_path[j-1]][new_path[j % n]])
                
                if new_score > old_score:
                    path = new_path
                    improved = True
                    break
            if improved:
                break
        iteration += 1
    
    return path, iteration


def find_optimal_path_graph_approach(similarity_matrix):
    print("\nFinding optimal frame order using graph-based approach...")
    
    print("Step 1: Finding best starting pair (highest similarity frames)")
    best_pair, max_sim = find_best_starting_pair(similarity_matrix)
    print(f"  Best pair: Frame {best_pair[0]} and {best_pair[1]} (similarity: {max_sim})")
    
    print("\nStep 2: Building initial path with nearest neighbor from both directions")
    path1 = build_path_nearest_neighbor(similarity_matrix, best_pair[0])
    score1 = calculate_path_score(path1, similarity_matrix)
    
    path2 = build_path_nearest_neighbor(similarity_matrix, best_pair[1])
    score2 = calculate_path_score(path2, similarity_matrix)
    
    print(f"  Path from frame {best_pair[0]}: score = {score1}")
    print(f"  Path from frame {best_pair[1]}: score = {score2}")
    
    if score1 > score2:
        best_path = path1
        best_score = score1
        print(f"  Selected path starting from frame {best_pair[0]}")
    else:
        best_path = path2
        best_score = score2
        print(f"  Selected path starting from frame {best_pair[1]}")
    
    print(f"\nStep 3: Applying 2-opt optimization to improve path...")
    optimized_path, iterations = optimize_path_2opt(best_path.copy(), similarity_matrix)
    optimized_score = calculate_path_score(optimized_path, similarity_matrix)
    
    print(f"  Optimization completed in {iterations} iterations")
    print(f"  Initial score: {best_score}")
    print(f"  Optimized score: {optimized_score}")
    print(f"  Improvement: {optimized_score - best_score} ({((optimized_score - best_score) / best_score * 100):.2f}%)")
    
    return optimized_path, optimized_score


def save_frame_order(order, frames_data, output_file):
    print(f"\nSaving frame order to: {output_file}")
    
    order_data = {
        'frame_indices': order,
        'frame_filenames': [frames_data[i]['filename'] for i in order],
        'num_frames': len(order)
    }
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(order_data, f)
        print("Frame order saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving frame order: {e}")
        return False


def print_order_statistics(order, frames_data, similarity_matrix):
    print("\n" + "=" * 60)
    print("FRAME ORDERING STATISTICS")
    print("=" * 60)
    
    print(f"Total frames ordered: {len(order)}")
    print(f"First frame: {frames_data[order[0]]['filename']} (index {order[0]})")
    print(f"Last frame: {frames_data[order[-1]]['filename']} (index {order[-1]})")
    
    similarities = []
    for i in range(len(order) - 1):
        sim = similarity_matrix[order[i]][order[i + 1]]
        similarities.append(sim)
    
    print(f"\nConsecutive frame similarities:")
    print(f"  Average: {np.mean(similarities):.2f}")
    print(f"  Median: {np.median(similarities):.2f}")
    print(f"  Min: {np.min(similarities)}")
    print(f"  Max: {np.max(similarities)}")
    print(f"  Std Dev: {np.std(similarities):.2f}")
    
    low_similarity = sum(1 for s in similarities if s < 100)
    print(f"\nPotential issues:")
    print(f"  Consecutive pairs with low similarity (<100): {low_similarity}")
    
    print("=" * 60)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matrix_file = os.path.join(project_root, "similarity_matrix.npy")
    features_file = os.path.join(project_root, "frames_features.pkl")
    output_file = os.path.join(project_root, "frame_order.pkl")
    
    print("=" * 60)
    print("STEP 5: DETERMINE OPTIMAL FRAME ORDER")
    print("=" * 60)
    
    if not os.path.exists(matrix_file):
        print(f"Error: Similarity matrix not found at {matrix_file}")
        print("Please run build_similarity_matrix.py first.")
        return
    
    if not os.path.exists(features_file):
        print(f"Error: Features file not found at {features_file}")
        print("Please run extract_features.py first.")
        return
    
    similarity_matrix = load_similarity_matrix(matrix_file)
    frames_data = load_frames_data(features_file)
    
    if similarity_matrix is None or frames_data is None:
        print("Failed to load required data. Exiting.")
        return
    
    optimal_order, score = find_optimal_path_graph_approach(similarity_matrix)
    
    print_order_statistics(optimal_order, frames_data, similarity_matrix)
    
    success = save_frame_order(optimal_order, frames_data, output_file)
    
    if success:
        print("\nPhase 5A Complete.")
        print(f"Optimal frame order saved to: {output_file}")
        print(f"Total path score: {score}")
        print("\nNext Step: Reconstruct the video using the determined frame order.")
    else:
        print("\nFailed to save frame order.")


if __name__ == "__main__":
    main()
