
import numpy as np
import pandas as pd
import os

def solve_optimal_order(cost_matrix_path, output_csv_path, use_2opt=True):
    """
    Solves for the optimal frame order using greedy nearest-neighbor with 2-opt improvement.
    Starts from a frame where person is at expected starting position.

    Args:
        cost_matrix_path (str): Path to the cost matrix.
        output_csv_path (str): Path to save the ordered frame list.
        use_2opt (bool): Whether to apply 2-opt local search improvement (default: True)
    """
    # Load the cost matrix
    cost_matrix = np.load(cost_matrix_path)
    n_frames = cost_matrix.shape[0]
    
    # Load frame indices
    frame_indices_path = cost_matrix_path.replace('_cost_matrix.npy', '_frame_indices.csv')
    if os.path.exists(frame_indices_path):
        frame_df = pd.read_csv(frame_indices_path)
        frames = frame_df['frame'].values
        positions_x = frame_df['cx'].values
        positions_y = frame_df['cy'].values
    else:
        print("ERROR: Frame indices not found!")
        return
    
    # Find starting frame: person at bottom-right (typical start for this video)
    # Higher x + y = bottom-right
    position_scores = positions_x + positions_y
    start_idx = np.argmax(position_scores)
    
    print(f"Starting from frame index {start_idx} (frame {frames[start_idx]}) at position ({positions_x[start_idx]:.1f}, {positions_y[start_idx]:.1f})")
    
    # Greedy nearest-neighbor path with motion vector awareness
    current_idx = start_idx
    path = [current_idx]
    unvisited = set(range(n_frames))
    unvisited.remove(current_idx)
    
    total_cost = 0.0
    large_jumps = 0
    
    # Build path by always choosing nearest unvisited neighbor
    while unvisited:
        last_idx = path[-1]
        
        # Find the nearest unvisited frame
        costs_to_unvisited = [(cost_matrix[last_idx][j], j) for j in unvisited]
        min_cost, next_idx = min(costs_to_unvisited)
        
        path.append(next_idx)
        unvisited.remove(next_idx)
        total_cost += min_cost
    
    print(f"\nInitial greedy path cost: {total_cost:.2f}")
    
    # Apply 2-opt local search to improve the path
    if use_2opt:
        print("Applying 2-opt local search optimization...")
        path, total_cost = two_opt_optimize(path, cost_matrix, max_iterations=1000)
        print(f"Optimized path cost: {total_cost:.2f}")
    
    # Convert path indices to actual frame numbers
    ordered_frames = [frames[i] for i in path]
    
    # Calculate statistics on the ordered sequence
    avg_cost = total_cost / (n_frames - 1) if n_frames > 1 else 0
    
    # Calculate actual spatial distances in the ordered sequence
    distances = []
    for i in range(len(path) - 1):
        idx1, idx2 = path[i], path[i+1]
        dx = positions_x[idx2] - positions_x[idx1]
        dy = positions_y[idx2] - positions_y[idx1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)
        if dist > 100:
            large_jumps += 1
    
    distances = np.array(distances)
    
    print(f"\n=== Ordering Results ===")
    print(f"Path length: {len(path)} frames")
    print(f"Total cost: {total_cost:.2f}")
    print(f"Average cost per transition: {avg_cost:.4f}")
    print(f"\n=== Spatial Motion Analysis ===")
    print(f"Average step distance: {distances.mean():.2f} pixels")
    print(f"Median step distance: {np.median(distances):.2f} pixels")
    print(f"Min step: {distances.min():.2f} pixels")
    print(f"Max step: {distances.max():.2f} pixels")
    print(f"Large jumps (>100px): {large_jumps} out of {len(distances)} transitions ({100*large_jumps/len(distances):.1f}%)")
    print(f"Total path distance: {distances.sum():.2f} pixels")
    
    # Create a DataFrame and save the ordered path
    df = pd.DataFrame({
        'ordered_frame': ordered_frames,
        'path_index': range(len(ordered_frames))
    })
    df.to_csv(output_csv_path, index=False)
    print(f"\nOptimal frame order saved to {output_csv_path}")


def two_opt_optimize(path, cost_matrix, max_iterations=1000):
    """
    Apply 2-opt local search to improve the path.
    2-opt works by repeatedly reversing segments of the path if it reduces cost.
    
    Args:
        path (list): Initial path (list of frame indices)
        cost_matrix (np.array): Cost matrix between frames
        max_iterations (int): Maximum number of improvement iterations
        
    Returns:
        tuple: (optimized_path, total_cost)
    """
    def calculate_path_cost(p):
        return sum(cost_matrix[p[i]][p[i+1]] for i in range(len(p)-1))
    
    current_path = path[:]
    current_cost = calculate_path_cost(current_path)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try all possible 2-opt swaps
        for i in range(1, len(current_path) - 2):
            for j in range(i + 1, len(current_path)):
                # Skip if segment is too small
                if j - i < 2:
                    continue
                
                # Calculate cost change for reversing segment [i:j]
                # Cost before: ... -> path[i-1] -> path[i] -> ... -> path[j-1] -> path[j] -> ...
                # Cost after:  ... -> path[i-1] -> path[j-1] -> ... -> path[i] -> path[j] -> ...
                
                # Remove old edges
                old_cost = cost_matrix[current_path[i-1]][current_path[i]]
                old_cost += cost_matrix[current_path[j-1]][current_path[j]] if j < len(current_path) else 0
                
                # Add new edges (with reversed segment)
                new_cost = cost_matrix[current_path[i-1]][current_path[j-1]]
                new_cost += cost_matrix[current_path[i]][current_path[j]] if j < len(current_path) else 0
                
                # If improvement found, apply the swap
                if new_cost < old_cost:
                    # Reverse the segment [i:j]
                    current_path[i:j] = reversed(current_path[i:j])
                    current_cost = calculate_path_cost(current_path)
                    improved = True
                    break  # Start over with new path
            
            if improved:
                break
        
        if iteration % 100 == 0 and iteration > 0:
            print(f"  2-opt iteration {iteration}, cost: {current_cost:.2f}")
    
    print(f"  2-opt completed after {iteration} iterations")
    
    return current_path, current_cost

if __name__ == '__main__':
    # Define paths
    input_matrix_path = '../../output/v8_cost_matrix.npy'
    output_dir = '../../output'
    output_csv_path = os.path.join(output_dir, 'v8_optimal_order.csv')

    # Solve for the optimal order
    solve_optimal_order(input_matrix_path, output_csv_path)
