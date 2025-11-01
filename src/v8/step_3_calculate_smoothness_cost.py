
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_smoothness_cost(input_csv_path, output_matrix_path, video_path='../../jumbled_video.mp4', 
                              alpha=0.5, beta=0.5, use_ssim=True):
    """
    Calculates a hybrid pairwise cost matrix using motion vectors and SSIM.
    Lower cost = frames that should be consecutive.
    
    Hybrid Cost = alpha * (motion_cost) + beta * (1 - SSIM)
    
    Uses multiple factors:
    1. Position distance (spatial continuity)
    2. Size consistency (person scale/depth)
    3. SSIM (Structural Similarity) - perceptual similarity between frames
    
    Args:
        input_csv_path (str): Path to the motion model data.
        output_matrix_path (str): Path to save the output cost matrix.
        video_path (str): Path to the video file for SSIM calculation.
        alpha (float): Weight for motion cost (default: 0.5)
        beta (float): Weight for SSIM cost (default: 0.5)
        use_ssim (bool): Whether to use SSIM in the hybrid cost (default: True)
    """
    # Load the motion model data
    df = pd.read_csv(input_csv_path)
    
    n_frames = len(df)
    print(f"Building hybrid motion + SSIM cost matrix for {n_frames} frames")
    print(f"Hybrid weights: alpha (motion) = {alpha}, beta (SSIM) = {beta}")
    
    # Extract positions and features
    positions = df[['cx', 'cy']].values
    areas = df['area'].values
    frames = df['frame'].values
    
    # Load video frames for SSIM calculation if enabled
    frame_images = {}
    if use_ssim:
        print("Loading video frames for SSIM calculation...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}. Falling back to motion-only cost.")
            use_ssim = False
        else:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to grayscale and resize for faster SSIM computation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to 480p for faster computation
                gray_resized = cv2.resize(gray, (854, 480))
                frame_images[frame_idx] = gray_resized
                frame_idx += 1
            cap.release()
            print(f"Loaded {len(frame_images)} frames for SSIM calculation")
    
    # Initialize cost matrices
    motion_cost_matrix = np.zeros((n_frames, n_frames))
    ssim_cost_matrix = np.zeros((n_frames, n_frames)) if use_ssim else None
    
    # Calculate motion-based costs
    print("Calculating motion costs...")
    for i in range(n_frames):
        for j in range(n_frames):
            if i == j:
                motion_cost_matrix[i, j] = 0
                continue
            
            # Factor 1: Position distance (Euclidean distance between centroids)
            pos_dist = np.sqrt((positions[j, 0] - positions[i, 0])**2 + 
                              (positions[j, 1] - positions[i, 1])**2)
            
            # Factor 2: Size consistency (similar size = likely consecutive)
            size_diff = abs(areas[j] - areas[i]) / max(areas[i], areas[j], 1)
            
            # Penalize extreme movements
            movement_penalty = 1.0
            if pos_dist > 100:
                movement_penalty = 2.0
            elif pos_dist > 200:
                movement_penalty = 5.0
            elif pos_dist > 300:
                movement_penalty = 10.0
            
            # Combine factors
            position_cost = pos_dist / 100.0
            size_cost = size_diff * 2.0
            
            total_cost = (position_cost * movement_penalty) + size_cost
            motion_cost_matrix[i, j] = total_cost
    
    # Apply directional preference
    for i in range(n_frames):
        for j in range(n_frames):
            if i == j:
                continue
            dx = positions[j, 0] - positions[i, 0]
            if dx < -150:
                motion_cost_matrix[i, j] *= 1.5
    
    # Normalize motion costs to [0, 1]
    motion_max = motion_cost_matrix.max()
    if motion_max > 0:
        motion_cost_normalized = motion_cost_matrix / motion_max
    else:
        motion_cost_normalized = motion_cost_matrix
    
    # Calculate SSIM-based costs if enabled
    # Optimization: Only calculate SSIM for frames with low motion cost (likely to be consecutive)
    # This dramatically reduces computation from O(n²) to much less
    if use_ssim and ssim_cost_matrix is not None:
        print("Calculating SSIM costs (optimized - only for low motion cost pairs)...")
        
        # First, identify candidate pairs with low motion cost
        # Use motion cost to filter: only compute SSIM for top candidates
        motion_cost_threshold = np.percentile(motion_cost_normalized[motion_cost_normalized > 0], 25)
        print(f"  SSIM threshold: motion cost < {motion_cost_threshold:.4f}")
        
        ssim_computed = 0
        for i in range(n_frames):
            frame_i = int(frames[i])
            if frame_i not in frame_images:
                continue
            
            for j in range(n_frames):
                if i == j:
                    ssim_cost_matrix[i, j] = 0
                    continue
                
                # Only compute SSIM for low-cost motion pairs
                if motion_cost_normalized[i, j] > motion_cost_threshold:
                    # High motion cost - use a default high SSIM cost
                    ssim_cost_matrix[i, j] = 0.5
                    continue
                
                frame_j = int(frames[j])
                if frame_j not in frame_images:
                    ssim_cost_matrix[i, j] = 0.5  # Medium dissimilarity
                    continue
                
                # Calculate SSIM between frames
                ssim_value = ssim(frame_images[frame_i], frame_images[frame_j], 
                                 data_range=255)
                
                # Convert SSIM to cost (SSIM ranges from -1 to 1, typically 0 to 1)
                # Cost = 1 - SSIM, so similar frames have low cost
                ssim_cost_matrix[i, j] = (1 - ssim_value) / 2  # Normalize to [0, 1]
                ssim_computed += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_frames} frames, SSIM computed: {ssim_computed}")
        
        print(f"  Total SSIM computations: {ssim_computed} out of {n_frames * n_frames} pairs ({ssim_computed / (n_frames * n_frames) * 100:.1f}%)")
    
    # Combine costs using weighted sum
    if use_ssim and ssim_cost_matrix is not None:
        cost_matrix = alpha * motion_cost_normalized + beta * ssim_cost_matrix
        print(f"\n=== Hybrid Cost Statistics ===")
        print(f"Motion cost - avg: {motion_cost_normalized.mean():.4f}, max: {motion_cost_normalized.max():.4f}")
        print(f"SSIM cost - avg: {ssim_cost_matrix.mean():.4f}, max: {ssim_cost_matrix.max():.4f}")
    else:
        cost_matrix = motion_cost_normalized
        print(f"\n=== Motion-Only Cost Statistics ===")
    
    print(f"\n=== Final Cost Matrix ===")
    print(f"Cost matrix shape: {cost_matrix.shape}")
    print(f"Average cost: {cost_matrix.mean():.4f}")
    print(f"Min non-zero cost: {cost_matrix[cost_matrix > 0].min():.4f}")
    print(f"Max cost: {cost_matrix.max():.4f}")
    
    # Analyze cost distribution
    low_cost_pairs = np.sum(cost_matrix < 0.1) - n_frames
    print(f"Low cost pairs (<0.1): {low_cost_pairs} out of {n_frames * (n_frames - 1)}")

    # Save the cost matrix and frame mapping
    np.save(output_matrix_path, cost_matrix)
    
    # Save frame indices for reference
    frame_indices_path = output_matrix_path.replace('_cost_matrix.npy', '_frame_indices.csv')
    df[['frame', 'cx', 'cy']].to_csv(frame_indices_path, index=False)
    
    print(f"Cost matrix saved to {output_matrix_path}")
    print(f"Frame indices saved to {frame_indices_path}")

if __name__ == '__main__':
    # Define paths
    input_csv_path = '../../output/v8_motion_model.csv'
    output_dir = '../../output'
    output_matrix_path = os.path.join(output_dir, 'v8_cost_matrix.npy')
    video_path = '../../jumbled_video.mp4'

    # Hybrid cost parameters
    # alpha: weight for motion cost
    # beta: weight for SSIM cost
    # Setting equal weights (0.5, 0.5) balances motion smoothness with visual similarity
    alpha = 0.5
    beta = 0.5

    # Calculate the hybrid cost matrix
    calculate_smoothness_cost(input_csv_path, output_matrix_path, video_path, 
                            alpha=alpha, beta=beta, use_ssim=True)
