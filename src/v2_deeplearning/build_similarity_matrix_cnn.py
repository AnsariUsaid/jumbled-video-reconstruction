import numpy as np
import pickle
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def load_cnn_features(features_file):
    print(f"Loading CNN features from: {features_file}")
    try:
        with open(features_file, 'rb') as f:
            frames_data = pickle.load(f)
        print(f"Loaded features for {len(frames_data)} frames")
        return frames_data
    except Exception as e:
        print(f"Error loading features: {e}")
        return None


def build_similarity_matrix_cosine(frames_data):
    n = len(frames_data)
    print(f"Building {n}x{n} similarity matrix using cosine similarity...")
    
    feature_matrix = np.array([frame['features'] for frame in frames_data])
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    similarity_matrix = cosine_similarity(feature_matrix)
    similarity_matrix = similarity_matrix.astype(np.float32)
    similarity_matrix = (similarity_matrix + 1) * 500
    
    print("Similarity matrix built successfully")
    return similarity_matrix


def save_similarity_matrix(matrix, output_file):
    print(f"Saving similarity matrix to: {output_file}")
    try:
        np.save(output_file, matrix)
        print("Matrix saved successfully")
        return True
    except Exception as e:
        print(f"Error saving matrix: {e}")
        return False


def print_matrix_statistics(similarity_matrix, frames_data):
    print("\n" + "=" * 60)
    print("SIMILARITY MATRIX STATISTICS")
    print("=" * 60)
    
    n = len(similarity_matrix)
    print(f"Matrix size: {n}x{n}")
    print(f"Memory size: {similarity_matrix.nbytes / (1024*1024):.2f} MB")
    
    triu_indices = np.triu_indices(n, k=1)
    non_diag = similarity_matrix[triu_indices]
    
    print(f"\nSimilarity scores:")
    print(f"  Min: {non_diag.min():.2f}")
    print(f"  Max: {non_diag.max():.2f}")
    print(f"  Mean: {non_diag.mean():.2f}")
    print(f"  Median: {np.median(non_diag):.2f}")
    print(f"  Std: {np.std(non_diag):.2f}")
    
    temp_matrix = similarity_matrix.copy()
    np.fill_diagonal(temp_matrix, -np.inf)
    max_idx = np.unravel_index(temp_matrix.argmax(), temp_matrix.shape)
    
    print(f"\nMost similar pair:")
    print(f"  {frames_data[max_idx[0]]['filename']} and {frames_data[max_idx[1]]['filename']}")
    print(f"  Score: {similarity_matrix[max_idx]:.2f}")
    print("=" * 60)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_file = os.path.join(project_root, "frames_features_cnn.pkl")
    output_file = os.path.join(project_root, "similarity_matrix_cnn.npy")
    
    print("=" * 60)
    print("V2: BUILD SIMILARITY MATRIX")
    print("=" * 60)
    
    if not os.path.exists(features_file):
        print(f"Error: Features file not found at {features_file}")
        return
    
    frames_data = load_cnn_features(features_file)
    if frames_data is None or len(frames_data) == 0:
        print("No frame data loaded")
        return
    
    similarity_matrix = build_similarity_matrix_cosine(frames_data)
    print_matrix_statistics(similarity_matrix, frames_data)
    success = save_similarity_matrix(similarity_matrix, output_file)
    
    if success:
        print("\nPhase 4 V2 Complete")
        print(f"Matrix saved to: {output_file}")
        print("Next: Run order_frames_improved.py")


if __name__ == "__main__":
    main()
