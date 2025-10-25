import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os


def load_features(features_file):
    print(f"Loading features from: {features_file}")
    try:
        with open(features_file, 'rb') as f:
            frames_data = pickle.load(f)
        print(f"Successfully loaded features for {len(frames_data)} frames")
        return frames_data
    except Exception as e:
        print(f"Error loading features: {e}")
        return None


def compare_frames(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    try:
        matches = bf.match(desc1, desc2)
        return len(matches)
    except:
        return 0


def build_similarity_matrix(frames_data):
    n = len(frames_data)
    similarity_matrix = np.zeros((n, n), dtype=np.int32)
    
    print(f"\nBuilding {n}x{n} similarity matrix...")
    print("Comparing every frame pair using Brute-Force Matcher with Hamming distance...")
    
    total_comparisons = (n * (n - 1)) // 2
    
    with tqdm(total=total_comparisons, desc="Computing similarities", unit="pair") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                matches = compare_frames(
                    frames_data[i]['descriptors'],
                    frames_data[j]['descriptors']
                )
                similarity_matrix[i][j] = matches
                similarity_matrix[j][i] = matches
                pbar.update(1)
    
    return similarity_matrix


def save_similarity_matrix(matrix, output_file):
    print(f"\nSaving similarity matrix to: {output_file}")
    try:
        np.save(output_file, matrix)
        print("Similarity matrix saved successfully.")
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
    print(f"Total elements: {n * n}")
    print(f"Memory size: {similarity_matrix.nbytes / (1024*1024):.2f} MB")
    
    non_diag = similarity_matrix[np.triu_indices(n, k=1)]
    
    print(f"\nSimilarity scores:")
    print(f"  Minimum matches: {non_diag.min()}")
    print(f"  Maximum matches: {non_diag.max()}")
    print(f"  Average matches: {non_diag.mean():.2f}")
    print(f"  Median matches: {np.median(non_diag):.2f}")
    
    max_idx = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
    if max_idx[0] != max_idx[1]:
        print(f"\nMost similar frames:")
        print(f"  {frames_data[max_idx[0]]['filename']} and {frames_data[max_idx[1]]['filename']}")
        print(f"  Matches: {similarity_matrix[max_idx]}")
    
    print("=" * 60)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_file = os.path.join(project_root, "frames_features.pkl")
    output_file = os.path.join(project_root, "similarity_matrix.npy")
    
    print("=" * 60)
    print("STEP 4: BUILD SIMILARITY MATRIX")
    print("=" * 60)
    
    if not os.path.exists(features_file):
        print(f"Error: Features file not found at {features_file}")
        print("Please run extract_features.py first.")
        return
    
    frames_data = load_features(features_file)
    
    if frames_data is None or len(frames_data) == 0:
        print("No frame data loaded. Exiting.")
        return
    
    similarity_matrix = build_similarity_matrix(frames_data)
    
    print_matrix_statistics(similarity_matrix, frames_data)
    
    success = save_similarity_matrix(similarity_matrix, output_file)
    
    if success:
        print("\nPhase 4 Complete.")
        print(f"Similarity matrix saved to: {output_file}")
        print("\nNext Step: Use the similarity matrix to determine correct frame order.")
    else:
        print("\nFailed to save similarity matrix.")


if __name__ == "__main__":
    main()
