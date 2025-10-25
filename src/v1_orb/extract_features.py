import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm


def extract_orb_features(frame):
    orb = cv2.ORB_create(nfeatures=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def load_and_process_frames(frames_dir):
    print(f"Reading frames from: {frames_dir}")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    if len(frame_files) == 0:
        print("Error: No frames found in the directory.")
        return []

    print(f"Found {len(frame_files)} frames to process")
    frames_data = []

    for frame_file in tqdm(frame_files, desc="Processing frames", unit="frame"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: Could not read {frame_file}, skipping...")
            continue

        keypoints, descriptors = extract_orb_features(frame)

        frame_info = {
            'filename': frame_file,
            'path': frame_path,
            'shape': frame.shape,
            'num_keypoints': len(keypoints) if keypoints else 0,
            'descriptors': descriptors
        }

        frames_data.append(frame_info)

    print(f"Successfully processed {len(frames_data)} frames")
    return frames_data


def save_features(frames_data, output_file):
    print(f"Saving features to: {output_file}")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(frames_data, f)
        print("Features saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving features: {e}")
        return False


def print_statistics(frames_data):
    print("=" * 60)
    print("FEATURE EXTRACTION STATISTICS")
    print("=" * 60)

    total_frames = len(frames_data)
    frames_with_features = sum(1 for f in frames_data if f['descriptors'] is not None)
    total_keypoints = sum(f['num_keypoints'] for f in frames_data)
    avg_keypoints = total_keypoints / total_frames if total_frames > 0 else 0

    print(f"Total frames processed: {total_frames}")
    print(f"Frames with features: {frames_with_features}")
    print(f"Total keypoints detected: {total_keypoints}")
    print(f"Average keypoints per frame: {avg_keypoints:.2f}")

    if total_frames > 0:
        min_kp_frame = min(frames_data, key=lambda x: x['num_keypoints'])
        max_kp_frame = max(frames_data, key=lambda x: x['num_keypoints'])

        print(f"Frame with fewest keypoints: {min_kp_frame['filename']} ({min_kp_frame['num_keypoints']} keypoints)")
        print(f"Frame with most keypoints: {max_kp_frame['filename']} ({max_kp_frame['num_keypoints']} keypoints)")

    print("=" * 60)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_dir = os.path.join(project_root, "frames")
    output_file = os.path.join(project_root, "frames_features.pkl")

    print("=" * 60)
    print("STEP 3: ORB FEATURE EXTRACTION")
    print("=" * 60)

    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found at {frames_dir}")
        print("Please run extract_frames.py first.")
        return

    frames_data = load_and_process_frames(frames_dir)

    if len(frames_data) == 0:
        print("No frames were processed. Exiting.")
        return

    print_statistics(frames_data)

    success = save_features(frames_data, output_file)

    if success:
        print("Phase 3 Complete.")
        print(f"Features saved to: {output_file}")
        print("Next Step: Run the frame comparison script to determine the correct order.")
    else:
        print("Failed to save features.")


if __name__ == "__main__":
    main()
