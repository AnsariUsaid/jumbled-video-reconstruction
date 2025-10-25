import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm


def load_frame_order(order_file):
    print(f"Loading frame order from: {order_file}")
    try:
        with open(order_file, 'rb') as f:
            order_data = pickle.load(f)
        print(f"Successfully loaded order for {order_data['num_frames']} frames")
        return order_data
    except Exception as e:
        print(f"Error loading frame order: {e}")
        return None


def get_video_properties(frames_dir, first_frame_filename):
    first_frame_path = os.path.join(frames_dir, first_frame_filename)
    frame = cv2.imread(first_frame_path)
    
    if frame is None:
        print(f"Error: Could not read first frame: {first_frame_filename}")
        return None
    
    height, width = frame.shape[:2]
    return width, height


def reconstruct_video(order_data, frames_dir, output_path, fps=30):
    print(f"\nReconstructing video...")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    
    frame_filenames = order_data['frame_filenames']
    
    width, height = get_video_properties(frames_dir, frame_filenames[0])
    if width is None:
        return False
    
    print(f"Resolution: {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer")
        return False
    
    print(f"\nWriting {len(frame_filenames)} frames in correct order...")
    
    for filename in tqdm(frame_filenames, desc="Writing frames", unit="frame"):
        frame_path = os.path.join(frames_dir, filename)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {filename}, skipping...")
            continue
        
        out.write(frame)
    
    out.release()
    print(f"\nVideo reconstruction complete!")
    return True


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    order_file = os.path.join(project_root, "frame_order.pkl")
    frames_dir = os.path.join(project_root, "frames")
    output_dir = os.path.join(project_root, "output")
    output_video = os.path.join(output_dir, "reconstructed_video.mp4")
    
    print("=" * 60)
    print("STEP 5B: RECONSTRUCT VIDEO FROM ORDERED FRAMES")
    print("=" * 60)
    
    if not os.path.exists(order_file):
        print(f"Error: Frame order file not found at {order_file}")
        print("Please run order_frames.py first.")
        return
    
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found at {frames_dir}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    order_data = load_frame_order(order_file)
    
    if order_data is None:
        print("Failed to load frame order. Exiting.")
        return
    
    success = reconstruct_video(order_data, frames_dir, output_video, fps=30)
    
    if success:
        file_size = os.path.getsize(output_video) / (1024 * 1024)
        print("\n" + "=" * 60)
        print("VIDEO RECONSTRUCTION COMPLETE!")
        print("=" * 60)
        print(f"Output file: {output_video}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Total frames: {order_data['num_frames']}")
        print(f"FPS: 30")
        print(f"Duration: {order_data['num_frames'] / 30:.2f} seconds")
        print("\nYou can now play the reconstructed video!")
        print("=" * 60)
    else:
        print("\nVideo reconstruction failed.")


if __name__ == "__main__":
    main()
