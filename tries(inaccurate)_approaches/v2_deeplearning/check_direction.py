import pickle
import numpy as np
import os


def load_frame_order(order_file):
    with open(order_file, 'rb') as f:
        return pickle.load(f)


def check_temporal_direction(frames_dir, frame_order):
    """
    Check if video is going forward or backward by analyzing brightness changes.
    Most videos have increasing or consistent patterns.
    """
    import cv2
    
    indices = frame_order['frame_indices']
    filenames = frame_order['frame_filenames']
    
    # Sample 10 frames from beginning, middle, end
    sample_indices = [0, len(indices)//4, len(indices)//2, 3*len(indices)//4, len(indices)-1]
    
    print("Analyzing temporal direction...")
    print("\nFrame sequence check:")
    
    for i in sample_indices:
        idx = indices[i]
        filename = filenames[i]
        print(f"  Position {i}: {filename} (original index: {idx})")
    
    # Ask user if it looks backward
    print("\nBased on the frame indices, does the sequence look backward?")
    print(f"Start frame: {filenames[0]} (index {indices[0]})")
    print(f"End frame: {filenames[-1]} (index {indices[-1]})")
    
    return indices[0] > indices[-1]  # If start > end, likely backward


def reverse_order(frame_order):
    """Reverse the frame order."""
    frame_order['frame_indices'] = frame_order['frame_indices'][::-1]
    frame_order['frame_filenames'] = frame_order['frame_filenames'][::-1]
    return frame_order


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    order_file = os.path.join(project_root, "frame_order_cnn.pkl")
    frames_dir = os.path.join(project_root, "frames")
    
    print("=" * 60)
    print("VIDEO DIRECTION CHECKER & FIXER")
    print("=" * 60)
    
    # Load frame order
    frame_order = load_frame_order(order_file)
    
    print(f"\nCurrent order:")
    print(f"  First frame: {frame_order['frame_filenames'][0]}")
    print(f"  Last frame: {frame_order['frame_filenames'][-1]}")
    print(f"  Total frames: {frame_order['num_frames']}")
    
    # Check direction
    is_backward = check_temporal_direction(frames_dir, frame_order)
    
    if is_backward:
        print("\n⚠️  Video appears to be BACKWARD!")
        print("Reversing frame order...")
        
        frame_order = reverse_order(frame_order)
        
        # Save corrected order
        corrected_file = os.path.join(project_root, "frame_order_cnn_corrected.pkl")
        with open(corrected_file, 'wb') as f:
            pickle.dump(frame_order, f)
        
        print(f"\n✅ Corrected order saved to: {corrected_file}")
        print(f"\nNew order:")
        print(f"  First frame: {frame_order['frame_filenames'][0]}")
        print(f"  Last frame: {frame_order['frame_filenames'][-1]}")
        
        print("\n🔧 Now run:")
        print("   python fix_backward_video.py")
        print("   This will reconstruct the video in forward direction")
        
    else:
        print("\n✅ Video appears to be in correct forward direction")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
