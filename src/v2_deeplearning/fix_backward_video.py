import pickle
import os
import sys


def reverse_frame_order():
    """Simply reverse the frame order to fix backward video."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    order_file = os.path.join(project_root, "frame_order_cnn.pkl")
    
    print("=" * 60)
    print("FIXING BACKWARD VIDEO")
    print("=" * 60)
    
    # Load current order
    with open(order_file, 'rb') as f:
        frame_order = pickle.load(f)
    
    print(f"\nCurrent (backward) order:")
    print(f"  First: {frame_order['frame_filenames'][0]}")
    print(f"  Last: {frame_order['frame_filenames'][-1]}")
    
    # Reverse
    frame_order['frame_indices'] = frame_order['frame_indices'][::-1]
    frame_order['frame_filenames'] = frame_order['frame_filenames'][::-1]
    
    print(f"\nReversed (forward) order:")
    print(f"  First: {frame_order['frame_filenames'][0]}")
    print(f"  Last: {frame_order['frame_filenames'][-1]}")
    
    # Save
    with open(order_file, 'wb') as f:
        pickle.dump(frame_order, f)
    
    print(f"\n✅ Frame order reversed and saved!")
    print(f"📁 File: {order_file}")
    print(f"\n🎬 Now run reconstruct_video.py to rebuild the video")
    print("=" * 60)


if __name__ == "__main__":
    reverse_frame_order()
