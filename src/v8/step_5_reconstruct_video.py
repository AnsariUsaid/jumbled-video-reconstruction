
import cv2
import pandas as pd
import numpy as np
import os

def reconstruct_video(ordered_frames_path, video_path, output_video_path):
    """
    Reconstructs the video from the ordered frames.
    Uses all detected frames in their optimal order.

    Args:
        ordered_frames_path (str): Path to the CSV file with the ordered frame indices.
        video_path (str): Path to the original jumbled video.
        output_video_path (str): Path to save the reconstructed video.
    """
    # Load the ordered frame list
    df_order = pd.read_csv(ordered_frames_path)
    ordered_frames = df_order['ordered_frame'].tolist()
    
    print(f"Reconstructing video with {len(ordered_frames)} frames")

    # Open the original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {total_frames} frames, {frame_width}x{frame_height}, {fps} FPS")

    # Create a list of all frames
    print("Loading all frames from video...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames from video")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Write the frames in the correct order
    written_count = 0
    for frame_index in ordered_frames:
        if frame_index < len(frames):
            out.write(frames[frame_index])
            written_count += 1
        else:
            print(f"Warning: Frame index {frame_index} out of range")

    # Release everything when job is finished
    out.release()
    
    detection_rate = (written_count / total_frames) * 100
    
    print(f"\n=== Reconstruction Complete ===")
    print(f"Output video: {output_video_path}")
    print(f"Written {written_count} frames")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Missing frames: {total_frames - written_count} ({100 * (total_frames - written_count) / total_frames:.1f}%)")
    
    if detection_rate == 100:
        print("✅ Perfect! All frames included")
    elif detection_rate >= 90:
        print("✅ Excellent! Most frames included")
    elif detection_rate >= 75:
        print("⚠️  Good, but some frames missing")
    else:
        print("⚠️  Warning: Many frames missing - consider using hybrid approach (V6)")

if __name__ == '__main__':
    # Define paths
    ordered_frames_path = '../../output/v8B_optimal_order.csv'
    video_path = '../../jumbled_video.mp4'
    output_dir = '../../output'
    output_video_path = os.path.join(output_dir, 'reconstructed_video_v8B.mp4')

    # Reconstruct the video
    reconstruct_video(ordered_frames_path, video_path, output_video_path)
