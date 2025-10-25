import cv2
import os
from tqdm import tqdm


def extract_frames(video_path, output_dir):

    # Handle output directory if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    #Takes vedio properties and metadata
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Properties:")
    print(f"  Total Frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"\nExtracting frames to: {output_dir}")
    
    # Extract frames with progress bar using tqdm lib
    frame_count = 0
    success = True
    
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while success:
            success, frame = video.read()
            
            if success:
                # the imwrite fn saves the frame as jpg
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
                pbar.update(1)
    
    # release the object
    video.release()
    
    print(f"\n Successfully extracted {frame_count} frames!")
    print(f" Frames saved in: {output_dir}")
    
    return frame_count, fps, (width, height)


def main():
    """Main function to run frame extraction."""
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(project_root, "jumbled_video.mp4")
    output_dir = os.path.join(project_root, "frames")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f" Error: Video file not found at {video_path}")
        print("Please place your jumbled_video.mp4 in the project root directory.")
        return
    
    # Extract frames
    result = extract_frames(video_path, output_dir)
    
    if result:
        frame_count, fps, resolution = result
        print(f"\n Summary:")
        print(f"  Frames extracted: {frame_count}")
        print(f"  Video FPS: {fps}")
        print(f"  Frame resolution: {resolution[0]}x{resolution[1]}")


if __name__ == "__main__":
    main()
