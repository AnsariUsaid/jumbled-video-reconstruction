
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

def run_yolo_bytetrack(video_path, output_csv_path, conf_threshold=0.2):
    """
    Runs YOLOv8 with ByteTrack on a video to generate tracking data.

    Args:
        video_path (str): Path to the input video file.
        output_csv_path (str): Path to save the output tracking data in CSV format.
        conf_threshold (float): Confidence threshold for detections (default: 0.2 or 20%).
    """
    # Load the YOLOv8 model
    model = YOLO('yolov8x.pt')

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a list to store tracking data
    tracking_data = []

    # Loop through the video frames
    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection with confidence threshold
        results = model(frame, conf=conf_threshold, verbose=False)

        # Get the bounding boxes for 'person' class (class 0 in COCO)
        boxes = results[0].boxes
        
        # Filter for person detections only
        person_boxes = []
        for box in boxes:
            if int(box.cls[0]) == 0:  # Person class
                person_boxes.append(box)
        
        # Skip if no person detected
        if len(person_boxes) == 0:
            continue

        # Get the largest person detection (main subject)
        largest_box = max(person_boxes, key=lambda b: b.xywh[0][2] * b.xywh[0][3])
        xywh = largest_box.xywh.cpu()[0]
        
        x, y, w, h = xywh.tolist()
        
        # Convert to float to ensure numeric values (not tensor strings)
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        
        # Append data to the list
        tracking_data.append({
            'frame': frame_num,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'confidence': float(largest_box.conf[0])
        })

    # Release the video capture object
    cap.release()

    # Create a pandas DataFrame and save to CSV
    df = pd.DataFrame(tracking_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Tracking data saved to {output_csv_path}")
    print(f"Total frames in video: {frame_count}")
    print(f"Frames with person detected: {len(tracking_data)}")
    print(f"Frames without detection: {frame_count - len(tracking_data)}")
    print(f"Detection rate: {len(tracking_data)/frame_count*100:.1f}%")

if __name__ == '__main__':
    # Define paths
    video_path = '../../jumbled_video.mp4'
    output_dir = '../../output'
    output_csv_path = os.path.join(output_dir, 'v8B_tracking_data.csv')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the tracking
    run_yolo_bytetrack(video_path, output_csv_path)
