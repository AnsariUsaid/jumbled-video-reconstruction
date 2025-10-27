import cv2
import numpy as np
import os
import pickle
from pathlib import Path

def detect_person_multiple_methods(frame):
    detections = []
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(
        frame, 
        winStride=(4,4), 
        padding=(8,8), 
        scale=1.03,
        hitThreshold=0.3
    )
    
    if len(boxes) > 0:
        best_idx = np.argmax(weights)
        detections.append(boxes[best_idx])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
    if len(bodies) > 0:
        detections.append(bodies[0])
    
    if len(detections) == 0:
        return None
    
    if len(detections) == 1:
        return tuple(detections[0])
    
    areas = [w*h for x,y,w,h in detections]
    best_idx = np.argmax(areas)
    return tuple(detections[best_idx])

def get_reference_point(bbox):
    if bbox is None:
        return None
    x, y, w, h = bbox
    center_x = x + w // 2
    bottom_y = y + h
    return (center_x, bottom_y)

def interpolate_missing_points(coords):
    coords = list(coords)
    n = len(coords)
    
    for i in range(n):
        if coords[i] is not None:
            continue
        
        prev_idx = None
        for j in range(i-1, -1, -1):
            if coords[j] is not None:
                prev_idx = j
                break
        
        next_idx = None
        for j in range(i+1, n):
            if coords[j] is not None:
                next_idx = j
                break
        
        if prev_idx is not None and next_idx is not None:
            prev_coord = coords[prev_idx]
            next_coord = coords[next_idx]
            alpha = (i - prev_idx) / (next_idx - prev_idx)
            interp_x = prev_coord[0] + alpha * (next_coord[0] - prev_coord[0])
            interp_y = prev_coord[1] + alpha * (next_coord[1] - prev_coord[1])
            coords[i] = (interp_x, interp_y)
        elif prev_idx is not None:
            coords[i] = coords[prev_idx]
        elif next_idx is not None:
            coords[i] = coords[next_idx]
    
    return coords

def smooth_coordinates(coords, window_size=5):
    if len(coords) < window_size:
        return coords
    
    smoothed = []
    for i in range(len(coords)):
        start = max(0, i - window_size // 2)
        end = min(len(coords), i + window_size // 2 + 1)
        
        valid_coords = [c for c in coords[start:end] if c is not None]
        if valid_coords:
            avg_x = np.mean([c[0] for c in valid_coords])
            avg_y = np.mean([c[1] for c in valid_coords])
            smoothed.append((avg_x, avg_y))
        else:
            smoothed.append(coords[i])
    
    return smoothed

def detect_and_track_person(frames_dir):
    print("Starting person detection and tracking...")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    total_frames = len(frame_files)
    
    frame_data = []
    
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Failed to load frame: {frame_file}")
            frame_data.append({
                'filename': frame_file,
                'bbox': None,
                'reference_point': None
            })
            continue
        
        bbox = detect_person_multiple_methods(frame)
        ref_point = get_reference_point(bbox)
        
        frame_data.append({
            'filename': frame_file,
            'bbox': bbox,
            'reference_point': ref_point
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total_frames} frames")
    
    detected_count = sum(1 for fd in frame_data if fd['bbox'] is not None)
    print(f"Initial detection: {detected_count}/{total_frames} frames")
    
    print("Interpolating missing detections...")
    ref_points = [fd['reference_point'] for fd in frame_data]
    interpolated_points = interpolate_missing_points(ref_points)
    
    print("Smoothing reference points...")
    smoothed_points = smooth_coordinates(interpolated_points)
    
    for i, fd in enumerate(frame_data):
        fd['interpolated_reference_point'] = interpolated_points[i]
        fd['smoothed_reference_point'] = smoothed_points[i]
    
    output_file = 'frame_tracking_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(frame_data, f)
    
    print(f"Tracking data saved to {output_file}")
    
    valid_count = sum(1 for fd in frame_data if fd['smoothed_reference_point'] is not None)
    print(f"Final valid tracking points: {valid_count}/{total_frames} frames")
    
    return frame_data

if __name__ == "__main__":
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    os.chdir(project_root)
    
    frames_dir = "frames"
    detect_and_track_person(frames_dir)
