import cv2
import numpy as np
from pathlib import Path
import pickle
import time

def detect_person_motion(frame, frame_gray, median_frame=None):
    if median_frame is None:
        return None
    
    # Compute difference from median (static background)
    diff = cv2.absdiff(frame_gray, median_frame)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour (assuming it's the person)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Filter by minimum area (person should be reasonably sized)
        if area > 5000:  # Adjust based on your video
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Additional check: aspect ratio should be roughly person-like
            aspect_ratio = h / w if w > 0 else 0
            if 0.5 < aspect_ratio < 4:
                return [int(x), int(y), int(w), int(h)]
    
    return None

def detect_person_hog(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Try with lower threshold first
    boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    
    if len(boxes) > 0:
        # Filter by weight threshold - lower threshold for better recall
        valid_boxes = [(box, weight) for box, weight in zip(boxes, weights) if weight > 0.2]
        if valid_boxes:
            # Get the box with highest weight
            best_box, best_weight = max(valid_boxes, key=lambda x: x[1])
            x, y, w, h = best_box
            return [int(x), int(y), int(w), int(h)]
    
    return None

def get_centroid(bbox):
    if bbox is None:
        return None
    x, y, w, h = bbox
    centroid_x = x + w // 2
    centroid_y = y + h // 2
    return (centroid_x, centroid_y)

def interpolate_missing_centroids(centroids):
    result = centroids.copy()
    n = len(result)
    
    for i in range(n):
        if result[i] is None:
            prev_idx = None
            next_idx = None
            
            for j in range(i - 1, -1, -1):
                if result[j] is not None:
                    prev_idx = j
                    break
            
            for j in range(i + 1, n):
                if result[j] is not None:
                    next_idx = j
                    break
            
            if prev_idx is not None and next_idx is not None:
                prev_x, prev_y = result[prev_idx]
                next_x, next_y = result[next_idx]
                ratio = (i - prev_idx) / (next_idx - prev_idx)
                interp_x = prev_x + (next_x - prev_x) * ratio
                interp_y = prev_y + (next_y - prev_y) * ratio
                result[i] = (interp_x, interp_y)
            elif prev_idx is not None:
                result[i] = result[prev_idx]
            elif next_idx is not None:
                result[i] = result[next_idx]
    
    return result

def main():
    start_time = time.time()
    
    frames_dir = Path("frames")
    output_file = "person_centroids.pkl"
    
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"Processing {len(frame_files)} frames for person detection...")
    
    # First pass: compute median frame for background subtraction
    print("Computing median frame for background...")
    frames_sample = []
    for i, frame_file in enumerate(frame_files):
        if i % 10 == 0:  # Sample every 10th frame
            frame = cv2.imread(str(frame_file))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            frames_sample.append(gray)
    
    median_frame = np.median(frames_sample, axis=0).astype(np.uint8)
    print(f"Computed median frame from {len(frames_sample)} samples")
    
    # Second pass: detect person
    frame_data = []
    detection_count_hog = 0
    detection_count_motion = 0
    detection_count_total = 0
    
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Try HOG first
        bbox = detect_person_hog(frame)
        if bbox is not None:
            detection_count_hog += 1
            detection_count_total += 1
        else:
            # Try motion-based detection
            bbox = detect_person_motion(frame, gray, median_frame)
            if bbox is not None:
                detection_count_motion += 1
                detection_count_total += 1
        
        if bbox is not None:
            centroid = get_centroid(bbox)
        else:
            centroid = None
        
        frame_data.append({
            'frame_file': frame_file.name,
            'bbox': bbox,
            'centroid': centroid
        })
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames, detected {detection_count_total} people (HOG: {detection_count_hog}, Motion: {detection_count_motion})")
    
    detection_rate = detection_count_total / len(frame_files) * 100
    print(f"\nDetection complete: {detection_count_total}/{len(frame_files)} frames ({detection_rate:.1f}%)")
    print(f"  HOG detections: {detection_count_hog}")
    print(f"  Motion detections: {detection_count_motion}")
    
    centroids = [d['centroid'] for d in frame_data]
    interpolated_centroids = interpolate_missing_centroids(centroids)
    
    interpolated_count = sum(1 for i, c in enumerate(interpolated_centroids) if c is not None and centroids[i] is None)
    print(f"Interpolated {interpolated_count} missing centroids")
    
    for i, centroid in enumerate(interpolated_centroids):
        frame_data[i]['interpolated_centroid'] = centroid
    
    with open(output_file, 'wb') as f:
        pickle.dump(frame_data, f)
    
    elapsed = time.time() - start_time
    print(f"\nSaved centroid data to {output_file}")
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
