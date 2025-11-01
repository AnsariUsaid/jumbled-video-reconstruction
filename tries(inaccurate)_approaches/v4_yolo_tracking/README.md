# V4: YOLO + Nearest Neighbor Approach

## Overview

This approach uses **YOLOv8** for person detection and tracking, combined with a **nearest-neighbor greedy algorithm** for frame ordering.

## Algorithm

### Step 1: Person Detection & Tracking
- Use YOLOv8 (lightweight `yolov8n.pt` model) to detect person in each frame
- Extract bounding box centroid (x, y) and area for each detection
- Store tracking data in CSV format

### Step 2: Frame Ordering
- Start from frame closest to bottom-right corner
- Greedily select nearest unvisited frame based on centroid distance
- Build sequence by always picking the spatially closest next frame

### Step 3: Video Reconstruction
- Load all frames from original video
- Reorder according to computed sequence
- Write reconstructed video

## Why This Approach?

**Advantages:**
- 🎯 **Direct spatial tracking** - Uses actual person position, not image features
- 🚀 **Simple & interpretable** - Easy to understand and debug
- 💡 **Robust to lighting** - YOLO detections work across various conditions
- 📊 **Handles occlusions** - YOLO is trained to detect partially visible people

**Potential Limitations:**
- Requires person to be visible in most frames
- Greedy algorithm may not find globally optimal path
- Relies on spatial proximity assumption (person moves continuously)

## Requirements

```bash
pip install ultralytics opencv-python pandas numpy tqdm
```

The YOLOv8 model will be automatically downloaded on first use (~6 MB for yolov8n).

## Usage

### Quick Start (Run Complete Pipeline)

```bash
source venv/bin/activate
cd src/v4_yolo_tracking
python run_pipeline.py
```

### Step-by-Step Execution

```bash
source venv/bin/activate
cd src/v4_yolo_tracking

# Step 1: Extract tracking data with YOLO
python 1_extract_tracking_data.py

# Step 2: Compute frame order
python 2_compute_frame_order.py

# Step 3: Reconstruct video
python 3_reconstruct_video.py
```

## Output Files

- **`frame_tracking_data.csv`** - Person tracking data for all frames
  ```
  frame,centroid_x,centroid_y,bbox_area
  0,1042.0,778.0,359394.0
  1,962.0,791.0,228000.0
  ...
  ```

- **`correct_frame_order.csv`** - Computed frame sequence
  ```
  ordered_frame
  176
  290
  172
  ...
  ```

- **`output/reconstructed_video_v4.mp4`** - Final reconstructed video

## Comparison with Other Approaches

| Approach | Method | Complexity | Speed | Robustness |
|----------|--------|------------|-------|------------|
| V1 (ORB) | Feature matching | Medium | Fast | Moderate |
| V2 (CNN) | Deep features | High | Slower | High |
| V3 (Distance) | Position tracking | Low | Very Fast | Low |
| **V4 (YOLO)** | **Object detection + NN** | **Medium** | **Fast** | **High** |

## Performance Expectations

- **Step 1** (YOLO Tracking): ~30-60 seconds for 300 frames
- **Step 2** (Frame Ordering): ~5 seconds
- **Step 3** (Reconstruction): ~10 seconds
- **Total**: ~1-2 minutes

## Algorithm Details

### Nearest-Neighbor Greedy Search

1. Calculate distance from each frame's person position to bottom-right corner
2. Start with frame having minimum distance (person closest to corner)
3. While unvisited frames remain:
   - Calculate Euclidean distance from current position to all unvisited frames
   - Select frame with minimum distance
   - Move to that frame and repeat

This creates a path that minimizes total spatial displacement between consecutive frames.

### Distance Formula

```python
distance = sqrt((x2 - x1)² + (y2 - y1)²)
```

Where `(x1, y1)` is current centroid and `(x2, y2)` is candidate frame centroid.

## Troubleshooting

### Person Not Detected in Some Frames
- Frames without detections are excluded from ordering
- Check `frame_tracking_data.csv` for NaN values
- May indicate person is occluded or out of frame

### YOLOv8 Model Download Issues
- Model auto-downloads on first use (requires internet)
- If download fails, manually download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases)
- Place in project root or specify custom path

### Memory Issues
- Video is loaded entirely into RAM for reordering
- For very large videos, consider processing in chunks
- Current implementation optimized for videos under 1GB

## Future Improvements

- [ ] Use YOLO tracking features (ByteTrack) for temporal consistency
- [ ] Implement 2-opt optimization to improve greedy path
- [ ] Add confidence threshold filtering for detections
- [ ] Support multiple person tracking (track specific person ID)
- [ ] Bidirectional greedy search from both start and end

## Credits

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Nearest Neighbor**: Classic greedy algorithm
