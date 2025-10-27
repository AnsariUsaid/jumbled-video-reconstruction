# Execution Time Log

## System Specifications
- **Device**: MacBook Air M1
- **Processor**: Apple M1 chip
- **RAM**: 8 GB
- **Storage**: 256 GB
- **System Type**: ARM64 architecture

## V1: ORB-based Approach

### Complete Pipeline
- **Total Time**: ~4 minutes
- **Output**: reconstructed_video.mp4 (62MB)
- **Similarity Score**: 89% average frame-to-frame

### Phase Breakdown:
1. **Frame Extraction**: ~5 seconds
2. **Feature Extraction**: ~30 seconds
3. **Similarity Matrix**: ~180 seconds
4. **Frame Ordering**: ~2 seconds
5. **Video Reconstruction**: ~5 seconds

**Key Characteristics:**
- Baseline approach
- Good speed vs quality balance
- No heavy dependencies

---

## V2: Deep Learning Approach (ResNet50) 

### Complete Pipeline
- **Total Time**: ~3 minutes
- **Output**: reconstructed_video_cnn.mp4 (64MB)
- **Similarity Score**: 99.6% average frame-to-frame

### Phase Breakdown:
1. **Frame Extraction**: ~5 seconds (if needed)
2. **CNN Feature Extraction**: ~60 seconds
3. **Similarity Matrix**: ~2 seconds
4. **Frame Ordering**: ~5 seconds
5. **Video Reconstruction**: ~5 seconds

**Key Characteristics:**
- Highest similarity score
- Semantic feature understanding
- Requires TensorFlow

---

## V4: YOLO Object Detection + Nearest Neighbor ⭐ **FASTEST**

### Complete Pipeline
- **Total Time**: ~2 minutes
- **Output**: reconstructed_video_v4.mp4 (54MB)
- **Quality Metric**: 5.7 pixels average step distance

### Phase Breakdown:
1. **YOLO Detection**: ~60 seconds
   - Person detection in 297/300 frames (99%)
   - Extract centroid positions
2. **Frame Ordering**: ~5 seconds
   - Nearest neighbor greedy search
   - Average 5.7px displacement
3. **Video Reconstruction**: ~10 seconds

**Key Characteristics:**
- **Fastest execution**
- **Smallest file size**
- **Excellent visual quality**
- **Direct spatial tracking**

---

## Performance Comparison

| Metric | V1 (ORB) | V2 (ResNet50) | V4 (YOLO) ⭐ |
|--------|----------|---------------|--------------|
| **Total Time** | ~4 min | ~3 min | **~2 min** |
| **File Size** | 62MB | 64MB | **54MB** |
| **Quality Metric** | 89% similarity | 99.6% similarity | **5.7px avg step** |
| **Memory Usage** | ~500 MB | ~1.2 GB | ~800 MB |
| **Detection Rate** | - | - | **99%** |
| **Visual Quality** | Good | Excellent | **Excellent** |
| **Recommended For** | Baseline | High similarity | **Best overall** ⭐ |

---

## Optimization Notes

### V1 Optimizations
- Grayscale conversion for faster processing
- Limited ORB features to 500 per frame
- NumPy vectorization for similarity computations

### V2 Optimizations
- Batch processing for feature extraction
- Pre-loaded ResNet50 model with GlobalAveragePooling
- Efficient cosine similarity using sklearn
- Graph-based optimization

### V4 Optimizations ⭐
- YOLOv8n (lightweight model) for speed
- Direct spatial tracking (no similarity matrix needed)
- Simple greedy nearest-neighbor (O(n²))
- Minimal memory footprint

---

## Conclusion

**V4 (YOLO + Nearest Neighbor) is the recommended approach** because:
- ⚡ **Fastest** (~2 minutes)
- 💾 **Smallest** output (54MB)
- 📹 **Excellent** quality (smooth 5.7px motion)
- 🎯 **Highest** detection rate (99%)
- 🔍 **Simple** and interpretable
- **Best Output and correct result**

**V2 (CNN)** provides the highest similarity score (99.6%) and is excellent for semantic matching.

**V1 (ORB)** serves as a good baseline and learning example with fast execution and minimal dependencies.

---

*Last Updated: October 27, 2024*
