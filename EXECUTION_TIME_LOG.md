# Execution Time Log

## System Specifications
- **Device**: MacBook Air M1
- **Processor**: Apple M1 chip
- **RAM**: 8 GB
- **Storage**: 256 GB
- **System Type**: ARM64 architecture

## V1: ORB-based Approach

### Phase 1: Frame Extraction
- **Time**: ~2 seconds
- **Output**: 300 frames extracted from jumbled video

### Phase 2: Feature Extraction (ORB)
- **Time**: ~8 seconds
- **Process**: ORB keypoint and descriptor extraction for 300 frames

### Phase 3: Similarity Matrix Construction
- **Time**: ~25 seconds
- **Process**: Brute-force matching with Hamming distance for all frame pairs

### Phase 4: Graph-based Ordering
- **Time**: ~3 seconds
- **Process**: Greedy nearest neighbor path construction

### Phase 5: Video Reconstruction
- **Time**: ~5 seconds
- **Output**: Reconstructed video (65.4 MB)

**Total V1 Execution Time**: ~43 seconds

---

## V2: Deep Learning Approach (ResNet50-based) ⭐ Best Solution

### Phase 1: Frame Extraction
- **Time**: ~2 seconds
- **Output**: 300 frames extracted from jumbled video

### Phase 2: Feature Extraction (CNN)
- **Time**: ~45 seconds
- **Process**: ResNet50 feature extraction (2048-dimensional vectors per frame)
- **Note**: First run includes model download time (~100 MB)

### Phase 3: Similarity Matrix Construction
- **Time**: ~1 second
- **Process**: Cosine similarity computation for all frame pairs

### Phase 4: Graph-based Ordering with Enhanced Heuristics
- **Time**: ~4 seconds
- **Process**: 
  - Start/end point detection using degree analysis
  - Greedy path construction with bidirectional search
  - Direction validation

### Phase 5: Video Reconstruction
- **Time**: ~5 seconds
- **Output**: Reconstructed video (67.0 MB)

**Total V2 Execution Time**: ~57 seconds (excluding first-time model download)

---

## Performance Comparison

| Metric | V1 (ORB) | V2 (ResNet50) |
|--------|----------|---------------|
| **Total Time** | 43s | 57s |
| **Feature Quality** | Low-level (edges, corners) | High-level (semantic) |
| **Similarity Accuracy** | ~85% | ~99.6% |
| **Memory Usage** | ~500 MB | ~1.2 GB |
| **Reconstruction Quality** | Moderate (many jumps) | Improved (fewer jumps, but still present) |

---

## Optimization Notes

### V1 Optimizations Applied
- Grayscale conversion for faster processing
- Limited ORB features to 500 per frame
- NumPy vectorization for similarity computations

### V2 Optimizations Applied
- Batch processing disabled (single frame processing for stability)
- Pre-loaded ResNet50 model with GlobalAveragePooling
- Efficient cosine similarity using sklearn
- Smart start/end point detection to avoid backward reconstruction

---

## Conclusion

**V2 (Deep Learning Approach)** shows improvement over V1 despite slightly longer execution time because:
- 99.6% average similarity vs 85% in V1
- Semantic understanding leads to more accurate frame ordering
- Better consecutive frame similarity scores
- Trade-off of 14 extra seconds provides 14.6% similarity improvement

**However, both approaches still have limitations:**
- Visible frame jumps remain in reconstructed video
- Not yet suitable for production-quality reconstruction
- Further improvements needed (see V2_Algorithm_Description.md for future work)
