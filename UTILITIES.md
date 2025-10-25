# Utility Files Documentation

This document explains the purpose and function of the utility files added in Phase 6.

---

## 📝 logger.py - Execution Time Tracker

### Purpose
Tracks and logs execution time for each phase of the pipeline with timestamps.

### Functions

#### `start_phase(phase_name)`
Records the start time when a phase begins.
```python
logger.start_phase("Phase 2: Frame Extraction")
# Output: [2025-10-25 16:20:00] Starting: Phase 2: Frame Extraction
```

#### `end_phase(phase_name, details="")`
Calculates and logs elapsed time when a phase completes.
```python
logger.end_phase("Phase 2: Frame Extraction", "300 frames extracted")
# Output: [2025-10-25 16:20:05] Completed: Phase 2 | Time: 5.23s | 300 frames extracted
```

#### `log(message)`
Adds a timestamped message to the log.
```python
logger.log("Starting similarity computation")
# Output: [2025-10-25 16:20:10] Starting similarity computation
```

#### `save()`
Writes all log entries to `execution_log.txt`.

### Example Output (execution_log.txt)
```
================================================================================
JUMBLED FRAMES RECONSTRUCTION - EXECUTION LOG
================================================================================

[2025-10-25 16:20:00] Starting: Phase 1: Project Setup Verification
[2025-10-25 16:20:00] Completed: Phase 1 | Time: 0.12s | Directories verified
[2025-10-25 16:20:00] Starting: Phase 2: Frame Extraction
[2025-10-25 16:20:05] Completed: Phase 2 | Time: 5.23s | 300 frames extracted
[2025-10-25 16:20:05] Starting: Phase 3: ORB Feature Extraction
[2025-10-25 16:20:35] Completed: Phase 3 | Time: 30.12s | 150000 total keypoints
...
[2025-10-25 16:24:00] Total execution time: 240.45 seconds (4.01 minutes)
================================================================================
```

---

## 🔄 run_pipeline.py - Automated Full Pipeline

### Purpose
Runs the entire video reconstruction pipeline automatically with integrated timing.

### What It Does
1. **Verifies Setup** - Checks directories exist
2. **Extracts Frames** - Calls `extract_frames.py`
3. **Extracts Features** - Calls `extract_features.py`
4. **Builds Matrix** - Calls `build_similarity_matrix.py`
5. **Orders Frames** - Calls `order_frames.py`
6. **Reconstructs Video** - Calls `reconstruct_video.py`
7. **Logs Everything** - Saves timing to `execution_log.txt`

### Benefits
- ✅ **One Command**: Run entire project with one script
- ✅ **Automatic Timing**: No need to manually time each phase
- ✅ **Error Handling**: Stops and logs if any phase fails
- ✅ **Progress Tracking**: See what's happening in real-time
- ✅ **Reproducible**: Same results every time

### Usage
```bash
source venv/bin/activate
python src/run_pipeline.py
```

### Console Output Example
```
============================================================
COMPLETE PIPELINE EXECUTION
============================================================
[2025-10-25 16:20:00] Starting: Phase 1: Project Setup Verification
✓ Created output directory
[2025-10-25 16:20:00] Completed: Phase 1 | Time: 0.12s
[2025-10-25 16:20:00] Starting: Phase 2: Frame Extraction
Video Properties:
  Total Frames: 300
  FPS: 30
  Resolution: 1920x1080
Extracting frames: 100%|████████████| 300/300 [00:05<00:00]
[2025-10-25 16:20:05] Completed: Phase 2 | Time: 5.23s | 300 frames extracted
...
============================================================
PIPELINE EXECUTION COMPLETE
============================================================
Total execution time: 240.45 seconds (4.01 minutes)
✓ All phases completed successfully!
✓ Execution log saved to: execution_log.txt
✓ Reconstructed video: output/reconstructed_video.mp4
```

---

## 📚 Algorithm_Description.md - Technical Documentation

### Purpose
Comprehensive 2-page technical documentation explaining the algorithm, design choices, and trade-offs.

### Sections

#### 1. Executive Summary
Quick overview of the problem and solution approach.

#### 2. Problem Statement
- What we're solving
- Input/output
- Challenges

#### 3. Algorithm Overview
Detailed explanation of each phase:
- **Phase 1**: ORB Feature Extraction
  - Why ORB vs SIFT/SURF
  - How it works
  - Parameters used
  
- **Phase 2**: Similarity Matrix
  - Brute-Force Matching
  - Hamming distance
  - Complexity analysis
  
- **Phase 3**: Frame Ordering
  - Graph-based approach
  - Hamiltonian path problem
  - Hybrid greedy + 2-opt optimization
  
- **Phase 4**: Video Reconstruction
  - Sequential write process

#### 4. Algorithm Trade-offs and Alternatives
Explains why we chose this approach over:
- Deep Learning (CNNs, LSTMs)
- Brute Force Search (factorial complexity)
- Clustering-based approaches
- Pure TSP solvers

#### 5. Performance Metrics
- Execution time breakdown
- Quality metrics
- Space complexity analysis

#### 6. Future Improvements
- Parallel processing
- Adaptive feature counts
- Multi-scale matching
- Temporal consistency checks

### Target Audience
- Developers reviewing the code
- Researchers interested in the approach
- Interviewers evaluating technical knowledge
- Students learning computer vision

### When to Read It
- Before implementing modifications
- When presenting the project
- For understanding design decisions
- When explaining to technical audiences

---

## 📊 Comparison of Approaches

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_pipeline.py` | Run complete pipeline | Production, first-time setup |
| Individual scripts | Run specific phases | Debugging, testing changes |
| `logger.py` | Track timing | Performance analysis |
| `Algorithm_Description.md` | Understand approach | Learning, presentations |

---

## 🎯 Recommended Workflow

### For First-Time Users:
1. Read main `README.md`
2. Run `python src/run_pipeline.py`
3. Check `execution_log.txt` for timing
4. View `output/reconstructed_video.mp4`

### For Developers:
1. Read `Algorithm_Description.md` first
2. Run individual scripts to test changes
3. Use `logger.py` in your own scripts
4. Document improvements

### For Researchers:
1. Start with `Algorithm_Description.md`
2. Examine trade-offs section
3. Review performance metrics
4. Consider suggested improvements

---

## 💡 Tips

### Viewing Logs
```bash
# View execution log
cat execution_log.txt

# Watch log in real-time (while running)
tail -f execution_log.txt
```

### Debugging
If a phase fails:
1. Check `execution_log.txt` for error timestamp
2. Run that specific phase script manually
3. Fix the issue
4. Re-run `run_pipeline.py` (it will skip completed phases)

### Performance Tracking
Compare execution times across runs:
```bash
# Save log with timestamp
cp execution_log.txt logs/run_$(date +%Y%m%d_%H%M%S).txt
```

---

This documentation helps you understand what each utility file does and how to use them effectively!
