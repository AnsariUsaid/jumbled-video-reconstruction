import cv2
import os
import pickle
import time
from tqdm import tqdm
from logger import ExecutionLogger


def main():
    logger = ExecutionLogger("execution_log.txt")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    logger.log("=" * 60)
    logger.log("COMPLETE PIPELINE EXECUTION")
    logger.log("=" * 60)
    
    overall_start = time.time()
    
    # Phase 1: Check setup
    logger.start_phase("Phase 1: Project Setup Verification")
    video_path = os.path.join(project_root, "jumbled_video.mp4")
    frames_dir = os.path.join(project_root, "frames")
    output_dir = os.path.join(project_root, "output")
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.end_phase("Phase 1: Project Setup Verification", "Directories verified")
    
    # Phase 2: Extract Frames
    logger.start_phase("Phase 2: Frame Extraction")
    from extract_frames import extract_frames
    frame_count, fps, resolution = extract_frames(video_path, frames_dir)
    logger.end_phase("Phase 2: Frame Extraction", f"{frame_count} frames extracted")
    
    # Phase 3: Extract ORB Features
    logger.start_phase("Phase 3: ORB Feature Extraction")
    from extract_features import load_and_process_frames, save_features
    frames_data = load_and_process_frames(frames_dir)
    features_file = os.path.join(project_root, "frames_features.pkl")
    save_features(frames_data, features_file)
    total_keypoints = sum(f['num_keypoints'] for f in frames_data)
    logger.end_phase("Phase 3: ORB Feature Extraction", 
                    f"{total_keypoints} total keypoints, avg {total_keypoints/len(frames_data):.0f} per frame")
    
    # Phase 4: Build Similarity Matrix
    logger.start_phase("Phase 4: Similarity Matrix Construction")
    from build_similarity_matrix import build_similarity_matrix, save_similarity_matrix
    similarity_matrix = build_similarity_matrix(frames_data)
    matrix_file = os.path.join(project_root, "similarity_matrix.npy")
    save_similarity_matrix(similarity_matrix, matrix_file)
    logger.end_phase("Phase 4: Similarity Matrix Construction", 
                    f"{len(similarity_matrix)}x{len(similarity_matrix)} matrix")
    
    # Phase 5A: Determine Frame Order
    logger.start_phase("Phase 5A: Frame Order Optimization")
    from order_frames import find_optimal_path_graph_approach, save_frame_order
    optimal_order, score = find_optimal_path_graph_approach(similarity_matrix)
    order_file = os.path.join(project_root, "frame_order.pkl")
    save_frame_order(optimal_order, frames_data, order_file)
    avg_similarity = score / (len(optimal_order) - 1)
    logger.end_phase("Phase 5A: Frame Order Optimization", 
                    f"Path score: {score}, Avg similarity: {avg_similarity:.2f}")
    
    # Phase 5B: Reconstruct Video
    logger.start_phase("Phase 5B: Video Reconstruction")
    from reconstruct_video import reconstruct_video, load_frame_order
    order_data = load_frame_order(order_file)
    output_video = os.path.join(output_dir, "reconstructed_video.mp4")
    reconstruct_video(order_data, frames_dir, output_video, fps=fps)
    video_size = os.path.getsize(output_video) / (1024 * 1024)
    logger.end_phase("Phase 5B: Video Reconstruction", 
                    f"Output: {video_size:.2f}MB, {fps} FPS")
    
    # Summary
    total_time = time.time() - overall_start
    logger.log("")
    logger.log("=" * 60)
    logger.log("PIPELINE EXECUTION COMPLETE")
    logger.log("=" * 60)
    logger.log(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.log(f"Input: {frame_count} jumbled frames")
    logger.log(f"Output: Reconstructed video at {output_video}")
    logger.log(f"Quality: {avg_similarity:.2f}/500 average frame similarity")
    logger.log("=" * 60)
    
    logger.save()
    
    print("\n✓ All phases completed successfully!")
    print(f"✓ Execution log saved to: execution_log.txt")
    print(f"✓ Reconstructed video: {output_video}")


if __name__ == "__main__":
    main()
