import os
from step_1_run_yolo_bytetrack import run_yolo_bytetrack
from step_2_build_motion_model import build_motion_model
from step_3_calculate_smoothness_cost import calculate_smoothness_cost
from step_4_solve_optimal_order import solve_optimal_order
from step_5_reconstruct_video import reconstruct_video

if __name__ == '__main__':
    # Define paths
    video_path = '../../jumbled_video.mp4'
    output_dir = '../../output'
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define intermediate file paths
    tracking_data_path = os.path.join(output_dir, 'v8_tracking_data.csv')
    motion_model_path = os.path.join(output_dir, 'v8_motion_model.csv')
    cost_matrix_path = os.path.join(output_dir, 'v8_cost_matrix.npy')
    optimal_order_path = os.path.join(output_dir, 'v8_optimal_order.csv')
    output_video_path = os.path.join(output_dir, 'reconstructed_video_v8.mp4')

    # Run the pipeline
    print("--- Running Step 1: YOLO+ByteTrack Tracking ---")
    run_yolo_bytetrack(video_path, tracking_data_path)

    print("\n--- Running Step 2: Building Motion Model ---")
    build_motion_model(tracking_data_path, motion_model_path)

    print("\n--- Running Step 3: Calculating Hybrid Motion + SSIM Cost ---")
    # Hybrid cost with more weight on SSIM for better perceptual quality
    # alpha=0.3 for motion, beta=0.7 for SSIM (prioritize visual similarity)
    calculate_smoothness_cost(motion_model_path, cost_matrix_path, video_path, 
                            alpha=0.3, beta=0.7, use_ssim=True)

    print("\n--- Running Step 4: Solving for Optimal Order with 2-opt ---")
    solve_optimal_order(cost_matrix_path, optimal_order_path)

    print("\n--- Running Step 5: Reconstructing Video ---")
    reconstruct_video(optimal_order_path, video_path, output_video_path)

    print("\n--- V8 Hybrid Pipeline Complete! ---")
