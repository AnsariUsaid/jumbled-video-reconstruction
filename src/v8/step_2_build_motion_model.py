
import pandas as pd
import numpy as np
import os

def build_motion_model(input_csv_path, output_csv_path):
    """
    Builds a motion model by calculating motion vectors from person positions.
    Uses all detected frames to build comprehensive motion understanding.

    Args:
        input_csv_path (str): Path to the input tracking data CSV file.
        output_csv_path (str): Path to save the output motion model data.
    """
    # Load the tracking data
    df = pd.read_csv(input_csv_path)

    # Convert to numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['w'] = pd.to_numeric(df['w'], errors='coerce')
    df['h'] = pd.to_numeric(df['h'], errors='coerce')

    # Drop rows with NaN
    df.dropna(subset=['x', 'y'], inplace=True)
    
    print(f"Total detections: {len(df)}")
    print(f"Unique frames: {df['frame'].nunique()}")
    print(f"Detection rate: {len(df)/300*100:.1f}%")
    
    if len(df) == 0:
        print("ERROR: No valid detections found!")
        return
    
    # Sort by frame
    df = df.sort_values(by='frame').reset_index(drop=True)
    
    # Calculate centroid
    df['cx'] = df['x']
    df['cy'] = df['y']
    
    # For each frame, calculate motion vectors to all other frames
    # This creates a pairwise motion prediction matrix
    print(f"\nCalculating motion vectors between all frame pairs...")
    
    # Simple velocity calculation for consecutive detections in current jumbled order
    df['vx_jumbled'] = df['cx'].diff().fillna(0)
    df['vy_jumbled'] = df['cy'].diff().fillna(0)
    df['speed_jumbled'] = np.sqrt(df['vx_jumbled']**2 + df['vy_jumbled']**2)
    
    print(f"Average speed in jumbled order: {df['speed_jumbled'].mean():.2f} pixels/frame")
    
    # Calculate area (person size indicator - closer = bigger)
    df['area'] = df['w'] * df['h']
    
    # Normalize features for motion analysis
    df['cx_norm'] = df['cx'] / 1920.0  # Normalize by video width
    df['cy_norm'] = df['cy'] / 1080.0  # Normalize by video height
    df['area_norm'] = df['area'] / (1920.0 * 1080.0)
    
    # Save the motion model data
    df.to_csv(output_csv_path, index=False)
    print(f"Motion model data saved to {output_csv_path}")
    print(f"Saved {len(df)} frames with motion data")

if __name__ == '__main__':
    # Define paths
    input_csv_path = '../../output/v8B_tracking_data.csv'
    output_dir = '../../output'
    output_csv_path = os.path.join(output_dir, 'v8B_motion_model.csv')

    # Build the motion model
    build_motion_model(input_csv_path, output_csv_path)
