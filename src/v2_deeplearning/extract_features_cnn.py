import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

print("TensorFlow version:", tf.__version__)


def load_resnet_model():
    print("Loading ResNet50 model...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    print("Model loaded successfully")
    return model


def extract_cnn_features(frame, model):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img, verbose=0)
    return features.flatten()


def load_and_process_frames(frames_dir, model):
    print(f"Reading frames from: {frames_dir}")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        print("Error: No frames found")
        return []
    
    print(f"Found {len(frame_files)} frames to process")
    frames_data = []
    
    print("Extracting CNN features...")
    for frame_file in tqdm(frame_files, desc="Processing frames", unit="frame"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read {frame_file}, skipping")
            continue
        
        features = extract_cnn_features(frame, model)
        frame_info = {
            'filename': frame_file,
            'path': frame_path,
            'shape': frame.shape,
            'features': features,
            'feature_dim': len(features)
        }
        frames_data.append(frame_info)
    
    print(f"Successfully processed {len(frames_data)} frames")
    print(f"Feature dimension: {frames_data[0]['feature_dim']}")
    return frames_data


def save_features(frames_data, output_file):
    print(f"Saving CNN features to: {output_file}")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(frames_data, f)
        print("Features saved successfully")
        return True
    except Exception as e:
        print(f"Error saving features: {e}")
        return False


def print_statistics(frames_data):
    print("\n" + "=" * 60)
    print("CNN FEATURE EXTRACTION STATISTICS")
    print("=" * 60)
    
    total_frames = len(frames_data)
    feature_dim = frames_data[0]['feature_dim']
    
    print(f"Total frames: {total_frames}")
    print(f"Feature dimension: {feature_dim}")
    
    all_features = np.array([f['features'] for f in frames_data])
    print(f"\nFeature statistics:")
    print(f"  Mean: {all_features.mean():.4f}")
    print(f"  Std: {all_features.std():.4f}")
    print(f"  Min: {all_features.min():.4f}")
    print(f"  Max: {all_features.max():.4f}")
    print("=" * 60)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    frames_dir = os.path.join(project_root, "frames")
    output_file = os.path.join(project_root, "frames_features_cnn.pkl")
    
    print("=" * 60)
    print("V2: CNN FEATURE EXTRACTION")
    print("=" * 60)
    
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found at {frames_dir}")
        return
    
    model = load_resnet_model()
    frames_data = load_and_process_frames(frames_dir, model)
    
    if len(frames_data) == 0:
        print("No frames processed")
        return
    
    print_statistics(frames_data)
    success = save_features(frames_data, output_file)
    
    if success:
        print("\nPhase 3 V2 Complete")
        print(f"Features saved to: {output_file}")
        print("Next: Run build_similarity_matrix_cnn.py")


if __name__ == "__main__":
    main()
