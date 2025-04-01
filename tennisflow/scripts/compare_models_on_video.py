#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from pathlib import Path
from tqdm import tqdm
import tempfile
from datetime import datetime
from collections import Counter, defaultdict
from scipy.signal import savgol_filter
import ssl

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Handle SSL verification issue for macOS Python
# This is needed for downloading TensorFlow Hub models
ssl._create_default_https_context = ssl._create_unverified_context

from tennisflow.src.classification.rnn_classifier import RNNShotClassifier
from tennisflow.src.pose_estimation.movenet import MoveNetPoseEstimator
from tennisflow.src.utils.visualization import draw_keypoints

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("compare_models")

# Shot class colors
SHOT_COLORS = {
    'forehand': 'green',
    'backhand': 'blue',
    'serve': 'red',
    'volley': 'orange',
    'neutral': 'gray'
}

def extract_frames(video_path):
    """
    Extract all frames from a video
    
    Args:
        video_path: Path to the video file
    
    Returns:
        List of frames as numpy arrays
    """
    logging.info(f"Extracting frames from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        logging.error(f"Error opening video file {video_path}")
        return frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
    
    cap.release()
    logging.info(f"Extracted {len(frames)} frames")
    return frames

def draw_prediction(frame, prediction, position=(50, 50), font_scale=1.0, thickness=2):
    """
    Draw model prediction on a frame
    
    Args:
        frame: Input frame
        prediction: Dictionary with shot class and confidence
        position: Position for text
        font_scale: Font scale for text
        thickness: Text thickness
        
    Returns:
        Frame with prediction drawn
    """
    vis_frame = frame.copy()
    
    # Get shot class and confidence
    shot_class = prediction.get('class', 'unknown')
    confidence = prediction.get('confidence', 0.0)
    
    # Set color based on shot class
    color_map = {
        'forehand': (0, 255, 0),     # Green in BGR
        'backhand': (255, 0, 0),     # Blue in BGR
        'serve': (0, 0, 255),        # Red in BGR
        'volley': (0, 165, 255),     # Orange in BGR
        'neutral': (128, 128, 128)   # Gray in BGR
    }
    color = color_map.get(shot_class, (255, 255, 255))
    
    # Draw text with background rectangle
    text = f"{shot_class.upper()} ({confidence:.2f})"
    text_size = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )[0]
    
    # Draw background rectangle
    cv2.rectangle(
        vis_frame,
        (position[0] - 10, position[1] - text_size[1] - 10),
        (position[0] + text_size[0] + 10, position[1] + 10),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        vis_frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    return vis_frame

def normalize_poses(poses):
    """
    Normalize pose keypoints to be centered and scaled
    
    Args:
        poses: List of pose keypoints from the pose estimator
        
    Returns:
        Normalized pose keypoints
    """
    logging.info("Normalizing poses")
    
    normalized_poses = []
    
    for pose in poses:
        if not pose.get('keypoints', []):
            # Skip invalid poses
            normalized_poses.append([])
            continue
            
        # Extract keypoint coordinates
        keypoints = []
        for kp in pose.get('keypoints', []):
            if 'position' in kp:
                keypoints.append([kp['position']['x'], kp['position']['y']])
            
        if not keypoints:
            normalized_poses.append([])
            continue
            
        # Convert to numpy array
        keypoints_array = np.array(keypoints)
        
        # Center and scale
        min_coords = np.min(keypoints_array, axis=0)
        max_coords = np.max(keypoints_array, axis=0)
        
        # Avoid division by zero
        range_coords = max_coords - min_coords
        range_coords = np.where(range_coords > 0, range_coords, 1.0)
        
        # Normalize to [0, 1]
        normalized_keypoints = (keypoints_array - min_coords) / range_coords
        
        normalized_poses.append(normalized_keypoints)
    
    return normalized_poses

def apply_filter(poses, filter_type='savgol'):
    """
    Apply temporal filtering to pose keypoints
    
    Args:
        poses: List of normalized pose keypoints
        filter_type: Type of filter to apply ('savgol' or 'none')
        
    Returns:
        Filtered pose keypoints
    """
    logging.info(f"Applying {filter_type} filter to poses")
    
    if filter_type == 'none':
        return poses
        
    filtered_poses = []
    
    # Find valid poses (non-empty)
    valid_poses = [i for i, pose in enumerate(poses) if isinstance(pose, np.ndarray) and pose.size > 0]
    
    if not valid_poses:
        logging.warning("No valid poses found for filtering")
        return poses
    
    # Get the first valid pose to determine dimensions
    first_valid_pose = poses[valid_poses[0]]
    
    # Check if the pose array is properly shaped
    if len(first_valid_pose.shape) < 2:
        logging.warning(f"Invalid pose shape: {first_valid_pose.shape}, cannot filter")
        return poses
        
    # Create a time series for each keypoint coordinate
    # First, determine the number of keypoints from a valid pose
    num_keypoints = first_valid_pose.shape[0]
    
    # Initialize arrays for x and y coordinates
    x_coords = np.zeros((len(poses), num_keypoints))
    y_coords = np.zeros((len(poses), num_keypoints))
    
    # Fill in coordinates where available
    for i, pose in enumerate(poses):
        if isinstance(pose, np.ndarray) and pose.size > 0:
            # Handle potential dimension mismatch
            if len(pose.shape) < 2:
                logging.warning(f"Skipping malformed pose at index {i} with shape {pose.shape}")
                continue
                
            # Get actual number of keypoints in this pose (may be different from first valid pose)
            pose_keypoints = min(pose.shape[0], num_keypoints)
            
            # Ensure we have at least 2 dimensions for indexing
            if len(pose.shape) >= 2 and pose.shape[1] >= 2:
                x_coords[i, :pose_keypoints] = pose[:pose_keypoints, 0]
                y_coords[i, :pose_keypoints] = pose[:pose_keypoints, 1]
    
    # Apply Savitzky-Golay filter
    if filter_type == 'savgol':
        window_size = min(15, len(poses) - (len(poses) % 2) - 1)  # Must be odd and < len(poses)
        if window_size < 3:
            window_size = 3
            
        polynomial_order = min(3, window_size - 1)
        
        # Only filter coordinates with valid data
        for j in range(num_keypoints):
            # Get mask of valid frames for this keypoint
            valid_x = np.array([
                i < len(poses) and isinstance(poses[i], np.ndarray) and 
                poses[i].size > 0 and len(poses[i].shape) >= 2 and 
                j < poses[i].shape[0] for i in range(len(poses))
            ])
            valid_y = valid_x.copy()
            
            if np.sum(valid_x) > window_size:
                # Create coordinate arrays for this keypoint
                kp_x = x_coords[:, j].copy()
                kp_y = y_coords[:, j].copy()
                
                # Apply filter to valid regions
                try:
                    if np.sum(valid_x) > window_size:
                        x_coords[valid_x, j] = savgol_filter(
                            kp_x[valid_x], window_size, polynomial_order
                        )
                    if np.sum(valid_y) > window_size:
                        y_coords[valid_y, j] = savgol_filter(
                            kp_y[valid_y], window_size, polynomial_order
                        )
                except Exception as e:
                    logging.warning(f"Error applying filter to keypoint {j}: {e}")
    
    # Reconstruct poses
    for i in range(len(poses)):
        if i < len(poses) and isinstance(poses[i], np.ndarray) and poses[i].size > 0:
            try:
                filtered_pose = np.zeros_like(poses[i])
                
                # Ensure we respect the original shape of each pose
                pose_keypoints = min(poses[i].shape[0], num_keypoints)
                
                if len(poses[i].shape) >= 2 and poses[i].shape[1] >= 2:
                    filtered_pose[:pose_keypoints, 0] = x_coords[i, :pose_keypoints]
                    filtered_pose[:pose_keypoints, 1] = y_coords[i, :pose_keypoints]
                    
                filtered_poses.append(filtered_pose)
            except Exception as e:
                logging.warning(f"Error reconstructing filtered pose {i}: {e}")
                filtered_poses.append(poses[i])  # Use original if filtering fails
        else:
            filtered_poses.append(poses[i] if i < len(poses) else [])
    
    return filtered_poses

class ModelComparer:
    """Class to compare multiple models on a video."""
    
    def __init__(self, model_paths, static_poses_dir=None, window_size=30):
        """
        Initialize the model comparer.
        
        Args:
            model_paths: Dictionary mapping model names to model file paths
            static_poses_dir: Directory containing static pose data
            window_size: Size of the sliding window for shot sequences
        """
        self.model_paths = model_paths
        self.static_poses_dir = static_poses_dir
        self.window_size = window_size
        self.models = {}
        self.pose_estimator = MoveNetPoseEstimator()
        self.static_poses = None
        
        # Load all models
        for model_name, model_path in model_paths.items():
            logger.info(f"Loading model {model_name} from {model_path}")
            model = RNNShotClassifier()
            model.load(model_path)
            self.models[model_name] = model
            
        # Load static poses if available
        if static_poses_dir:
            self.load_static_poses()
    
    def load_static_poses(self):
        """Load static poses for similarity features."""
        logger.info(f"Loading static poses from {self.static_poses_dir}")
        self.static_poses = {}
        shot_types = ['forehand', 'backhand', 'serve', 'neutral']
        
        for shot_type in shot_types:
            # Handle 'neutral' special case
            folder_name = shot_type
            if folder_name == 'neutral':
                folder_name = 'ready_position'
                
            json_path = os.path.join(self.static_poses_dir, f"{folder_name}.json")
            
            if not os.path.exists(json_path):
                logger.warning(f"Static pose data not found for {shot_type} at {json_path}")
                continue
                
            try:
                with open(json_path, 'r') as f:
                    pose_data = json.load(f)
                    
                # Extract keypoints from each annotation
                keypoints_list = []
                
                for annotation in pose_data:
                    if 'keypoints' in annotation:
                        # Keypoints are stored as [x1, y1, v1, x2, y2, v2, ...]
                        keypoints = annotation['keypoints']
                        
                        # Reshape to [x, y, v] format and filter to only include points with visibility
                        kp_array = np.array(keypoints).reshape(-1, 3)
                        visible_kp = kp_array[kp_array[:, 2] > 0][:, :2]  # Only x,y for visible points
                        
                        if len(visible_kp) > 0:
                            # Normalize coordinates
                            x_min, y_min = visible_kp.min(axis=0)
                            x_max, y_max = visible_kp.max(axis=0)
                            
                            width = max(1, x_max - x_min)
                            height = max(1, y_max - y_min)
                            
                            normalized_kp = (visible_kp - [x_min, y_min]) / [width, height]
                            keypoints_list.append(normalized_kp)
                
                if keypoints_list:
                    # Preprocess to standard format
                    standardized_poses = []
                    n_keypoints = 13  # Standard number of keypoints
                    
                    for pose in keypoints_list:
                        # If pose has fewer than n_keypoints, pad with zeros
                        if len(pose) < n_keypoints:
                            padded_pose = np.zeros((n_keypoints, 2))
                            padded_pose[:len(pose)] = pose
                            standardized_poses.append(padded_pose)
                        # If pose has more than n_keypoints, truncate
                        elif len(pose) > n_keypoints:
                            standardized_poses.append(pose[:n_keypoints])
                        else:
                            standardized_poses.append(pose)
                            
                    # Store a sample of standardized poses (up to 10)
                    n_samples = min(10, len(standardized_poses))
                    self.static_poses[shot_type] = np.array(standardized_poses[:n_samples])
                    logger.info(f"Loaded {n_samples} static poses for {shot_type}")
            except Exception as e:
                logger.error(f"Error loading static pose data for {shot_type}: {e}")
    
    def compute_similarity_features(self, sequence, shot_types):
        """
        Compute similarity features between a sequence and static poses.
        
        Args:
            sequence: Sequence of poses
            shot_types: List of shot type names
            
        Returns:
            Similarity features
        """
        if self.static_poses is None or len(self.static_poses) == 0:
            # Return zero features if no static poses
            return np.zeros((sequence.shape[0], len(shot_types)))
            
        seq_length = sequence.shape[0]
        n_shot_types = len(shot_types)
        similarity_features = np.zeros((seq_length, n_shot_types))
        
        # For each frame in the sequence
        for i in range(seq_length):
            frame = sequence[i]  # (n_keypoints, coords)
            
            # For each shot type
            for j, shot_type in enumerate(shot_types):
                if shot_type not in self.static_poses:
                    continue
                    
                # Compute similarity with each reference pose for this shot type
                reference_poses = self.static_poses[shot_type]
                
                similarities = []
                for ref_pose in reference_poses:
                    # Simple Euclidean distance-based similarity
                    try:
                        # Reshape if necessary to match dimensions
                        if frame.shape != ref_pose.shape:
                            min_keypoints = min(frame.shape[0], ref_pose.shape[0])
                            frame_subset = frame[:min_keypoints]
                            ref_pose_subset = ref_pose[:min_keypoints]
                            dist = np.mean(np.sqrt(np.sum((frame_subset - ref_pose_subset)**2, axis=1)))
                        else:
                            dist = np.mean(np.sqrt(np.sum((frame - ref_pose)**2, axis=1)))
                        
                        # Convert distance to similarity (1 for identical, 0 for very different)
                        similarity = np.exp(-dist * 5.0)  # Scale factor to make differences more pronounced
                        similarities.append(similarity)
                    except Exception as e:
                        logger.warning(f"Error computing similarity: {e}")
                        similarities.append(0.0)
                
                # Use the highest similarity among all reference poses
                if similarities:
                    similarity_features[i, j] = max(similarities)
        
        return similarity_features
    
    def process_video(self, video_path, output_dir):
        """
        Process a video and compare model predictions.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save outputs
        
        Returns:
            Dictionary with model comparison results
        """
        logger.info(f"Processing video: {video_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames from the video
        frames = extract_frames(video_path)
        logger.info(f"Extracted {len(frames)} frames from video")
        
        # Extract poses from frames
        poses = []
        for frame in tqdm(frames, desc="Extracting poses"):
            pose = self.pose_estimator.estimate_pose(frame)
            poses.append(pose)
        
        # Normalize poses
        normalized_poses = normalize_poses(poses)
        
        # Apply filter
        filtered_poses = apply_filter(normalized_poses, filter_type='savgol')
        
        # Create sliding windows
        sequences = []
        for i in range(len(filtered_poses) - self.window_size + 1):
            seq = filtered_poses[i:i+self.window_size]
            # Convert list of sequences to a standardized format
            standardized_seq = self._standardize_sequence(seq)
            if standardized_seq is not None:
                sequences.append(standardized_seq)
        
        logger.info(f"Created {len(sequences)} sequence windows")
        
        # Prepare data for Model 4 (if needed)
        model4_similarity_features = None
        if 'Model 4' in self.models and self.static_poses:
            shot_types = ['forehand', 'backhand', 'serve', 'neutral']
            sequences_array = np.array(sequences)
            model4_similarity_features = np.zeros((len(sequences), self.window_size, len(shot_types)))
            
            for i, seq in enumerate(tqdm(sequences_array, desc="Computing similarity features")):
                model4_similarity_features[i] = self.compute_similarity_features(seq, shot_types)
        
        # Run each model on the sequences
        results = {}
        model_predictions = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Running {model_name} on {len(sequences)} sequences")
            
            # Get shot types for this model
            shot_types = model.config['shot_types']
            
            # Process sequences
            predictions = []
            confidences = []
            
            for i, seq in enumerate(tqdm(sequences, desc=f"Processing with {model_name}")):
                # Special handling for Model 4
                if model_name == 'Model 4' and model4_similarity_features is not None:
                    # Reshape for Model 4
                    similarity_feature = model4_similarity_features[i]
                    
                    # Predict with CPU (avoiding GPU issues)
                    with tf.device('/cpu:0'):
                        # Add batch dimension
                        X = np.expand_dims(seq, axis=0)
                        # Preprocess
                        X_processed = model.preprocess_input(X)
                        # Add batch dimension to similarity features
                        S = np.expand_dims(similarity_feature, axis=0)
                        
                        # Predict with both inputs
                        probs = model.model.predict({
                            'sequence_input': X_processed,
                            'similarity_input': S
                        }, verbose=0)[0]
                    
                    class_idx = np.argmax(probs)
                    confidence = probs[class_idx]
                else:
                    # Standard prediction for other models
                    shot_type, confidence = model.predict_sequence(seq)
                    class_idx = shot_types.index(shot_type)
                
                # Get shot type
                shot_type = shot_types[class_idx]
                
                predictions.append(shot_type)
                confidences.append(float(confidence))
            
            # Store results
            model_predictions[model_name] = {
                'predictions': predictions,
                'confidences': confidences,
                'shot_types': shot_types
            }
            
            # Calculate statistics
            prediction_counts = Counter(predictions)
            total = len(predictions)
            percentages = {shot: count / total * 100 for shot, count in prediction_counts.items()}
            
            results[model_name] = {
                'counts': {k: int(v) for k, v in prediction_counts.items()},
                'percentages': {k: float(v) for k, v in percentages.items()}
            }
            
            # Plot distribution
            plt.figure(figsize=(10, 6))
            plt.bar(percentages.keys(), percentages.values())
            plt.title(f"{model_name} Shot Distribution")
            plt.xlabel("Shot Type")
            plt.ylabel("Percentage (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_distribution.png"))
            plt.close()
        
        # Create comparison visualization
        self.create_comparison_visualization(
            frames, model_predictions, output_dir, 
            output_filename=f"comparison_{Path(video_path).stem}.mp4"
        )
        
        # Plot confidence over time for each model
        plt.figure(figsize=(10, 6))
        for model_name, data in model_predictions.items():
            plt.plot(data['confidences'], label=model_name)
        
        plt.title("Model Confidence Over Time")
        plt.xlabel("Frame Window")
        plt.ylabel("Confidence")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_comparison.png"))
        plt.close()
        
        # Save results to JSON
        with open(os.path.join(output_dir, "model_comparison_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def create_comparison_visualization(self, frames, model_predictions, output_dir, output_filename):
        """
        Create a video visualization comparing model predictions.
        
        Args:
            frames: List of video frames
            model_predictions: Dictionary of predictions for each model
            output_dir: Directory to save the output
            output_filename: Name of the output video file
        """
        logger.info("Creating comparison visualization")
        
        # Get window size and number of frames
        window_size = self.window_size
        num_frames = len(frames)
        num_sequences = len(next(iter(model_predictions.values()))['predictions'])
        
        if not frames:
            logger.error("No frames to process for visualization.")
            return

        # Check orientation and rotate if necessary
        h, w = frames[0].shape[:2]
        rotated = False
        if h > w:
            logger.info("Detected portrait video, rotating frames for visualization.")
            rotated = True
            # Swap height and width for rotated frame
            h, w = w, h 

        # Initialize video writer
        output_path = os.path.join(output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Use the potentially swapped dimensions
        height, width = h, w
        
        # Calculate grid layout
        models_count = len(model_predictions)
        grid_rows = 3 # Ensure enough rows for original + 4 models
        grid_cols = 2
        
        # Create larger frame to hold grid
        grid_width = width * grid_cols
        grid_height = height * grid_rows
        
        video_writer = cv2.VideoWriter(output_path, fourcc, 30, (grid_width, grid_height))
        
        # Process each frame
        for i in tqdm(range(num_frames - window_size + 1), desc="Creating comparison video"):
            # Get the current frame and rotate if needed
            base_frame = frames[i + window_size - 1].copy()
            if rotated:
                base_frame = cv2.rotate(base_frame, cv2.ROTATE_90_CLOCKWISE)
            
            current_frame = base_frame.copy() # Use rotated frame as base for models
            
            # Create grid frame
            grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Add original (rotated) frame at position (0, 0)
            grid_frame[0:height, 0:width] = current_frame
            
            # Add model frames
            model_idx = 1
            for model_name, data in model_predictions.items():
                if i >= len(data['predictions']):
                    continue
                    
                # Get prediction for this frame
                prediction = data['predictions'][i]
                confidence = data['confidences'][i]
                
                # Create frame with prediction overlay
                model_frame = current_frame.copy() # Start with the rotated frame
                
                # Determine shot color
                shot_color = SHOT_COLORS.get(prediction, 'white')
                
                # --- Improved Text Overlay ---
                text = f"{model_name}: {prediction} ({confidence:.2f})"
                font_scale = 1.0 # Increased font scale
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (20, 50) # Adjusted position
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw background rectangle (semi-transparent black)
                rect_start = (text_position[0] - 10, text_position[1] - text_height - 10)
                rect_end = (text_position[0] + text_width + 10, text_position[1] + baseline + 10)
                overlay = model_frame.copy()
                cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1)
                alpha = 0.6 # Transparency factor
                model_frame = cv2.addWeighted(overlay, alpha, model_frame, 1 - alpha, 0)
                
                # Draw text (white)
                cv2.putText(
                    model_frame, text, text_position, 
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
                )
                # --- End Improved Text Overlay ---

                # Add colored border
                border_color = self.get_color_bgr(shot_color)
                border_size = 10
                model_frame = cv2.copyMakeBorder(
                    model_frame, border_size, border_size, border_size, border_size,
                    cv2.BORDER_CONSTANT, value=border_color
                )
                
                # Resize back to original dimensions (ensure it matches grid slot size)
                model_frame = cv2.resize(model_frame, (width, height))
                
                # Calculate grid position
                row = model_idx // grid_cols
                col = model_idx % grid_cols
                
                # Add to grid
                grid_frame[row*height:(row+1)*height, col*width:(col+1)*width] = model_frame
                
                model_idx += 1
            
            # Write frame to video
            video_writer.write(grid_frame)
        
        # Release video writer
        video_writer.release()
        logger.info(f"Comparison video saved to {output_path}")
    
    def get_color_bgr(self, color_name):
        """Convert color name to BGR tuple for OpenCV."""
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'orange': (0, 165, 255),
            'gray': (128, 128, 128),
            'white': (255, 255, 255)
        }
        return color_map.get(color_name, (255, 255, 255))
    
    def _standardize_sequence(self, sequence_list):
        """
        Standardize a sequence to ensure consistent shape for model input.
        
        Args:
            sequence_list: List of pose arrays that may have inhomogeneous shapes
            
        Returns:
            Standardized sequence as numpy array with shape based on model requirements
            or None if the sequence can't be standardized
        """
        # Get expected input dimensions from the models
        first_model_name = next(iter(self.models))
        first_model = self.models[first_model_name]
        
        # Check if model1 is in a special format requiring (seq_len, n_keypoints, 2) format
        model1_needs_special_format = False
        if 'Model 1' in self.models:
            model1 = self.models['Model 1']
            if hasattr(model1.model, 'inputs'):
                try:
                    # Handle different shape representations safely
                    shape_str = str(model1.model.inputs[0].shape)
                    # Check for specific dimensions in the shape string
                    if '13, 2' in shape_str:
                        model1_needs_special_format = True
                        logger.info(f"Model 1 requires special input format based on shape: {shape_str}")
                except Exception as e:
                    logger.warning(f"Error checking model input shape: {e}")
                    # Continue with standard format
        
        # Use appropriate shape based on model requirements
        if model1_needs_special_format and first_model_name == 'Model 1':
            # For Model 1 with (seq_len, n_keypoints, 2) format
            seq_len = len(sequence_list)
            n_keypoints = 13  # As indicated by the model's input shape
            standardized = np.zeros((seq_len, n_keypoints, 2))
            
            # For each frame in the sequence
            for i, frame_pose in enumerate(sequence_list):
                if not isinstance(frame_pose, np.ndarray) or frame_pose.size == 0:
                    continue
                    
                # If the pose is already in 2D format (n_keypoints, 2)
                if len(frame_pose.shape) == 2 and frame_pose.shape[1] == 2:
                    # Copy as many keypoints as possible
                    keypoints_to_copy = min(frame_pose.shape[0], n_keypoints)
                    standardized[i, :keypoints_to_copy, :] = frame_pose[:keypoints_to_copy, :]
                # If the pose is flattened
                elif len(frame_pose.shape) == 2 and frame_pose.shape[1] > 2:
                    # Assume flattened format (n_keypoints*2)
                    flattened = frame_pose.flatten()
                    # Reshape to (n_keypoints, 2) format
                    try:
                        reshaped = flattened[:n_keypoints*2].reshape(-1, 2)
                        keypoints_to_copy = min(reshaped.shape[0], n_keypoints)
                        standardized[i, :keypoints_to_copy, :] = reshaped[:keypoints_to_copy, :]
                    except ValueError:
                        logger.warning(f"Could not reshape pose with shape {frame_pose.shape} to (n_keypoints, 2) format")
        else:
            # Standard format (seq_len, features_per_frame)
            expected_features = first_model.config.get('features_per_frame', 34)
            seq_len = len(sequence_list)
            standardized = np.zeros((seq_len, expected_features))
            
            # For each frame in the sequence
            for i, frame_pose in enumerate(sequence_list):
                if not isinstance(frame_pose, np.ndarray) or frame_pose.size == 0:
                    continue
                    
                # If the pose has the right shape, copy it directly
                if len(frame_pose.shape) == 2:
                    # Flatten the keypoints if needed
                    flattened = frame_pose.flatten()
                    # Copy as many features as possible
                    features_to_copy = min(flattened.shape[0], expected_features)
                    standardized[i, :features_to_copy] = flattened[:features_to_copy]
        
        return standardized

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare multiple models on a tennis video')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--model1', required=True, help='Path to Model 1 (Bidirectional LSTM)')
    parser.add_argument('--model2', required=True, help='Path to Model 2 (Larger Bidirectional LSTM)')
    parser.add_argument('--model3', required=True, help='Path to Model 3 (Unidirectional LSTM)')
    parser.add_argument('--model4', required=True, help='Path to Model 4 (Static Pose Enhanced LSTM)')
    parser.add_argument('--static-poses-dir', help='Directory containing static pose data for Model 4')
    parser.add_argument('--output-dir', default='comparison_results', help='Directory to save outputs')
    parser.add_argument('--window-size', type=int, default=30, help='Size of sliding window')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default='info',
                       help='Logging level')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_dir = os.path.join(args.output_dir, f"{video_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model paths dictionary
    model_paths = {
        'Model 1': args.model1,
        'Model 2': args.model2,
        'Model 3': args.model3,
        'Model 4': args.model4
    }
    
    # Create model comparer
    comparer = ModelComparer(
        model_paths=model_paths,
        static_poses_dir=args.static_poses_dir,
        window_size=args.window_size
    )
    
    # Process video
    results = comparer.process_video(args.video_path, output_dir)
    
    # Print summary
    logger.info("Model Comparison Summary:")
    for model_name, data in results.items():
        logger.info(f"{model_name}:")
        for shot_type, percentage in data['percentages'].items():
            logger.info(f"  {shot_type}: {percentage:.2f}%")
    
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 