#!/usr/bin/env python
"""
Tennis Video Analysis Script

This script analyzes a tennis video, detecting shots and generating a report
with shot analysis and kinematic data.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import time # Import time for log filename

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import local modules
from src.pose_estimation.pose_estimator import MoveNetEstimator
from src.smoothing.temporal_filter import TemporalFilter
from src.classification.rnn_classifier import RNNShotClassifier
from src.analysis.kinematic_analyzer import KinematicAnalyzer
from src.visualization.video_visualizer import VideoVisualizer
from src.reporting.report_generator import ReportGenerator

# --- Configure Logging --- 
# We will configure file logging inside the analyze_video function 
# once the output directory is created.
# Basic console logging first:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Explicitly log to stdout
)
logger = logging.getLogger(__name__) # Use __name__ for logger
# --- End Basic Logging Config ---

# --- Configuration ---
WINDOW_SIZE = 30 # Sequence length used by the model
SHOT_COLORS = {
    'forehand': 'blue',
    'backhand': 'green',
    'serve': 'red',
    'volley': 'orange',
    'neutral': 'gray'
}
DEFAULT_MODEL_PATH = 'models/model4_classifier.h5'
DEFAULT_STATIC_POSES_DIR = 'prepared_data/static/poses'
DEFAULT_OUTPUT_DIR = 'analysis_results'

class TennisVideoAnalyzer:
    """Tennis video analysis pipeline."""
    
    def __init__(self, config_path: str):
        """
        Initialize the analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.pose_estimator = self._init_pose_estimator()
        self.temporal_filter = self._init_temporal_filter()
        self.classifier = self._init_classifier()
        self.kinematic_analyzer = self._init_kinematic_analyzer()
        self.visualizer = self._init_visualizer()
        self.report_generator = self._init_report_generator()
        
        # Set up paths
        self.output_dir = self.config.get('paths', {}).get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _init_pose_estimator(self) -> MoveNetEstimator:
        """Initialize the pose estimator."""
        model_type = self.config.get('pose_estimation', {}).get('model_type', 'lightning')
        return MoveNetEstimator(model_type=model_type)
    
    def _init_temporal_filter(self) -> TemporalFilter:
        """Initialize the temporal filter."""
        filter_config = self.config.get('smoothing', {})
        return TemporalFilter(filter_config)
    
    def _init_classifier(self) -> RNNShotClassifier:
        """Initialize the classifier."""
        classifier_config = self.config.get('rnn_classifier', {})
        model_path = self.config.get('paths', {}).get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model path not found: {model_path}")
            logger.warning("Using untrained classifier")
            return RNNShotClassifier()
        
        try:
            logger.info(f"Loading classifier from {model_path}")
            return RNNShotClassifier(model_path=model_path)
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return RNNShotClassifier()
    
    def _init_kinematic_analyzer(self) -> KinematicAnalyzer:
        """Initialize the kinematic analyzer."""
        return KinematicAnalyzer()
    
    def _init_visualizer(self) -> VideoVisualizer:
        """Initialize the video visualizer."""
        return VideoVisualizer()
    
    def _init_report_generator(self) -> ReportGenerator:
        """Initialize the report generator."""
        return ReportGenerator()
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process a tennis video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Analysis results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.output_dir, f"{video_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract video information
        video_info = self._extract_video_info(video_path)
        
        # Extract pose keypoints
        keypoints = self._extract_keypoints(video_path)
        
        # Apply temporal smoothing
        smoothed_keypoints = self._apply_smoothing(keypoints)
        
        # Detect shot sequences
        shot_sequences, shot_timestamps = self._detect_shot_sequences(smoothed_keypoints, video_info)
        
        # Classify shots
        shot_classifications = self._classify_shots(shot_sequences)
        
        # Analyze kinematics
        kinematic_data = self._analyze_kinematics(smoothed_keypoints, shot_timestamps)
        
        # Create visualizations
        visualization_paths = self._create_visualizations(
            video_path, 
            smoothed_keypoints, 
            shot_timestamps, 
            shot_classifications,
            output_dir
        )
        
        # Generate report
        report_path = self._generate_report(
            video_info,
            shot_classifications,
            kinematic_data,
            visualization_paths,
            output_dir
        )
        
        # Compile results
        results = {
            'video_info': video_info,
            'shot_classifications': shot_classifications,
            'kinematic_data': kinematic_data,
            'output_dir': output_dir,
            'report_path': report_path,
            'visualization_paths': visualization_paths
        }
        
        logger.info(f"Analysis completed. Results saved to {output_dir}")
        return results
    
    def _extract_video_info(self, video_path: str) -> Dict:
        """Extract basic information from the video."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {}
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
    
    def _extract_keypoints(self, video_path: str) -> np.ndarray:
        """
        Extract pose keypoints from the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Array of keypoints, shape (frames, keypoints, coordinates)
        """
        logger.info("Extracting pose keypoints")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return np.array([])
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints_list = []
        
        for i in range(frame_count):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # +++ Get frame dimensions for normalization +++
            frame_h, frame_w = frame.shape[:2]
            # +++++++++++++++++++++++++++++++++++++++++++++
            
            # Convert frame to RGB for pose estimator
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Estimate pose - This is the correct location for pose estimation and handling
            try:
                keypoints = self.pose_estimator.estimate_pose(frame_rgb)
                
                # +++ LOG ESTIMATED KEYPOINTS (first few frames) +++
                if i < 5: # Log only for the first 5 frames
                    kpts_log = f"[ESTIMATE Frame {i}] Type: {type(keypoints)}"
                    if keypoints is not None:
                        # Check if keypoints is a numpy array before accessing shape/size
                        if isinstance(keypoints, np.ndarray):
                             kpts_log += f", Shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'N/A'}"
                             if keypoints.size > 0:
                                  kpts_log += f", Mean: {keypoints.mean():.4f}, Min: {keypoints.min():.4f}, Max: {keypoints.max():.4f}"
                             else:
                                  kpts_log += ", Empty Array"
                        else:
                             kpts_log += f", Non-array type: {type(keypoints)}" # Log if not numpy array
                    else:
                        kpts_log += ", Result: None"
                    logger.info(kpts_log)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++
                
                # Handle keypoints based on shape (expecting (17, 3) or (17, 2))
                if keypoints is None:
                    logger.warning(f"Pose estimator returned None for frame {i}. Appending zeros (17, 2).")
                    keypoints_list.append(np.zeros((17, 2)))
                elif keypoints.shape == (17, 3):
                    keypoints_yx_pixels = keypoints[:, :2] # Extract y, x coordinates (pixel values)
                    # +++ Normalize Keypoints +++
                    keypoints_yx_normalized = keypoints_yx_pixels.astype(np.float32) # Ensure float for division
                    keypoints_yx_normalized[:, 0] /= frame_h # Normalize Y
                    keypoints_yx_normalized[:, 1] /= frame_w # Normalize X
                    keypoints_list.append(keypoints_yx_normalized)
                    # +++++++++++++++++++++++++++
                elif keypoints.shape == (17, 2):
                    logger.warning(f"Pose estimator returned keypoints with shape (17, 2) for frame {i}. Assuming pixel coords and normalizing.")
                    keypoints_yx_pixels = keypoints # Assume they are pixel coords
                    # +++ Normalize Keypoints +++
                    keypoints_yx_normalized = keypoints_yx_pixels.astype(np.float32)
                    keypoints_yx_normalized[:, 0] /= frame_h # Normalize Y
                    keypoints_yx_normalized[:, 1] /= frame_w # Normalize X
                    keypoints_list.append(keypoints_yx_normalized)
                    # +++++++++++++++++++++++++++
                else:
                    logger.warning(f"Pose estimator returned unexpected keypoints shape {keypoints.shape} for frame {i}. Appending zeros (17, 2).")
                    keypoints_list.append(np.zeros((17, 2)))
                
            except Exception as e:
                logger.error(f"Error estimating pose for frame {i}: {e}")
                # Append zeros if any error occurs during estimation
                keypoints_list.append(np.zeros((17, 2)))
                
        cap.release()
        
        # Convert to numpy array
        keypoints_array = np.array(keypoints_list)
        logger.info(f"Extracted keypoints shape: {keypoints_array.shape}")
        
        return keypoints_array
    
    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to keypoints."""
        logger.info("Applying temporal smoothing")
        return self.temporal_filter.smooth_keypoints(keypoints)
    
    def _detect_shot_sequences(
        self, 
        keypoints: np.ndarray, 
        video_info: Dict
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Detect shot sequences in the video.
        
        Args:
            keypoints: Smoothed keypoints
            video_info: Video information
            
        Returns:
            Tuple of (shot_sequences, shot_timestamps)
        """
        logger.info("Detecting shot sequences")
        
        # This is a simplified detection that looks for significant motion
        # In a real system, this would be more sophisticated
        
        fps = video_info.get('fps', 30)
        sequence_length = self.config.get('rnn_classifier', {}).get('sequence_length', 60)
        overlap = self.config.get('detection', {}).get('sequence_overlap', 15)
        
        # Calculate velocities of wrist keypoints
        right_wrist_idx = 10  # Right wrist keypoint index
        left_wrist_idx = 9    # Left wrist keypoint index
        
        # Calculate velocities
        right_wrist_vel = np.linalg.norm(
            np.diff(keypoints[:, right_wrist_idx, :], axis=0), 
            axis=1
        )
        left_wrist_vel = np.linalg.norm(
            np.diff(keypoints[:, left_wrist_idx, :], axis=0), 
            axis=1
        )
        
        # Combine velocities (take the maximum at each frame)
        combined_vel = np.maximum(right_wrist_vel, left_wrist_vel)
        
        # Pad to match original length
        combined_vel = np.pad(combined_vel, (0, 1), 'edge')
        
        # Threshold for shot detection
        vel_threshold = np.percentile(combined_vel, 90)  # Use 90th percentile as threshold
        
        # Find frames where velocity exceeds threshold
        high_vel_frames = np.where(combined_vel > vel_threshold)[0]
        
        # Group consecutive frames
        shot_starts = []
        if len(high_vel_frames) > 0:
            shot_groups = np.split(high_vel_frames, np.where(np.diff(high_vel_frames) > 10)[0] + 1)
            
            for group in shot_groups:
                if len(group) >= 5:  # Minimum 5 frames of high velocity to be considered a shot
                    shot_starts.append(group[0])
        
        # Extract shot sequences
        shot_sequences = []
        shot_timestamps = []
        
        for start_frame in shot_starts:
            # Ensure we have enough frames before and after
            start_idx = max(0, start_frame - sequence_length // 3)
            end_idx = min(len(keypoints), start_idx + sequence_length)
            
            # Skip if sequence is too short
            if end_idx - start_idx < sequence_length:
                continue
            
            # Extract sequence
            sequence = keypoints[start_idx:end_idx]
            
            # Skip if sequence already exists with significant overlap
            if any(np.array_equal(seq, sequence) for seq in shot_sequences):
                continue
            
            # Add sequence
            shot_sequences.append(sequence)
            
            # Calculate timestamp
            timestamp = {
                'start_frame': start_idx,
                'end_frame': end_idx,
                'start_time': start_idx / fps,
                'end_time': end_idx / fps
            }
            shot_timestamps.append(timestamp)
        
        logger.info(f"Detected {len(shot_sequences)} shot sequences")
        return shot_sequences, shot_timestamps
    
    def _classify_shots(self, shot_sequences: List[np.ndarray]) -> List[Dict]:
        """
        Classify shots using the RNN classifier.
        
        Args:
            shot_sequences: List of shot sequences
            
        Returns:
            List of classification results
        """
        if not shot_sequences:
            logger.warning("No shot sequences to classify")
            return []
        
        logger.info(f"Classifying {len(shot_sequences)} shots")
        
        # Preprocess sequences for the classifier
        preprocessed_sequences = []
        for seq in shot_sequences:
            # Reshape to classifier input format
            n_frames, n_keypoints, n_coords = seq.shape
            flat_seq = seq.reshape(n_frames, n_keypoints * n_coords)
            preprocessed_sequences.append(flat_seq)
        
        # Convert to numpy array
        X = np.array(preprocessed_sequences)
        
        # Get predictions
        try:
            probs = self.classifier.predict(X)
            class_indices = np.argmax(probs, axis=1)
            
            # Create classification results
            results = []
            for i, class_idx in enumerate(class_indices):
                class_name = "unknown"
                if self.classifier.classes and class_idx < len(self.classifier.classes):
                    class_name = self.classifier.classes[class_idx]
                
                confidence = probs[i, class_idx]
                
                results.append({
                    'class_index': int(class_idx),
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'probabilities': {
                        self.classifier.classes[j] if self.classifier.classes and j < len(self.classifier.classes) else f"class_{j}": float(probs[i, j])
                        for j in range(probs.shape[1])
                    }
                })
            
            logger.info(f"Classification results: {[r['class_name'] for r in results]}")
            return results
        
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return []
    
    def _analyze_kinematics(
        self, 
        keypoints: np.ndarray, 
        shot_timestamps: List[Dict]
    ) -> List[Dict]:
        """
        Analyze kinematics for each shot.
        
        Args:
            keypoints: Smoothed keypoints
            shot_timestamps: Shot timestamp information
            
        Returns:
            List of kinematic analysis results
        """
        if not shot_timestamps:
            logger.warning("No shots to analyze")
            return []
        
        logger.info(f"Analyzing kinematics for {len(shot_timestamps)} shots")
        
        results = []
        for i, timestamp in enumerate(shot_timestamps):
            start_frame = timestamp['start_frame']
            end_frame = timestamp['end_frame']
            
            # Extract keypoints for this shot
            shot_keypoints = keypoints[start_frame:end_frame]
            
            # Analyze kinematics
            analysis = self.kinematic_analyzer.analyze_shot(shot_keypoints)
            
            # Add timestamp information
            analysis['start_frame'] = start_frame
            analysis['end_frame'] = end_frame
            analysis['start_time'] = timestamp['start_time']
            analysis['end_time'] = timestamp['end_time']
            
            results.append(analysis)
        
        return results
    
    def _create_visualizations(
        self, 
        video_path: str, 
        keypoints: np.ndarray, 
        shot_timestamps: List[Dict], 
        shot_classifications: List[Dict],
        output_dir: str
    ) -> Dict[str, str]:
        """
        Create visualizations for the analysis.
        
        Args:
            video_path: Path to the video file
            keypoints: Smoothed keypoints
            shot_timestamps: Shot timestamp information
            shot_classifications: Shot classification results
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of visualization paths
        """
        logger.info("Creating visualizations")
        
        # Create visualization directory
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create pose visualization
        pose_video_path = os.path.join(viz_dir, "pose_overlay.mp4")
        self.visualizer.create_pose_visualization(
            video_path, 
            keypoints, 
            pose_video_path
        )
        
        # Create shot visualization
        shot_video_path = os.path.join(viz_dir, "shot_detection.mp4")
        self.visualizer.create_shot_visualization(
            video_path, 
            keypoints, 
            shot_timestamps, 
            shot_classifications, 
            shot_video_path
        )
        
        # Create trajectory plots
        trajectory_path = os.path.join(viz_dir, "trajectories.png")
        self._create_trajectory_plots(keypoints, shot_timestamps, trajectory_path)
        
        return {
            'pose_video': pose_video_path,
            'shot_video': shot_video_path,
            'trajectory_plot': trajectory_path
        }
    
    def _create_trajectory_plots(
        self, 
        keypoints: np.ndarray, 
        shot_timestamps: List[Dict], 
        output_path: str
    ) -> None:
        """
        Create trajectory plots for key points.
        
        Args:
            keypoints: Smoothed keypoints
            shot_timestamps: Shot timestamp information
            output_path: Path to save the plot
        """
        # Key points to track
        key_indices = {
            'Right Wrist': 10,
            'Left Wrist': 9,
            'Right Elbow': 8,
            'Left Elbow': 7,
            'Right Shoulder': 6,
            'Left Shoulder': 5
        }
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot trajectories for each key point
        for i, (name, idx) in enumerate(key_indices.items()):
            ax = axes[i]
            
            # Plot trajectory for all frames
            ax.plot(keypoints[:, idx, 0], keypoints[:, idx, 1], 'b-', alpha=0.3, label='Full Trajectory')
            
            # Highlight shot trajectories
            for j, timestamp in enumerate(shot_timestamps):
                start_frame = timestamp['start_frame']
                end_frame = timestamp['end_frame']
                
                ax.plot(
                    keypoints[start_frame:end_frame, idx, 0],
                    keypoints[start_frame:end_frame, idx, 1],
                    'r-', linewidth=2,
                    label=f'Shot {j+1}' if j == 0 else None
                )
            
            ax.set_title(f'{name} Trajectory')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True)
            ax.set_aspect('equal')
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _generate_report(
        self, 
        video_info: Dict, 
        shot_classifications: List[Dict], 
        kinematic_data: List[Dict], 
        visualization_paths: Dict[str, str],
        output_dir: str
    ) -> str:
        """
        Generate an analysis report.
        
        Args:
            video_info: Video information
            shot_classifications: Shot classification results
            kinematic_data: Kinematic analysis results
            visualization_paths: Paths to visualizations
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating report")
        
        # Combine results
        report_data = {
            'video_info': video_info,
            'shot_classifications': shot_classifications,
            'kinematic_data': kinematic_data,
            'visualization_paths': visualization_paths
        }
        
        # Generate report
        report_path = self.report_generator.generate_report(report_data, output_dir)
        
        return report_path

def get_color_bgr(color_name):
    """Convert color name to BGR tuple."""
    colors = {
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'orange': (0, 165, 255),
        'gray': (128, 128, 128),
        'white': (255, 255, 255)
    }
    return colors.get(color_name.lower(), (255, 255, 255))

def load_static_poses(static_poses_dir):
    """Load static pose embeddings from JSON files."""
    static_poses = {}
    target_dir = static_poses_dir
    if not os.path.isdir(target_dir):
        logger.warning(f"Static poses directory not found: {target_dir}")
        return static_poses

    logger.info(f"Loading static poses from: {target_dir}")
    file_count = 0
    loaded_count = 0
    try:
        dir_contents = os.listdir(target_dir)
    except Exception as list_err:
        dir_contents = [] # Fallback to empty list
        
    for filename in dir_contents:
        if filename.endswith('.json'):
            file_count += 1
            pose_name = Path(filename).stem
            filepath = os.path.join(target_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    embedding = None
                    # Try common keys for embeddings
                    if 'embedding' in data:
                        embedding = data['embedding']
                    elif 'pose_embedding' in data:
                        embedding = data['pose_embedding']
                    elif isinstance(data, list): # Check if the entire file is the list
                        embedding = data
                    
                    if embedding is not None:
                        static_poses[pose_name] = np.array(embedding)
                        loaded_count += 1
                    else:
                        logger.warning(f"Could not find embedding data in {filename}")
            except Exception as e:
                logger.error(f"Error loading static pose {filename}: {e}")
                
    logger.info(f"Found {file_count} JSON files, loaded {loaded_count} static poses from {target_dir}")
    # Log the names of loaded poses for verification
    if loaded_count > 0:
        logger.info(f"Loaded pose names: {list(static_poses.keys())}")
    else:
        logger.warning("No static poses were successfully loaded.")
        
    return static_poses

def calculate_similarity_features(sequence, static_poses):
    """Calculate similarity features between sequence frames and static poses."""
    # --- Ensure all prints are removed ---
    
    num_frames = sequence.shape[0]
    num_static_poses = len(static_poses)
    similarity_features = np.zeros((num_frames, num_static_poses))

    if not static_poses:
        return similarity_features # Return zeros of shape (num_frames, 0)

    # Ensure static_poses values are numpy arrays before stacking
    static_pose_embeddings_list = []
    static_pose_names = []
    for name, embedding in static_poses.items():
        if isinstance(embedding, np.ndarray):
            static_pose_embeddings_list.append(embedding)
            static_pose_names.append(name)
        else:
            logger.warning(f"Skipping static pose '{name}' due to unexpected type: {type(embedding)}")
            
    if not static_pose_embeddings_list:
        logger.warning("No valid static pose embeddings found to calculate similarity.")
        return np.zeros((num_frames, 0))

    static_pose_embeddings = np.array(static_pose_embeddings_list)
    num_static_poses = len(static_pose_embeddings)
    similarity_features = np.zeros((num_frames, num_static_poses))

    if static_pose_embeddings.size == 0:
         logger.warning("Static pose embeddings array is empty.")
         return similarity_features

    try:
        static_pose_embeddings_norm = np.linalg.norm(static_pose_embeddings, axis=1, keepdims=True)
        static_pose_embeddings_normalized = static_pose_embeddings / (static_pose_embeddings_norm + 1e-6)
    except Exception as e:
        logger.error(f"Error normalizing static poses: {e}. Shape: {static_pose_embeddings.shape}")
        return similarity_features

    for i in range(num_frames):
        if sequence[i].size == 0:
            logger.warning(f"Empty keypoints found at frame {i}. Skipping similarity calculation.")
            continue
            
        frame_features = sequence[i].flatten()
        frame_features_norm = np.linalg.norm(frame_features)
        if frame_features_norm < 1e-6:
             frame_features_normalized = frame_features
        else:
             frame_features_normalized = frame_features / frame_features_norm

        if frame_features_normalized.shape[0] != static_pose_embeddings_normalized.shape[1]:
             if i == 0: 
                  logger.warning(f"Dimension mismatch for similarity calc: Frame features length {frame_features_normalized.shape[0]} vs Static poses embedding dim {static_pose_embeddings_normalized.shape[1]}. Similarity calculation will be skipped for all frames.")
             continue

        try:
             similarities = np.dot(frame_features_normalized.reshape(1, -1),
                                   static_pose_embeddings_normalized.T)
             similarity_features[i, :] = similarities[0]
        except ValueError as e:
             logger.error(f"Error during dot product at frame {i}: {e}. Shapes: Frame {frame_features_normalized.reshape(1, -1).shape}, Static {static_pose_embeddings_normalized.T.shape}")

    return similarity_features

# +++ NEW HELPER FUNCTION MOVED HERE +++
def _prepare_sequence_and_similarity(
    all_keypoints: np.ndarray,
    i: int,
    window_size: int,
    static_poses: dict,
    num_classes: int,
    similarity_input_shape: tuple | None,
    is_multi_input: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Prepare sequence keypoints slice and calculate similarity features for one step.

    Args:
        all_keypoints: Full array of keypoints (frames, keypoints, 2). MUST be float32.
        i: Current starting index for the window.
        window_size: The desired sequence length.
        static_poses: Dictionary of loaded static poses (values are float32 ndarrays).
        num_classes: Expected number of classes (for similarity feature shape).
        similarity_input_shape: Expected shape tuple for the model's similarity input.
        is_multi_input: Boolean indicating if the model uses multiple inputs.

    Returns:
        Tuple containing:
        - sequence_keypoints (np.ndarray | None): The (window_size, 17, 2) slice (float32),
                                                  or None if invalid length.
        - similarity_features (np.ndarray | None): Calculated similarity features (window_size, num_classes) (float32)
                                                  or zeros array if calculation fails/not applicable,
                                                  or None if sequence is invalid or not multi-input.
    """
    # --- 1. Prepare Sequence Keypoints ---
    # Ensure slice indexing is valid
    if i < 0 or i + window_size > all_keypoints.shape[0]:
         logger.warning(f"Invalid slice index i={i} for window_size={window_size} and total frames={all_keypoints.shape[0]}")
         return None, None

    # Slicing maintains dtype, ensure source is float32
    sequence_keypoints = all_keypoints[i : i + window_size].copy() # Copy ensures modification doesn't affect original

    if sequence_keypoints.shape[0] != window_size:
        # This case should be rare due to loop range, but good safeguard
        logger.warning(f"Frame {i}: Incorrect sequence length {sequence_keypoints.shape[0]} after slice (expected {window_size}) - This shouldn't happen.")
        return None, None # Indicate sequence preparation failure

    # --- 2. Prepare Similarity Features (if needed) ---
    similarity_features_final = None # Initialize for single-input or failure cases
    if is_multi_input:
        # Determine expected shape based on model's input layer
        # Example shape: (None, 30, 5) -> batch, seq_len, num_features
        expected_seq_len = similarity_input_shape[1] if similarity_input_shape and len(similarity_input_shape) == 3 else window_size
        expected_num_features = similarity_input_shape[2] if similarity_input_shape and len(similarity_input_shape) == 3 else num_classes # Use num_classes as fallback

        # Validate expected_num_features against num_classes
        if expected_num_features != num_classes:
             logger.warning(f"Frame {i}: Model expects {expected_num_features} similarity features, but config has {num_classes} classes. Using {expected_num_features} based on model shape.")
             # If we proceed, the calculated features might mismatch model input if static_poses keys don't align

        # Default to zeros if calculation fails or static poses are missing
        similarity_features_final = np.zeros((expected_seq_len, expected_num_features), dtype=np.float32)

        if not static_poses:
             logger.debug(f"Frame {i}: No static poses loaded, using zero similarity features ({expected_seq_len}x{expected_num_features}).")
             # Already defaulted to zeros above
        else:
            # Check if number of loaded poses matches expected features
            if len(static_poses) != expected_num_features:
                 logger.warning(f"Frame {i}: Number of loaded static poses ({len(static_poses)}) doesn't match expected similarity features ({expected_num_features}). Using zero features.")
                 # Keep default zeros
            else:
                try:
                    # Calculate raw similarities
                    calculated_features = calculate_similarity_features(sequence_keypoints, static_poses) # Returns shape (seq_len, num_loaded_poses)

                    # Validate calculated features shape against expected model input
                    if calculated_features.shape[0] != expected_seq_len:
                        logger.warning(f"Frame {i}: Calculated similarity seq length ({calculated_features.shape[0]}) != expected ({expected_seq_len}). Using zeros.")
                        # Keep default zeros
                    elif calculated_features.shape[1] != expected_num_features:
                         # This should only happen if static_poses keys didn't match expected_num_features check above
                         logger.error(f"Frame {i}: Calculated similarity features ({calculated_features.shape[1]}) != expected ({expected_num_features}) despite matching pose count. Logic error? Using zeros.")
                         # Keep default zeros
                    else:
                        similarity_features_final = calculated_features # Use calculated features
                        logger.debug(f"Frame {i}: Successfully calculated similarity features with shape {similarity_features_final.shape}")

                except Exception as e:
                    logger.error(f"Frame {i}: Error calculating similarity features: {e}. Using zeros.", exc_info=True)
                    # Keep default zeros

    # Return sequence (float32) and similarity features (float32 or None)
    return sequence_keypoints.astype(np.float32), similarity_features_final
# +++++++++++++++++++++++++++++++++++++++++++

def analyze_video(video_path, model_path, static_poses_dir_arg, output_dir):
    """Analyze video using the specified model and save annotated video."""
    logger.info(f"Starting video analysis for: {video_path}")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Attempting to load static poses from: {static_poses_dir_arg}")

    # Create output directory
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_output_dir = os.path.join(output_dir, f"{video_name}_analysis_{timestamp}")
    os.makedirs(analysis_output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {analysis_output_dir}")

    # +++ Configure File Logging +++
    log_file_path = os.path.join(analysis_output_dir, f"analysis_{timestamp}.log")
    # Ensure logging isn't duplicated if called multiple times
    root_logger = logging.getLogger()
    # Remove existing file handlers if any to avoid duplicate logs in testing
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and hasattr(handler, 'baseFilename') and handler.baseFilename == log_file_path:
            logger.debug(f"Removing existing file handler for {log_file_path}")
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO) # Log INFO level and above to file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file_path}")
    # ++++++++++++++++++++++++++++++

    # === 1. Initialize Pose Estimator ===
    try:
        pose_estimator = MoveNetEstimator(model_type='lightning')
        logger.info("Pose estimator initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize pose estimator: {e}", exc_info=True)
        # Clean up file handler before returning on error
        root_logger.removeHandler(file_handler)
        file_handler.close()
        return
    # ===================================

    # === 2. Load Classification Model ===
    classifier = None # Initialize
    try:
        # Minimal config needed here, actual class names loaded from metadata if possible
        config = {'shot_types': []} # Start empty, will be populated
        classifier = RNNShotClassifier(config=config)
        metadata_path = Path(model_path).with_suffix('.json')
        if os.path.exists(metadata_path):
            logger.info(f"Loading model from {model_path} with metadata {metadata_path}")
            # Load updates the internal config, including shot_types
            classifier.load(model_path, metadata_path=str(metadata_path))
            logger.info(f"Model loaded. Classifier config updated: {classifier.config}")
        else:
            logger.warning(f"Metadata file not found at {metadata_path}, loading model without it.")
            classifier.load(model_path) # May not have correct class names if metadata missing
            # Manually set default if no metadata and config is empty
            if not classifier.config.get('shot_types'):
                 classifier.config['shot_types'] = ['forehand', 'backhand', 'serve', 'volley', 'neutral']
                 logger.warning(f"Using default shot types: {classifier.config['shot_types']}")

        logger.info(f"Model {Path(model_path).name} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}", exc_info=True)
        # Clean up file handler before returning on error
        root_logger.removeHandler(file_handler)
        file_handler.close()
        return
    # Ensure classifier is loaded
    if not classifier or not classifier.model:
         logger.error("Classifier or its model failed to load properly.")
         root_logger.removeHandler(file_handler)
         file_handler.close()
         return
    # ===================================

    # === 3. Determine Model Input Shapes and Expected Classes ===
    sequence_input_shape = None
    similarity_input_shape = None
    is_multi_input = False # Default
    num_classes = 0 # Default

    try:
        is_multi_input = hasattr(classifier.model, 'inputs') and len(classifier.model.inputs) > 1
        # Use num_classes FROM THE CLASSIFIER'S (potentially updated) CONFIG
        num_classes = len(classifier.config.get('shot_types', []))
        if num_classes == 0:
             logger.error("Classifier config has zero shot types after loading. Cannot proceed.")
             raise ValueError("Zero shot types found in classifier config.")
        logger.info(f"Number of classes based on classifier config: {num_classes}")

        if is_multi_input:
            # Get input shapes directly from the loaded model
            input_shapes = [tuple(inp.shape) for inp in classifier.model.inputs] # Use tuple for consistency
            input_names = [inp.name.split(':')[0] for inp in classifier.model.inputs] # Get names without :0 suffix
            logger.info(f"Model Inputs Found: Names={input_names}, Shapes={input_shapes}")

            if len(input_shapes) == 2:
                # Attempt to identify sequence and similarity inputs by name pattern or order
                # Assume sequence input doesn't contain 'sim' or 'static'
                seq_input_index = next((idx for idx, name in enumerate(input_names) if not ('sim' in name or 'static' in name)), 0)
                sim_input_index = 1 - seq_input_index # The other one

                sequence_input_shape = input_shapes[seq_input_index]
                similarity_input_shape = input_shapes[sim_input_index]
                logger.info(f"Detected multi-input shapes: Sequence {sequence_input_shape} (name: {input_names[seq_input_index]}), Similarity {similarity_input_shape} (name: {input_names[sim_input_index]})")

                # Validate similarity shape num features against num_classes
                if similarity_input_shape and len(similarity_input_shape) == 3:
                    model_num_features = similarity_input_shape[2]
                    if model_num_features != num_classes:
                         logger.warning(f"Model similarity input expects {model_num_features} features, but config/metadata has {num_classes} classes. This might cause errors during prediction if static poses don't align with model expectation.")
                         # We will prioritize num_classes from config for loading static poses,
                         # but the helper function will use model_num_features for final shape checks.
                    else:
                         logger.info(f"Model similarity features ({model_num_features}) matches config classes ({num_classes}).")
                else:
                    logger.warning(f"Could not validate num_classes against similarity shape {similarity_input_shape}")

            else:
                logger.warning(f"Model has {len(input_shapes)} inputs, expected 1 or 2. Treating as single input.")
                is_multi_input = False # Treat as single input if structure is unexpected
        # If not multi_input initially, or if multi-input detection failed/was ambiguous
        if not is_multi_input:
            sequence_input_shape = tuple(classifier.model.input_shape)
            logger.info(f"Model treated as single-input. Shape: {sequence_input_shape}")

    except Exception as e:
         logger.error(f"Error determining model shapes: {e}. Cannot proceed.", exc_info=True)
         root_logger.removeHandler(file_handler)
         file_handler.close()
         return
    # =========================================================

    # === 4. Load Static Poses ===
    static_poses = {}
    if is_multi_input: # Only load if model expects similarity features
        static_poses = load_static_poses(static_poses_dir_arg) # Returns dict with float32 arrays
        if not static_poses:
             logger.warning("Multi-input model but could not load any valid static poses. Similarity features will be zeros.")
        else:
             # Validate against the num_classes from config
             loaded_pose_names = sorted(list(static_poses.keys()))
             config_class_names = sorted(classifier.config.get('shot_types', []))
             if len(loaded_pose_names) != num_classes:
                  logger.warning(f"Loaded {len(loaded_pose_names)} static poses ({loaded_pose_names}), but expected {num_classes} based on config/metadata ({config_class_names}). Check pose names match config shot_types.")
             else:
                  # Check if names match config/metadata names
                  if loaded_pose_names != config_class_names:
                      logger.warning(f"Loaded static pose names {loaded_pose_names} do not match config class names {config_class_names}. Ensure JSON filenames match shot_types in config/metadata for correct similarity calculation.")
                  else:
                      logger.info(f"Loaded {len(static_poses)} static poses, matching expected {num_classes} classes.")
    else:
        logger.info("Model is single-input, skipping static pose loading.")
    # ===========================

    # === 5. Load Video and Extract Keypoints ===
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        root_logger.removeHandler(file_handler)
        file_handler.close()
        return

    frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
         logger.warning(f"Video FPS reported as {fps}. Using default 30 FPS for writer.")
         fps = 30.0 # Default FPS if invalid

    frames = []
    all_keypoints_list = [] # Store keypoints per frame
    logger.info(f"Loading {frame_count_prop} frames (estimated) and extracting keypoints...")

    frame_idx = 0
    pbar_poses = tqdm(total=frame_count_prop if frame_count_prop > 0 else None, desc="Loading video & extracting poses")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Finished reading frames after {frame_idx} frames.")
            break
        frames.append(frame) # Store original frame

        # +++ Get frame dimensions for normalization +++
        frame_h, frame_w = frame.shape[:2]
        # +++++++++++++++++++++++++++++++++++++++++++++

        # Convert frame to RGB for pose estimator
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Estimate pose - This is the correct location for pose estimation and handling
        keypoints_normalized = np.zeros((17, 2), dtype=np.float32) # Default to zeros, ensure float32
        try:
            # Assume estimate_pose returns (17, 3) [y, x, conf] in PIXEL COORDS or None
            keypoints_raw_pixels_conf = pose_estimator.estimate_pose(frame_rgb)

            if keypoints_raw_pixels_conf is None:
                logger.debug(f"Pose estimator returned None for frame {frame_idx}. Using zeros.")
                # keypoints_normalized remains zeros
            elif keypoints_raw_pixels_conf.shape == (17, 3):
                keypoints_yx_pixels = keypoints_raw_pixels_conf[:, :2] # Extract y, x coordinates (pixel values)
                # +++ Normalize Keypoints +++
                # Avoid division by zero if frame dimensions are invalid
                if frame_h > 0 and frame_w > 0:
                    # Perform normalization directly into the pre-allocated array
                    keypoints_normalized[:, 0] = keypoints_yx_pixels[:, 0].astype(np.float32) / frame_h # Normalize Y
                    keypoints_normalized[:, 1] = keypoints_yx_pixels[:, 1].astype(np.float32) / frame_w # Normalize X
                    # Clip values to [0, 1] after normalization
                    np.clip(keypoints_normalized, 0.0, 1.0, out=keypoints_normalized)
                else:
                     logger.warning(f"Frame {frame_idx} has invalid dimensions ({frame_h}x{frame_w}). Using zero keypoints.")
                     # keypoints_normalized remains zeros
                # +++++++++++++++++++++++++++
            # +++ ADDED HANDLING FOR (17, 2) SHAPE +++
            elif keypoints_raw_pixels_conf.shape == (17, 2):
                 logger.debug(f"Pose estimator returned shape (17, 2) for frame {frame_idx}. Assuming [y, x] pixel coords.")
                 keypoints_yx_pixels = keypoints_raw_pixels_conf # Shape (17, 2)
                 # Normalize Keypoints
                 if frame_h > 0 and frame_w > 0:
                     keypoints_normalized[:, 0] = keypoints_yx_pixels[:, 0].astype(np.float32) / frame_h # Normalize Y
                     keypoints_normalized[:, 1] = keypoints_yx_pixels[:, 1].astype(np.float32) / frame_w # Normalize X
                     np.clip(keypoints_normalized, 0.0, 1.0, out=keypoints_normalized)
                 else:
                     logger.warning(f"Frame {frame_idx} has invalid dimensions ({frame_h}x{frame_w}). Using zero keypoints for (17,2) input.")
                     # keypoints_normalized remains zeros
            # ++++++++++++++++++++++++++++++++++++++++
            else:
                logger.warning(f"Pose estimator returned unexpected keypoints shape {keypoints_raw_pixels_conf.shape} for frame {frame_idx}. Expected (17, 3). Using zeros.")
                # keypoints_normalized remains zeros

        except Exception as e:
            logger.error(f"Error estimating pose for frame {frame_idx}: {e}", exc_info=True)
            # keypoints_normalized remains zeros

        all_keypoints_list.append(keypoints_normalized)
        frame_idx += 1
        pbar_poses.update(1)
    pbar_poses.close()
    cap.release()

    if not frames:
        logger.error("No frames could be read from the video.")
        root_logger.removeHandler(file_handler)
        file_handler.close()
        return

    # Convert keypoints list to numpy array
    all_keypoints = np.array(all_keypoints_list, dtype=np.float32) # Ensure float32 type
    actual_frame_count = len(frames) # Use actual number of frames read
    if actual_frame_count != all_keypoints.shape[0]:
         logger.error(f"Mismatch between number of frames read ({actual_frame_count}) and keypoints extracted ({all_keypoints.shape[0]})")
         # Handle mismatch: Use the minimum length
         min_len = min(actual_frame_count, all_keypoints.shape[0])
         frames = frames[:min_len]
         all_keypoints = all_keypoints[:min_len]
         actual_frame_count = min_len
         logger.warning(f"Adjusted frames and keypoints to minimum length: {actual_frame_count}")


    # +++ LOG ALL KEYPOINTS ARRAY +++
    log_msg_ak = f"[ALL KEYPOINTS CREATED] Shape: {all_keypoints.shape}, Dtype: {all_keypoints.dtype}"
    if all_keypoints.size > 0:
        # Use nanmean etc. in case normalization produced NaNs (e.g., div by zero, although unlikely with clipping)
        mean_val = np.nanmean(all_keypoints)
        min_val = np.nanmin(all_keypoints)
        max_val = np.nanmax(all_keypoints)
        log_msg_ak += f", Mean: {mean_val:.4f}, Min: {min_val:.4f}, Max: {max_val:.4f}"
    else:
        log_msg_ak += ", Empty Array"
    logger.info(log_msg_ak)
    # +++++++++++++++++++++++++++++++
    # ==========================================

    # === 6. Prepare Video Writer ===
    if not frames: # Double check after potential truncation
         logger.error("No frames available to write video.")
         root_logger.removeHandler(file_handler)
         file_handler.close()
         return

    h, w = frames[0].shape[:2]
    output_h, output_w = h, w # Dimensions for the output frame
    rotated = False
    if h > w:
        logger.info("Detected portrait video, rotating frames for visualization.")
        rotated = True
        output_h, output_w = w, h # Swap dimensions for output video

    output_video_filename = f"analyzed_{Path(video_path).stem}.mp4"
    output_video_path = os.path.join(analysis_output_dir, output_video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (output_w, output_h))
    if not video_writer.isOpened():
         logger.error(f"Failed to open video writer for path: {output_video_path}")
         root_logger.removeHandler(file_handler)
         file_handler.close()
         return
    # =============================

    # === 7. Process Video Sequences and Annotate ===
    logger.info("Analyzing sequences and generating output video...")
    results = {} # Stores prediction for the *last* frame of the sequence window

    # --- Main Analysis Loop ---\
    num_sequences = actual_frame_count - WINDOW_SIZE + 1
    if num_sequences < 1:
         logger.warning(f"Video too short ({actual_frame_count} frames) for window size ({WINDOW_SIZE}). No analysis performed.")
    else:
        logger.info(f"Processing {num_sequences} sequences...")

    for i in tqdm(range(num_sequences), desc="Analyzing video"):
        # --- Prepare Data using Helper ---
        # Pass necessary info determined *before* the loop
        sequence_keypoints, similarity_features = _prepare_sequence_and_similarity(
            all_keypoints, # Source keypoints (float32)
            i,             # Current index
            WINDOW_SIZE,   # Sequence length
            static_poses,  # Loaded static poses (values are float32)
            num_classes,   # Determined number of classes
            similarity_input_shape, # Expected model sim input shape
            is_multi_input # Flag if model is multi-input
        )

        # Skip if sequence preparation failed (e.g., invalid slice length)
        if sequence_keypoints is None:
            logger.warning(f"Skipping iteration {i} due to sequence preparation failure.")
            continue
        # -------------------------------\

        # --- LOGGING (Kept for debugging, level changed) ---
        # Log stats of the prepared sequence
        log_msg_sk = f"[SEQ KEYPOINTS Frame {i}] Shape: {sequence_keypoints.shape}, Dtype: {sequence_keypoints.dtype}"
        if sequence_keypoints.size > 0:
            seq_mean = np.nanmean(sequence_keypoints)
            seq_min = np.nanmin(sequence_keypoints)
            seq_max = np.nanmax(sequence_keypoints)
            log_msg_sk += f", Mean: {seq_mean:.4f}, Min: {seq_min:.4f}, Max: {seq_max:.4f}"
            # Check for NaNs again just in case, although should be handled by normalization/clipping
            has_nan = np.isnan(sequence_keypoints).any()
            if has_nan:
                 logger.warning(f"[SEQ KEYPOINTS Frame {i}] NaNs detected in sequence_keypoints AFTER preparation!")
        else:
            log_msg_sk += ", Empty"
        logger.debug(log_msg_sk) # Changed to DEBUG level
        # --- END LOGGING ---

        # --- Preprocess Sequence for Classifier ---
        # classifier.preprocess_input should handle shape adjustment (e.g., flattening, selecting keypoints)
        # It expects a batch, so add batch dimension [1, seq_len, kpts, 2]
        processed_sequence_batch = None
        try:
             processed_sequence_batch = classifier.preprocess_input(np.expand_dims(sequence_keypoints, axis=0))
        except Exception as preproc_error:
             logger.error(f"Error during classifier preprocessing at frame {i}: {preproc_error}", exc_info=True)
             continue # Skip prediction if preprocessing fails

        if processed_sequence_batch is None or processed_sequence_batch.size == 0:
             logger.warning(f"Skipping frame {i}: Classifier preprocessing returned empty or None array.")
             continue
        # Remove batch dimension for predict_sequence if needed (check classifier method)
        # Assuming predict_sequence takes non-batched sequence input
        processed_sequence_for_predict = processed_sequence_batch[0]
        # ---------------------------------------

        # --- LOG INPUTS TO PREDICT ---
        log_msg = f"[PREDICT INPUTS Frame {i}] Seq Shape: {processed_sequence_for_predict.shape}"
        if processed_sequence_for_predict.size > 0:
            log_msg += f", Seq Mean: {np.nanmean(processed_sequence_for_predict):.4f}, Min: {np.nanmin(processed_sequence_for_predict):.4f}, Max: {np.nanmax(processed_sequence_for_predict):.4f}"
        if similarity_features is not None:
            log_msg += f", Sim Shape: {similarity_features.shape}"
            if similarity_features.size > 0:
                log_msg += f", Sim Mean: {np.nanmean(similarity_features):.4f}, Min: {np.nanmin(similarity_features):.4f}, Max: {np.nanmax(similarity_features):.4f}"
            else:
                log_msg += ", Sim: Empty"
        else:
             # Only log if multi-input was expected but features are None (shouldn't happen with helper defaults)
             if is_multi_input:
                  log_msg += ", Sim: None (multi-input expected)"
             else:
                  log_msg += ", Sim: N/A (single-input)" # More accurate
        logger.info(log_msg) # Keep INFO level for prediction inputs
        # +++++++++++++++++++++++++++++

        # --- Predict ---
        prediction = "error" # Default
        confidence = 0.0
        try:
            # Pass the processed sequence and prepared similarity features
            # Ensure predict_sequence handles similarity_features being None for single-input models
            prediction, confidence = classifier.predict_sequence(
                processed_sequence_for_predict, # Preprocessed sequence (no batch dim)
                similarity_features=similarity_features # Prepared similarity features (no batch dim, or None)
            )
            logger.debug(f"Frame {i}: Prediction={prediction}, Confidence={confidence:.3f}")

        except Exception as e:
            logger.error(f"Error during prediction at frame {i}: {e}", exc_info=True)
            # prediction, confidence remain default "error", 0.0
        # --------------

        # --- Store result & Annotate Frame ---
        current_frame_index = i + WINDOW_SIZE - 1 # Index of the *last* frame in the window
        results[current_frame_index] = {'prediction': prediction, 'confidence': confidence}

        # Get the frame corresponding to the *end* of the sequence window for annotation
        if current_frame_index < len(frames):
             frame_to_annotate = frames[current_frame_index]
             output_frame = frame_to_annotate.copy()

             if rotated:
                 output_frame = cv2.rotate(output_frame, cv2.ROTATE_90_CLOCKWISE)

             # Resize if needed (e.g., after rotation) to match writer dimensions
             if output_frame.shape[0] != output_h or output_frame.shape[1] != output_w:
                  try:
                       output_frame = cv2.resize(output_frame, (output_w, output_h))
                  except Exception as resize_error:
                       logger.error(f"Failed to resize frame {current_frame_index} for writing: {resize_error}")
                       continue # Skip writing this frame

             # Draw prediction text overlay
             text = f"{prediction} ({confidence:.2f})"
             font_scale = 1.0
             thickness = 2
             font = cv2.FONT_HERSHEY_SIMPLEX
             text_position = (20, 50)
             (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
             rect_start = (text_position[0] - 10, text_position[1] - text_height - 10)
             rect_end = (text_position[0] + text_width + 10, text_position[1] + baseline + 10)

             try:
                 # Use try-except for drawing operations as they can sometimes fail
                 overlay = output_frame.copy()
                 cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1) # Black background box
                 alpha = 0.6 # Semi-transparent background
                 output_frame = cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0)
                 cv2.putText(
                     output_frame, text, text_position,
                     font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA # White text
                 )

                 # Draw colored border
                 shot_color = SHOT_COLORS.get(prediction, 'white') # Default to white if prediction unknown
                 border_color = get_color_bgr(shot_color)
                 border_size = 10
                 final_frame_with_border = cv2.copyMakeBorder(
                     output_frame, border_size, border_size, border_size, border_size,
                     cv2.BORDER_CONSTANT, value=border_color
                 )

                 # Resize back to target output dimensions AFTER adding border
                 final_frame_resized = cv2.resize(final_frame_with_border, (output_w, output_h))

                 video_writer.write(final_frame_resized)

             except Exception as draw_error:
                 logger.error(f"Error drawing annotations or writing frame {current_frame_index}: {draw_error}", exc_info=True)
                 # Optionally write the unannotated frame or skip
                 # video_writer.write(output_frame) # Write unannotated if drawing fails
        else:
            logger.error(f"Attempted to access frame index {current_frame_index} which is out of bounds for frames list (length {len(frames)}). Skipping annotation.")


    video_writer.release()
    logger.info(f"Analysis complete. Annotated video saved to: {output_video_path}")
    # Clean up file handler after analysis is complete
    logger.debug(f"Removing file handler for {log_file_path} after completion.")
    root_logger.removeHandler(file_handler)
    file_handler.close()
    # =============================================

def main():
    parser = argparse.ArgumentParser(description="Analyze a tennis video using a specified classification model.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help=f"Path to the classification model (.h5 file). Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--static-poses-dir", default=DEFAULT_STATIC_POSES_DIR, help=f"Directory containing static pose JSON files. Default: {DEFAULT_STATIC_POSES_DIR}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory to save analysis results. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level for console output.")

    args = parser.parse_args()

    # Configure basic console logging based on command-line arg
    log_level_console = getattr(logging, args.log_level.upper(), logging.INFO)
    # Set root logger level FIRST to ensure handlers respect it
    logging.getLogger().setLevel(min(log_level_console, logging.INFO)) # Ensure root logger allows at least INFO for file handler

    # Remove default handlers to avoid duplicate messages if basicConfig was called before
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Add console handler with specified level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_console)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Simpler format for console
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # The file logger (always INFO) will be added inside analyze_video

    analyze_video(args.video_path, args.model, args.static_poses_dir, args.output_dir)

if __name__ == "__main__":
    # Configuration now happens in main() based on args
    main() 