"""
This module coordinates the tennis analysis pipeline by orchestrating the execution 
of pose estimation, smoothing, classification, kinematic analysis, and reporting.
"""

import os
import logging
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from ..pose_estimation.pose_estimator import MoveNetEstimator
from ..smoothing.temporal_filter import TemporalFilter
from ..classification.rnn_classifier import RNNClassifier
from ..kinematics.metrics import KinematicMetrics
from ..reporting.report_generator import ReportGenerator
from ..visualization.video_visualizer import VideoVisualizer

# Configure logger
logger = logging.getLogger(__name__)

class PipelineCoordinator:
    """
    Coordinates the execution of the tennis swing analysis pipeline.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline coordinator.
        
        Args:
            config_path: Path to the pipeline configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        logger.info(f"Initialized pipeline with config from {config_path}")
        
        # Initialize pipeline components
        self._init_components()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the pipeline configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _init_components(self) -> None:
        """Initialize all pipeline components based on configuration."""
        try:
            # Get component configs
            pose_config = self.config.get('pose_estimation', {})
            smooth_config = self.config.get('smoothing', {})
            classifier_config = self.config.get('classification', {})
            kinematics_config = self.config.get('kinematics', {})
            
            # Initialize components
            self.pose_estimator = MoveNetEstimator(model_type=pose_config.get('model_type', 'lightning'))
            self.temporal_filter = TemporalFilter(config=smooth_config)
            
            self.classifier = RNNClassifier(config=classifier_config)
            model_path = classifier_config.get('model_path')
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading classifier model from {model_path}")
                self.classifier.load_model(model_path)
            else:
                logger.warning(f"Classifier model not found at {model_path}, model will need to be trained")
            
            self.kinematic_metrics = KinematicMetrics(
                handedness=kinematics_config.get('handedness', 'right')
            )
            
            self.report_generator = ReportGenerator()
            self.video_visualizer = VideoVisualizer()
            
            logger.info("All pipeline components initialized")
        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
            raise
    
    def process_video(self, video_path: str, output_dir: str = None, generate_visualization: bool = True) -> Dict[str, Any]:
        """
        Process a tennis video through the entire analysis pipeline.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save outputs (if None, will use timestamp directory)
            generate_visualization: Whether to generate visualization video
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        logger.info(f"Starting analysis of video: {video_path}")
        
        # Create output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                self.config.get('output_base_dir', 'output'),
                f"analysis_{timestamp}"
            )
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output will be saved to {output_dir}")
        
        try:
            # Step 1: Extract pose keypoints
            keypoints, frames, fps = self._extract_pose_keypoints(video_path)
            logger.info(f"Extracted {len(keypoints)} frames of pose keypoints at {fps} fps")
            
            # Step 2: Apply temporal smoothing
            smooth_keypoints = self._apply_smoothing(keypoints)
            logger.info(f"Applied temporal smoothing to keypoints")
            
            # Step 3: Classify tennis shots
            shot_frames, shot_classes, shot_probs = self._classify_shots(smooth_keypoints)
            logger.info(f"Detected {len(shot_frames)} tennis shots")
            
            # Step 4: Calculate kinematic metrics for each shot
            kinematic_analyses = []
            for i, frame_idx in enumerate(shot_frames):
                # Extract keypoint sequence around the shot
                seq_start = max(0, frame_idx - self.config.get('sequence_radius', 15))
                seq_end = min(len(smooth_keypoints), frame_idx + self.config.get('sequence_radius', 15))
                shot_sequence = smooth_keypoints[seq_start:seq_end]
                
                # Calculate metrics for this shot
                shot_metrics = self.kinematic_metrics.calculate_metrics(shot_sequence, fps)
                shot_report = self.kinematic_metrics.generate_metrics_report(shot_metrics)
                
                # Add to analyses
                kinematic_analyses.append({
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / fps,
                    'shot_type': shot_classes[i],
                    'shot_confidence': float(shot_probs[i]),
                    'metrics': shot_metrics,
                    'report': shot_report
                })
            
            logger.info(f"Completed kinematic analysis for {len(kinematic_analyses)} shots")
            
            # Step 5: Generate report
            report_data = {
                'video_path': video_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'fps': fps,
                'total_frames': len(keypoints),
                'duration_seconds': len(keypoints) / fps,
                'shots': kinematic_analyses
            }
            
            report_file = os.path.join(output_dir, 'analysis_report.html')
            self.report_generator.generate_report(report_data, output_dir)
            logger.info(f"Generated analysis report at {report_file}")
            
            # Step 6: Generate visualization video if requested
            if generate_visualization:
                vis_output_path = os.path.join(output_dir, 'visualization.mp4')
                shot_timestamps = [shot['timestamp'] for shot in kinematic_analyses]
                shot_classifications = [shot['shot_type'] for shot in kinematic_analyses]
                
                self.video_visualizer.create_shot_visualization(
                    video_path=video_path,
                    keypoints=smooth_keypoints,
                    shot_timestamps=shot_timestamps,
                    shot_classifications=shot_classifications,
                    output_path=vis_output_path
                )
                logger.info(f"Generated visualization video at {vis_output_path}")
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            return {
                'report_data': report_data,
                'output_dir': output_dir,
                'processing_time': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")
            raise
    
    def _extract_pose_keypoints(self, video_path: str) -> tuple:
        """
        Extract pose keypoints from a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (keypoints, frames, fps)
        """
        import cv2
        
        keypoints = []
        frames = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            # Process each frame
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Apply pose estimation to frame
                frame_keypoints = self.pose_estimator.estimate_pose(frame)
                
                # Store results
                keypoints.append(frame_keypoints)
                frames.append(frame)
                
                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
        finally:
            cap.release()
        
        return np.array(keypoints), frames, fps
    
    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to keypoint sequences.
        
        Args:
            keypoints: Array of shape (sequence_length, 17, 2)
            
        Returns:
            Smoothed keypoints
        """
        return self.temporal_filter.smooth_keypoints(keypoints)
    
    def _classify_shots(self, keypoints: np.ndarray) -> tuple:
        """
        Classify tennis shots in a keypoint sequence.
        
        Args:
            keypoints: Array of shape (sequence_length, 17, 2)
            
        Returns:
            Tuple of (shot_frames, shot_classes, shot_probs)
        """
        # Frame sliding window size
        window_size = self.config.get('classification', {}).get('sequence_length', 30)
        stride = self.config.get('classification', {}).get('stride', 15)
        
        shot_frames = []
        shot_classes = []
        shot_probs = []
        
        # Process in sliding windows
        for i in range(0, len(keypoints) - window_size + 1, stride):
            sequence = keypoints[i:i+window_size]
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Classify the sequence
            probs, class_idx = self.classifier.predict(sequence)
            prob = probs[0][class_idx[0]]
            
            # Check if probability exceeds threshold and is not 'neutral'
            threshold = self.config.get('classification', {}).get('threshold', 0.7)
            neutral_class = self.config.get('classification', {}).get('neutral_class_idx', 4)
            
            if prob > threshold and class_idx[0] != neutral_class:
                # Find the center frame of this window
                center_frame = i + window_size // 2
                
                # Add to detected shots
                shot_frames.append(center_frame)
                shot_classes.append(self.classifier.class_names[class_idx[0]] 
                                   if self.classifier.class_names 
                                   else str(class_idx[0]))
                shot_probs.append(prob)
        
        # Remove duplicates (shots detected in overlapping windows)
        if shot_frames:
            # Merge shots that are close to each other
            min_separation = self.config.get('classification', {}).get('min_frame_separation', 15)
            merged_frames = [shot_frames[0]]
            merged_classes = [shot_classes[0]]
            merged_probs = [shot_probs[0]]
            
            for i in range(1, len(shot_frames)):
                if shot_frames[i] - merged_frames[-1] > min_separation:
                    merged_frames.append(shot_frames[i])
                    merged_classes.append(shot_classes[i])
                    merged_probs.append(shot_probs[i])
                else:
                    # If the new shot has higher probability, replace the previous one
                    if shot_probs[i] > merged_probs[-1]:
                        merged_frames[-1] = shot_frames[i]
                        merged_classes[-1] = shot_classes[i]
                        merged_probs[-1] = shot_probs[i]
            
            shot_frames = merged_frames
            shot_classes = merged_classes
            shot_probs = merged_probs
        
        return shot_frames, shot_classes, shot_probs
    
    def train_classifier(self, data_dir: str, output_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the shot classifier on prepared data.
        
        Args:
            data_dir: Directory containing prepared training data
            output_model_path: Path to save the trained model (if None, uses config path)
            
        Returns:
            Dictionary with training results
        """
        try:
            # Load training and validation data
            logger.info(f"Loading training data from {data_dir}")
            
            # Load training sequences and labels
            with open(os.path.join(data_dir, 'training_data.npz'), 'rb') as f:
                data = np.load(f)
                X_train = data['sequences']
                y_train = data['labels']
            
            # Load validation sequences and labels
            with open(os.path.join(data_dir, 'validation_data.npz'), 'rb') as f:
                data = np.load(f)
                X_val = data['sequences']
                y_val = data['labels']
            
            # Load class names
            with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                class_names = metadata.get('class_names')
            
            logger.info(f"Loaded {len(X_train)} training sequences and {len(X_val)} validation sequences")
            logger.info(f"Class names: {class_names}")
            
            # Set up model path
            if output_model_path is None:
                output_model_path = self.config.get('classification', {}).get('model_path', 'models/rnn_classifier.h5')
            
            # Configure the classifier
            config = self.config.get('classification', {})
            config['model_path'] = output_model_path
            sequence_length, num_keypoints, coords = X_train.shape[1:]
            config['input_shape'] = (sequence_length, num_keypoints, coords)
            config['num_classes'] = len(class_names)
            
            # Create a new classifier
            self.classifier = RNNClassifier(config=config)
            
            # Train the model
            logger.info("Starting classifier training...")
            history = self.classifier.train(
                X_train, y_train, 
                X_val, y_val,
                epochs=config.get('epochs', 100),
                batch_size=config.get('batch_size', 32),
                class_names=class_names
            )
            
            # Save the model
            self.classifier.save_model(output_model_path)
            logger.info(f"Model saved to {output_model_path}")
            
            # Evaluate on validation set
            results = self.classifier.evaluate(
                X_val, y_val, 
                output_dir=os.path.join(os.path.dirname(output_model_path), 'evaluation')
            )
            
            return {
                'model_path': output_model_path,
                'training_history': history.history if hasattr(history, 'history') else None,
                'evaluation_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in classifier training: {e}")
            raise 