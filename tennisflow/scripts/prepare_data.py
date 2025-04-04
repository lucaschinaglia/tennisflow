#!/usr/bin/env python
"""
Data Preparation Script

This script processes tennis videos to:
1. Extract frames at a specified rate
2. Run MoveNet pose estimation on each frame
3. Apply temporal smoothing to the keypoint sequences
4. Extract fixed-length sequences centered around annotated shot times
5. Save these sequences for RNN training

The workflow is based on the antoinekeller/tennis_shot_recognition repository approach.
"""

import os
import sys
import cv2
import numpy as np
import csv
import logging
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import local modules
from src.pose_estimation.movenet import MoveNetPoseEstimator
from src.pose_estimation.smoothing import smooth_keypoint_sequence
from src.utils.visualization import draw_keypoints

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_preparation")

class DataPreparation:
    """
    Tennis data preparation class for creating RNN training data.
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize data preparation with config.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.pose_estimator = None
        
        # Set paths
        paths = self.config.get("paths", {})
        self.raw_data_dir = paths.get("raw_data", "../data/raw")
        self.processed_data_dir = paths.get("processed_data", "../data/processed")
        
        # Make sure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize pose estimator
        self._init_pose_estimator()
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
            
    def _init_pose_estimator(self):
        """
        Initialize the pose estimator.
        """
        pose_config = self.config.get("pose_estimation", {})
        model_type = pose_config.get("model_type", "movenet_thunder")
        min_detection_confidence = pose_config.get("min_detection_confidence", 0.3)
        
        logger.info(f"Initializing {model_type} pose estimator")
        self.pose_estimator = MoveNetPoseEstimator(
            model_type=model_type,
            confidence_threshold=min_detection_confidence
        )
        
    def process_video(
        self,
        video_path: str,
        annotation_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        sample_rate: int = 1,
        sequence_length: int = 30,
        save_visualization: bool = False
    ) -> Dict[str, List]:
        """
        Process a video to extract keypoint sequences.
        
        Args:
            video_path: Path to the video file
            annotation_path: Path to annotation file (optional)
            output_dir: Output directory (optional)
            sample_rate: Process every Nth frame
            sequence_length: Length of extracted sequences
            save_visualization: Whether to save visualization frames
            
        Returns:
            Dictionary of extracted sequences by class
        """
        if self.pose_estimator is None:
            self._init_pose_estimator()
            
        # Determine output directory
        if output_dir is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(self.processed_data_dir, video_name)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Load annotations if provided
        annotations = []
        if annotation_path:
            annotations = self._load_annotations(annotation_path)
            logger.info(f"Loaded {len(annotations)} annotations")
            
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {}
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Create directories for visualization if needed
        vis_dir = os.path.join(output_dir, "visualization") if save_visualization else None
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
        
        # Process frames and extract keypoints
        all_keypoints = []
        frame_count = 0
        
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process at the specified sample rate
                if frame_count % sample_rate == 0:
                    # Estimate pose
                    pose_result = self.pose_estimator.estimate_pose(frame)
                    
                    # Get keypoint coordinates
                    if pose_result["keypoints"]:
                        keypoints = self.pose_estimator.get_keypoint_coordinates(pose_result["keypoints"])
                        
                        # Store keypoints along with frame number
                        all_keypoints.append({
                            "frame": frame_count,
                            "keypoints": keypoints
                        })
                        
                        # Save visualization if required
                        if vis_dir:
                            vis_frame = draw_keypoints(
                                frame, 
                                pose_result["keypoints"], 
                                pose_result["connections"]
                            )
                            cv2.imwrite(
                                os.path.join(vis_dir, f"frame_{frame_count:06d}.jpg"),
                                vis_frame
                            )
                
                frame_count += 1
                pbar.update(1)
                
        cap.release()
        logger.info(f"Processed {frame_count} frames, extracted keypoints from {len(all_keypoints)} frames")
        
        # Apply temporal smoothing to keypoint sequences
        smoothed_keypoints = self._apply_smoothing(all_keypoints)
        
        # Extract sequences around annotations
        sequences_by_class = self._extract_sequences(
            smoothed_keypoints,
            annotations,
            sequence_length,
            fps,
            sample_rate
        )
        
        # Save extracted sequences
        self._save_sequences(sequences_by_class, output_dir)
        
        return sequences_by_class
        
    def _load_annotations(self, annotation_path: str) -> List[Dict]:
        """
        Load shot annotations from a file.
        
        The annotation file can be in CSV or JSON format:
        - CSV: frame_number, shot_type
        - JSON: list of {"frame": N, "shot_type": "type"} objects
        
        Args:
            annotation_path: Path to the annotation file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        # Determine file type
        ext = os.path.splitext(annotation_path)[1].lower()
        
        try:
            if ext == '.csv':
                with open(annotation_path, 'r') as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)  # Skip header if exists
                    
                    for row in reader:
                        if len(row) >= 2:
                            annotations.append({
                                "frame": int(row[0]),
                                "shot_type": row[1].lower()
                            })
            elif ext == '.json':
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                    
                    for item in data:
                        if "frame" in item and "shot_type" in item:
                            annotations.append({
                                "frame": int(item["frame"]),
                                "shot_type": item["shot_type"].lower()
                            })
            else:
                logger.error(f"Unsupported annotation file format: {ext}")
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            
        return annotations
        
    def _apply_smoothing(self, keypoint_frames: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to keypoint sequences.
        
        Args:
            keypoint_frames: List of dictionaries with frame and keypoints
            
        Returns:
            List of dictionaries with frame and smoothed keypoints
        """
        if len(keypoint_frames) < 3:
            logger.warning("Too few frames for smoothing")
            return keypoint_frames
            
        # Extract keypoints into a numpy array
        frames = np.array([kf["frame"] for kf in keypoint_frames])
        keypoints = np.array([kf["keypoints"] for kf in keypoint_frames])
        
        # Apply smoothing based on config
        smoothing_config = self.config.get("smoothing", {})
        method = smoothing_config.get("method", "savgol")
        
        if method == "savgol":
            savgol_config = smoothing_config.get("savgol", {})
            window_size = savgol_config.get("window_size", 15)
            polynomial_order = savgol_config.get("polynomial_order", 3)
            
            logger.info(f"Applying Savitzky-Golay smoothing (window_size={window_size}, polynomial_order={polynomial_order})")
            
            smoothed = smooth_keypoint_sequence(
                keypoints,
                method="savgol",
                window_size=window_size,
                polynomial_order=polynomial_order
            )
        elif method == "kalman":
            kalman_config = smoothing_config.get("kalman", {})
            process_noise = kalman_config.get("process_noise", 0.01)
            measurement_noise = kalman_config.get("measurement_noise", 0.1)
            
            logger.info(f"Applying Kalman smoothing (process_noise={process_noise}, measurement_noise={measurement_noise})")
            
            smoothed = smooth_keypoint_sequence(
                keypoints,
                method="kalman",
                process_noise=process_noise,
                measurement_noise=measurement_noise
            )
        else:
            logger.warning(f"Unknown smoothing method: {method}, using raw keypoints")
            smoothed = keypoints
        
        # Create result list
        result = []
        for i, frame in enumerate(frames):
            result.append({
                "frame": frame,
                "keypoints": smoothed[i]
            })
            
        return result
        
    def _extract_sequences(
        self,
        keypoint_frames: List[Dict],
        annotations: List[Dict],
        sequence_length: int,
        fps: float,
        sample_rate: int
    ) -> Dict[str, List[np.ndarray]]:
        """
        Extract fixed-length sequences around annotated shots.
        
        Args:
            keypoint_frames: List of dictionaries with frame and keypoints
            annotations: List of annotation dictionaries
            sequence_length: Length of sequences to extract
            fps: Frames per second
            sample_rate: Frame sampling rate used during processing
            
        Returns:
            Dictionary of sequences grouped by shot type
        """
        sequences_by_class = {}
        
        # If no annotations, use automatic segmentation based on movement
        if not annotations:
            logger.info("No annotations provided, using automatic segmentation")
            # TODO: Implement automatic segmentation using wrist velocity
            return sequences_by_class
        
        # Sort keypoint frames by frame number
        keypoint_frames.sort(key=lambda x: x["frame"])
        
        # Convert keypoints to a dictionary for fast lookup
        keypoints_dict = {kf["frame"]: kf["keypoints"] for kf in keypoint_frames}
        frame_numbers = sorted(keypoints_dict.keys())
        
        # Extract sequences centered around each annotation
        for annotation in annotations:
            frame = annotation["frame"]
            shot_type = annotation["shot_type"]
            
            # Adjust for sample rate
            frame_adjusted = frame // sample_rate * sample_rate
            
            # Find the closest processed frame
            closest_frame = min(frame_numbers, key=lambda x: abs(x - frame_adjusted))
            
            # Calculate sequence boundaries
            half_length = sequence_length // 2
            start_idx = max(0, frame_numbers.index(closest_frame) - half_length)
            end_idx = min(len(frame_numbers), start_idx + sequence_length)
            
            # Adjust start if end is out of bounds
            if end_idx == len(frame_numbers):
                start_idx = max(0, end_idx - sequence_length)
            
            # Extract the sequence
            sequence_frames = frame_numbers[start_idx:end_idx]
            
            # Skip if sequence is too short
            if len(sequence_frames) < sequence_length:
                logger.warning(f"Skipping annotation at frame {frame}: sequence too short ({len(sequence_frames)} < {sequence_length})")
                continue
            
            # Create a sequence array
            sequence = np.array([keypoints_dict[f] for f in sequence_frames])
            
            # Add to sequences by class
            if shot_type not in sequences_by_class:
                sequences_by_class[shot_type] = []
            
            sequences_by_class[shot_type].append(sequence)
            
        # Add "neutral" sequences (areas without annotations)
        if annotations and "neutral" not in sequences_by_class:
            neutral_sequences = self._extract_neutral_sequences(
                keypoint_frames,
                annotations,
                sequence_length,
                sample_rate
            )
            if neutral_sequences:
                sequences_by_class["neutral"] = neutral_sequences
        
        # Log statistics
        for shot_type, sequences in sequences_by_class.items():
            logger.info(f"Extracted {len(sequences)} sequences for shot type '{shot_type}'")
            
        return sequences_by_class
    
    def _extract_neutral_sequences(
        self,
        keypoint_frames: List[Dict],
        annotations: List[Dict],
        sequence_length: int,
        sample_rate: int
    ) -> List[np.ndarray]:
        """
        Extract sequences from 'neutral' areas (between shots).
        
        Args:
            keypoint_frames: List of dictionaries with frame and keypoints
            annotations: List of annotation dictionaries
            sequence_length: Length of sequences to extract
            sample_rate: Frame sampling rate used during processing
            
        Returns:
            List of neutral sequences
        """
        neutral_sequences = []
        
        # Sort keypoint frames by frame number
        keypoint_frames.sort(key=lambda x: x["frame"])
        
        # Convert keypoints to a dictionary for fast lookup
        keypoints_dict = {kf["frame"]: kf["keypoints"] for kf in keypoint_frames}
        frame_numbers = sorted(keypoints_dict.keys())
        
        # Sort annotations by frame
        sorted_annotations = sorted(annotations, key=lambda x: x["frame"])
        
        # Buffer around each annotation (in frames)
        buffer_frames = int(1.0 * sequence_length)  # e.g., 1x sequence length
        
        # Create a mask of "shot" frames (including buffer)
        shot_frames = set()
        for annotation in sorted_annotations:
            frame = annotation["frame"]
            frame_adjusted = frame // sample_rate * sample_rate
            
            # Add buffer around shot
            for f in range(frame_adjusted - buffer_frames, frame_adjusted + buffer_frames + 1):
                shot_frames.add(f)
        
        # Find sequences of "neutral" frames
        neutral_start = 0
        while neutral_start < len(frame_numbers):
            # Skip if current frame is a shot frame
            if frame_numbers[neutral_start] in shot_frames:
                neutral_start += 1
                continue
            
            # Check if we have enough frames for a sequence
            if neutral_start + sequence_length > len(frame_numbers):
                break
            
            # Check if all frames in the potential sequence are neutral
            is_neutral_sequence = True
            for i in range(sequence_length):
                if frame_numbers[neutral_start + i] in shot_frames:
                    is_neutral_sequence = False
                    break
            
            if is_neutral_sequence:
                # Extract the sequence
                sequence_frames = frame_numbers[neutral_start:neutral_start + sequence_length]
                sequence = np.array([keypoints_dict[f] for f in sequence_frames])
                neutral_sequences.append(sequence)
                
                # Move to next potential sequence
                neutral_start += sequence_length
            else:
                neutral_start += 1
        
        # Limit the number of neutral sequences to balance the dataset
        max_neutral = max([len(seqs) for shot_type, seqs in neutral_sequences.items()]) if neutral_sequences else 0
        if len(neutral_sequences) > max_neutral:
            # Randomly sample to match the largest shot class
            indices = np.random.choice(
                len(neutral_sequences),
                size=max_neutral,
                replace=False
            )
            neutral_sequences = [neutral_sequences[i] for i in indices]
            
        return neutral_sequences
    
    def _save_sequences(self, sequences_by_class: Dict[str, List[np.ndarray]], output_dir: str):
        """
        Save extracted sequences to files.
        
        Args:
            sequences_by_class: Dictionary of sequences by shot type
            output_dir: Output directory
        """
        for shot_type, sequences in sequences_by_class.items():
            # Create subdirectory for this shot type
            shot_dir = os.path.join(output_dir, shot_type)
            os.makedirs(shot_dir, exist_ok=True)
            
            # Save each sequence
            for i, sequence in enumerate(sequences):
                # Save as numpy file
                np_path = os.path.join(shot_dir, f"sequence_{i:04d}.npy")
                np.save(np_path, sequence)
                
                # Also save as CSV for compatibility
                csv_path = os.path.join(shot_dir, f"sequence_{i:04d}.csv")
                
                # Flatten the sequence for CSV
                flattened = sequence.reshape(sequence.shape[0], -1)
                
                # Create headers
                headers = []
                for kp_idx in range(sequence.shape[1]):
                    headers.extend([f"x_{kp_idx}", f"y_{kp_idx}"])
                
                # Save as CSV
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(flattened)
            
            logger.info(f"Saved {len(sequences)} sequences for shot type '{shot_type}'")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process tennis videos for RNN training")
    parser.add_argument("--video", "-v", type=str, help="Path to the video file")
    parser.add_argument("--annotation", "-a", type=str, help="Path to the annotation file")
    parser.add_argument("--config", "-c", type=str, default="../config.yaml", help="Path to the configuration file")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--sequence-length", "-s", type=int, default=30, help="Length of sequences to extract")
    parser.add_argument("--sample-rate", "-r", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--visualize", action="store_true", help="Save visualization frames")
    args = parser.parse_args()
    
    if not args.video:
        logger.error("Video path must be provided. Use --video or -v.")
        return
    
    # Initialize data preparation
    data_prep = DataPreparation(args.config)
    
    # Process the video
    data_prep.process_video(
        video_path=args.video,
        annotation_path=args.annotation,
        output_dir=args.output,
        sequence_length=args.sequence_length,
        sample_rate=args.sample_rate,
        save_visualization=args.visualize
    )
    
if __name__ == "__main__":
    main() 