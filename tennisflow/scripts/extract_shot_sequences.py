#!/usr/bin/env python3
"""
Extract tennis shot sequences from annotated videos.

This script processes annotated videos to extract fixed-length sequences of
pose keypoints around each annotated shot. The extracted sequences can be used
for training the RNN classifier.

Based on extract_shots_as_features.py from the tennis_shot_recognition repository.
"""

import os
import sys
import argparse
import csv
import numpy as np
import cv2
import logging
import json
from tqdm import tqdm
import pandas as pd
import random

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.pose_estimation.pose_estimator import MoveNetEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_annotations(annotation_file):
    """
    Load annotations from a CSV file.
    
    Args:
        annotation_file: Path to the annotation CSV file
        
    Returns:
        List of tuples (shot_type, frame_id, timestamp)
    """
    annotations = []
    
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        # Determine the format based on the header
        has_timestamp = 'Timestamp' in header or 'timestamp' in header
        frame_col = header.index('FrameId') if 'FrameId' in header else header.index('frame_id')
        shot_col = header.index('Shot') if 'Shot' in header else header.index('shot')
        timestamp_col = header.index('Timestamp') if 'Timestamp' in header else (header.index('timestamp') if has_timestamp else None)
        
        for row in reader:
            shot_type = row[shot_col].lower()
            frame_id = int(row[frame_col])
            timestamp = float(row[timestamp_col]) if timestamp_col is not None else None
            
            annotations.append((shot_type, frame_id, timestamp))
    
    return annotations

def extract_neutral_frames(annotations, total_frames, sequence_length, min_spacing=60):
    """
    Extract neutral frames from the video.
    
    Args:
        annotations: List of tuples (shot_type, frame_id, timestamp)
        total_frames: Total number of frames in the video
        sequence_length: Length of sequences to extract
        min_spacing: Minimum spacing between neutral frames and shot frames
        
    Returns:
        List of frame indices for neutral sequences
    """
    # Get all annotated frame ids
    shot_frames = [f for _, f, _ in annotations]
    
    # Create mask of valid neutral frame starts
    valid_mask = np.ones(total_frames, dtype=bool)
    
    # Mark frames too close to shots as invalid
    for shot_frame in shot_frames:
        start = max(0, shot_frame - sequence_length - min_spacing)
        end = min(total_frames, shot_frame + min_spacing)
        valid_mask[start:end] = False
    
    # Also mark frames too close to the end of the video
    valid_mask[total_frames - sequence_length:] = False
    
    # Get valid frame indices
    valid_indices = np.where(valid_mask)[0]
    
    # If no valid frames, return empty list
    if len(valid_indices) == 0:
        return []
    
    # Select random frames for neutral sequences
    num_neutral = min(len(shot_frames), len(valid_indices))
    neutral_frames = np.random.choice(valid_indices, num_neutral, replace=False)
    
    return list(neutral_frames)

def extract_shot_sequence(video_path, shot_type, frame_id, sequence_length, pose_estimator, show=False):
    """
    Extract a fixed-length sequence around a shot frame.
    
    Args:
        video_path: Path to the video file
        shot_type: Type of the shot (e.g., 'forehand', 'backhand')
        frame_id: Frame ID of the shot
        sequence_length: Length of sequence to extract
        pose_estimator: Initialized pose estimator
        show: Whether to show the extracted frames
        
    Returns:
        NumPy array of keypoints for the sequence
    """
    # Calculate sequence range
    half_seq = sequence_length // 2
    start_frame = max(0, frame_id - half_seq)
    end_frame = start_frame + sequence_length
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return None
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust if sequence extends beyond the video
    if end_frame > total_frames:
        end_frame = total_frames
        start_frame = max(0, end_frame - sequence_length)
    
    # Initialize keypoints array
    sequence_keypoints = []
    
    try:
        # Process frames in sequence
        for i in range(start_frame, end_frame):
            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Error reading frame {i}")
                continue
            
            # Apply pose estimation
            keypoints = pose_estimator.estimate_pose(frame)
            sequence_keypoints.append(keypoints)
            
            # Show frame if requested
            if show:
                # Draw keypoints on frame
                for kp_idx, keypoint in enumerate(keypoints):
                    x, y = keypoint[:2]
                    conf = keypoint[2] if keypoint.shape[0] > 2 else 1.0
                    
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                
                # Add information
                cv2.putText(frame, f"Shot: {shot_type}, Frame: {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Shot Sequence', frame)
                if cv2.waitKey(30) & 0xFF == 27:  # Esc
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
    
    # Convert to numpy array
    sequence_keypoints = np.array(sequence_keypoints)
    
    # Pad sequence if not enough frames
    if len(sequence_keypoints) < sequence_length:
        padding = np.zeros((sequence_length - len(sequence_keypoints), 
                           sequence_keypoints.shape[1], sequence_keypoints.shape[2]))
        sequence_keypoints = np.concatenate([sequence_keypoints, padding])
    
    return sequence_keypoints

def save_sequence_csv(keypoints, shot_type, output_path):
    """
    Save a keypoint sequence to a CSV file.
    
    Args:
        keypoints: NumPy array of keypoints
        shot_type: Type of the shot
        output_path: Path to save the CSV file
        
    Returns:
        True if successful, False otherwise
    """
    # Create DataFrame
    columns = []
    for kp_idx in range(keypoints.shape[1]):
        columns.extend([f'keypoint{kp_idx}_y', f'keypoint{kp_idx}_x'])
    
    df = pd.DataFrame(columns=columns + ['shot'])
    
    # Format keypoints
    for frame_idx in range(keypoints.shape[0]):
        row = {}
        for kp_idx in range(keypoints.shape[1]):
            row[f'keypoint{kp_idx}_y'] = keypoints[frame_idx, kp_idx, 1]
            row[f'keypoint{kp_idx}_x'] = keypoints[frame_idx, kp_idx, 0]
        row['shot'] = shot_type
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    return True

def process_video(video_path, annotation_file, output_dir, show=False, sequence_length=30):
    """
    Process a video and extract shot sequences.
    
    Args:
        video_path: Path to the video file
        annotation_file: Path to the annotation CSV file
        output_dir: Directory to save extracted sequences
        show: Whether to show the extracted frames
        sequence_length: Length of sequences to extract
        
    Returns:
        List of extracted sequence paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pose estimator
    pose_estimator = MoveNetEstimator()
    
    # Load annotations
    annotations = load_annotations(annotation_file)
    logger.info(f"Loaded {len(annotations)} annotations from {annotation_file}")
    
    # Open video to get total frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Extract neutral frames
    neutral_frames = extract_neutral_frames(annotations, total_frames, sequence_length)
    logger.info(f"Generated {len(neutral_frames)} neutral frames")
    
    # Add neutral shots to annotations
    for frame in neutral_frames:
        annotations.append(('neutral', frame, None))
    
    # Extract sequences for each annotation
    extracted_paths = []
    
    for idx, (shot_type, frame_id, _) in enumerate(tqdm(annotations, desc="Extracting sequences")):
        # Extract sequence
        sequence = extract_shot_sequence(
            video_path, shot_type, frame_id, sequence_length, pose_estimator, show
        )
        
        if sequence is None:
            logger.warning(f"Failed to extract sequence for {shot_type} at frame {frame_id}")
            continue
        
        # Generate filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        count = 1
        while True:
            output_path = os.path.join(output_dir, f"{shot_type}_{base_name}_{count:03d}.csv")
            if not os.path.exists(output_path):
                break
            count += 1
        
        # Save sequence
        save_sequence_csv(sequence, shot_type, output_path)
        extracted_paths.append(output_path)
        
        logger.debug(f"Saved sequence to {output_path}")
    
    logger.info(f"Extracted {len(extracted_paths)} sequences to {output_dir}")
    
    # Generate metadata
    shot_counts = {}
    for shot_type, _, _ in annotations:
        shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
    
    metadata = {
        'video': os.path.basename(video_path),
        'annotation_file': os.path.basename(annotation_file),
        'total_frames': total_frames,
        'sequence_length': sequence_length,
        'extracted_sequences': len(extracted_paths),
        'shot_distribution': shot_counts
    }
    
    metadata_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return extracted_paths

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract tennis shot sequences from annotated videos")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("annotation_file", help="Path to the annotation CSV file")
    parser.add_argument("output_dir", help="Directory to save extracted sequences")
    parser.add_argument("--show", action="store_true", help="Show the extracted frames")
    parser.add_argument("--sequence-length", type=int, default=30, 
                       help="Length of sequences to extract (default: 30)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.isfile(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    if not os.path.isfile(args.annotation_file):
        logger.error(f"Annotation file not found: {args.annotation_file}")
        return 1
    
    # Process video
    process_video(
        args.video_path,
        args.annotation_file,
        args.output_dir,
        args.show,
        args.sequence_length
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 