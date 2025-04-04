#!/usr/bin/env python3
"""
Test script for the MoveNet pose estimator.
This script processes a video and visualizes the detected poses.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to sys.path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.pose_estimation.pose_estimator import MoveNetEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define keypoint connections for drawing skeleton
KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face and head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    """Draw keypoints and skeleton on a frame."""
    h, w, _ = frame.shape
    
    # Create a copy of the frame to draw on
    vis_img = frame.copy()
    
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint
        conf = keypoints[i, 2] if keypoints.shape[1] > 2 else 1.0
        
        if conf >= confidence_threshold:
            # Convert normalized coordinates to pixel coordinates if needed
            if 0 <= x <= 1 and 0 <= y <= 1:
                x = int(x * w)
                y = int(y * h)
                
            # Draw keypoint
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Draw connections
    for connection in KEYPOINT_CONNECTIONS:
        start_idx, end_idx = connection
        
        start_keypoint = keypoints[start_idx]
        end_keypoint = keypoints[end_idx]
        
        start_conf = keypoints[start_idx, 2] if keypoints.shape[1] > 2 else 1.0
        end_conf = keypoints[end_idx, 2] if keypoints.shape[1] > 2 else 1.0
        
        if start_conf >= confidence_threshold and end_conf >= confidence_threshold:
            # Convert normalized coordinates to pixel coordinates if needed
            start_x, start_y = start_keypoint[:2]
            end_x, end_y = end_keypoint[:2]
            
            if 0 <= start_x <= 1 and 0 <= start_y <= 1:
                start_x = int(start_x * w)
                start_y = int(start_y * h)
            
            if 0 <= end_x <= 1 and 0 <= end_y <= 1:
                end_x = int(end_x * w)
                end_y = int(end_y * h)
            
            # Draw line
            cv2.line(vis_img, (int(start_x), int(start_y)), 
                    (int(end_x), int(end_y)), (0, 255, 255), 2)
    
    return vis_img

def process_video(video_path, output_path, model_type='lightning', 
                 confidence_threshold=0.3, max_frames=None):
    """Process a video and create visualization with pose keypoints."""
    # Initialize pose estimator
    pose_estimator = MoveNetEstimator(model_type=model_type)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        # Process frames
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            success, frame = cap.read()
            if not success:
                break
            
            # Estimate pose
            keypoints = pose_estimator.estimate_pose(frame)
            
            # Draw keypoints on frame
            vis_frame = draw_keypoints(frame, keypoints, confidence_threshold)
            
            # Add frame number to top-left corner
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write to output video
            out.write(vis_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        logger.info(f"Processed {frame_idx} frames")
        
    finally:
        cap.release()
        out.release()
    
    logger.info(f"Output saved to {output_path}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test MoveNet pose estimator on a video")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", "-o", default="output_pose.mp4", 
                       help="Path for output video")
    parser.add_argument("--model", "-m", choices=['lightning', 'thunder'], 
                       default='lightning', help="MoveNet model type")
    parser.add_argument("--confidence", "-c", type=float, default=0.3,
                       help="Confidence threshold for keypoints")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.video_path):
        logger.error(f"Input video not found: {args.video_path}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the video
    success = process_video(
        args.video_path,
        args.output,
        model_type=args.model,
        confidence_threshold=args.confidence,
        max_frames=args.max_frames
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 