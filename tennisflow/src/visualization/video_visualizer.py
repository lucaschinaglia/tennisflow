"""
Video Visualizer Module

This module visualizes tennis analysis results on video frames.
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class VideoVisualizer:
    """Visualizes tennis analysis results on video frames."""
    
    def __init__(self):
        """Initialize the video visualizer."""
        # Define colors for visualization
        self.keypoint_color = (0, 255, 0)  # Green
        self.connection_color = (0, 255, 255)  # Yellow
        self.text_color = (255, 255, 255)  # White
        self.box_color = (0, 0, 255)  # Red
    
    def create_pose_visualization(
        self, 
        video_path: str, 
        keypoints: np.ndarray, 
        output_path: str
    ) -> str:
        """
        Create a visualization of pose keypoints overlaid on video.
        
        Args:
            video_path: Path to the input video file
            keypoints: Array of keypoints, shape (frames, keypoints, coordinates)
            output_path: Path to save the output video
            
        Returns:
            Path to the output video
        """
        logger.info(f"Creating pose visualization: {output_path}")
        
        # Define COCO keypoint connections
        connections = [
            (5, 6),   # Left-Right Shoulder
            (5, 7),   # Left Shoulder - Left Elbow
            (6, 8),   # Right Shoulder - Right Elbow
            (7, 9),   # Left Elbow - Left Wrist
            (8, 10),  # Right Elbow - Right Wrist
            (5, 11),  # Left Shoulder - Left Hip
            (6, 12),  # Right Shoulder - Right Hip
            (11, 12), # Left-Right Hip
            (11, 13), # Left Hip - Left Knee
            (12, 14), # Right Hip - Right Knee
            (13, 15), # Left Knee - Left Ankle
            (14, 16)  # Right Knee - Right Ankle
        ]
        
        try:
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return ""
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create output video writer
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process each frame
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip if we don't have keypoints for this frame
                if frame_idx >= len(keypoints):
                    out.write(frame)
                    frame_idx += 1
                    continue
                
                # Draw keypoints and connections
                for i in range(17):  # 17 keypoints in COCO format
                    x, y = keypoints[frame_idx, i]
                    if x > 0 and y > 0:  # Only draw valid keypoints
                        cv2.circle(frame, (int(x), int(y)), 5, self.keypoint_color, -1)
                
                # Draw connections
                for conn in connections:
                    p1 = keypoints[frame_idx, conn[0]]
                    p2 = keypoints[frame_idx, conn[1]]
                    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                        cv2.line(frame, (int(p1[0]), int(p1[1])), 
                                (int(p2[0]), int(p2[1])), 
                                self.connection_color, 2)
                
                # Write frame
                out.write(frame)
                frame_idx += 1
            
            # Release resources
            cap.release()
            out.release()
            
            logger.info(f"Pose visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating pose visualization: {e}")
            return ""
    
    def create_shot_visualization(
        self, 
        video_path: str, 
        keypoints: np.ndarray, 
        shot_timestamps: List[Dict], 
        shot_classifications: List[Dict], 
        output_path: str
    ) -> str:
        """
        Create a visualization of shot detections overlaid on video.
        
        Args:
            video_path: Path to the input video file
            keypoints: Array of keypoints, shape (frames, keypoints, coordinates)
            shot_timestamps: List of shot timestamp dictionaries
            shot_classifications: List of shot classification dictionaries
            output_path: Path to save the output video
            
        Returns:
            Path to the output video
        """
        logger.info(f"Creating shot visualization: {output_path}")
        
        # For this placeholder, we'll just copy the pose visualization logic
        # and add shot type labels
        try:
            # Create a map from frame index to shot info
            frame_to_shot = {}
            for i, shot in enumerate(shot_timestamps):
                start_frame = shot['start_frame']
                end_frame = shot['end_frame']
                
                # Get classification if available
                shot_type = "Unknown"
                confidence = 0.0
                if i < len(shot_classifications):
                    shot_type = shot_classifications[i].get('class_name', "Unknown")
                    confidence = shot_classifications[i].get('confidence', 0.0)
                
                # Map each frame in the shot range to this shot info
                for frame_idx in range(start_frame, end_frame + 1):
                    frame_to_shot[frame_idx] = {
                        'type': shot_type,
                        'confidence': confidence
                    }
            
            # Process the video
            return self.create_pose_visualization(video_path, keypoints, output_path)
            
        except Exception as e:
            logger.error(f"Error creating shot visualization: {e}")
            return "" 