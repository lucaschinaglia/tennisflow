"""
Pose Estimator Module

This module provides a wrapper around MoveNet for pose estimation in tennis videos.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logger = logging.getLogger(__name__)

class MoveNetEstimator:
    """Wrapper around MoveNet for pose estimation."""
    
    def __init__(self, model_type: str = "lightning"):
        """
        Initialize the pose estimator.
        
        Args:
            model_type: MoveNet model type ("lightning" or "thunder")
        """
        self.model_type = model_type
        logger.info(f"Initializing MoveNet pose estimator with model type: {model_type}")
        
        # Initialize MoveNet
        try:
            # Import here to avoid loading TF at module level
            from .movenet import MoveNetPoseEstimator
            self.model = MoveNetPoseEstimator(model_type=model_type)
            logger.info("MoveNet model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MoveNet model: {e}")
            self.model = None
    
    def estimate_pose(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate pose from a video frame.
        
        Args:
            frame: Video frame as numpy array (RGB format)
            
        Returns:
            Array of keypoints, shape (17, 2)
        """
        if self.model is None:
            logger.error("MoveNet model not initialized")
            return np.zeros((17, 2))
        
        try:
            # Get pose estimation
            pose_result = self.model.estimate_pose(frame)
            
            # Extract keypoints
            keypoints_data = pose_result.get("keypoints", [])
            
            # Convert to numpy array [17, 2]
            keypoints = np.zeros((17, 2))
            
            for kp in keypoints_data:
                name = kp.get("name", "")
                position = kp.get("position", {})
                confidence = kp.get("confidence", 0.0)
                
                # Skip keypoints with low confidence
                if confidence < 0.3:
                    continue
                
                # Map keypoint name to index
                idx = self._name_to_index(name)
                if idx >= 0:
                    keypoints[idx, 0] = position.get("x", 0)
                    keypoints[idx, 1] = position.get("y", 0)
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Error estimating pose: {e}")
            return np.zeros((17, 2))
    
    def _name_to_index(self, name: str) -> int:
        """
        Convert keypoint name to index.
        
        Args:
            name: Keypoint name
            
        Returns:
            Keypoint index (0-16)
        """
        keypoint_map = {
            "nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 3,
            "right_ear": 4,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_ankle": 15,
            "right_ankle": 16
        }
        
        return keypoint_map.get(name, -1) 