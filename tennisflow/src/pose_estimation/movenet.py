"""
MoveNet Pose Estimator Module

This module implements pose estimation using Google's MoveNet models 
via TensorFlow Hub, providing functions to extract human pose keypoints 
from images or video frames.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

# MoveNet keypoint indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Map to convert MoveNet output format to our internal format
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Define connections between keypoints for visualization
POSE_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle')
]

class MoveNetPoseEstimator:
    """
    Pose estimation using MoveNet models from TensorFlow Hub.
    
    This class provides methods to extract human pose keypoints from images
    or video frames using Google's MoveNet models.
    """
    
    def __init__(
        self, 
        model_type: str = "movenet_thunder",
        confidence_threshold: float = 0.3
    ):
        """
        Initialize the MoveNet pose estimator.
        
        Args:
            model_type: Type of MoveNet model to use ("movenet_thunder" or "movenet_lightning")
            confidence_threshold: Minimum confidence score to consider a keypoint valid
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.input_size = 256  # Default size
        
        # Load the MoveNet model
        self._load_model()
        
    def _load_model(self):
        """
        Load MoveNet model from TensorFlow Hub.
        """
        try:
            if self.model_type == "movenet_thunder":
                model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
                self.input_size = 256
            else:  # Default to lightning
                model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
                self.input_size = 192
                
            logger.info(f"Loading MoveNet model: {self.model_type}")
            self.model = hub.load(model_url)
            self.movenet = self.model.signatures['serving_default']
            logger.info(f"Successfully loaded MoveNet model")
        except Exception as e:
            logger.error(f"Failed to load MoveNet model: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """
        Preprocess image for MoveNet input.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image as TensorFlow tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to model's input size
        input_image = tf.image.resize_with_pad(
            tf.expand_dims(tf.convert_to_tensor(image_rgb), axis=0),
            self.input_size, self.input_size
        )
        
        # Convert to int32 as required by the model
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        return input_image
    
    def _postprocess_keypoints(
        self, 
        keypoints: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Convert MoveNet keypoints to our internal format.
        
        Args:
            keypoints: Raw keypoints from MoveNet [1, 1, 17, 3] - (y, x, confidence)
            image_shape: Original image shape (height, width)
            
        Returns:
            List of keypoint dictionaries in our internal format
        """
        height, width = image_shape
        processed_keypoints = []
        
        for idx, name in enumerate(KEYPOINT_NAMES):
            y, x, confidence = keypoints[0, 0, idx, :]
            
            # Convert normalized coordinates to pixel coordinates
            pixel_x = x * width
            pixel_y = y * height
            
            # Only include keypoints with confidence above threshold
            if confidence >= self.confidence_threshold:
                processed_keypoints.append({
                    "name": name,
                    "position": {"x": float(pixel_x), "y": float(pixel_y)},
                    "confidence": float(confidence)
                })
        
        return processed_keypoints
    
    def estimate_pose(self, image: np.ndarray) -> Dict:
        """
        Estimate pose from an input image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary with keypoints and connections
        """
        if self.model is None:
            logger.error("MoveNet model not loaded")
            return {"keypoints": [], "connections": []}
        
        try:
            # Preprocess image
            input_image = self._preprocess_image(image)
            
            # Run inference
            outputs = self.movenet(input_image)
            
            # Extract keypoints
            keypoints = outputs['output_0'].numpy()
            
            # Convert to our format
            processed_keypoints = self._postprocess_keypoints(keypoints, image.shape[:2])
            
            # Create connections for visualization
            connections = []
            for from_joint, to_joint in POSE_CONNECTIONS:
                connections.append({
                    "from": from_joint,
                    "to": to_joint
                })
            
            return {
                "keypoints": processed_keypoints,
                "connections": connections
            }
            
        except Exception as e:
            logger.error(f"Error estimating pose: {e}")
            return {"keypoints": [], "connections": []}
            
    def estimate_pose_from_frame(self, frame: np.ndarray) -> Dict:
        """
        Alias for estimate_pose - for compatibility with existing code.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary with keypoints and connections
        """
        return self.estimate_pose(frame)
        
    def get_keypoint_coordinates(self, keypoints: List[Dict]) -> np.ndarray:
        """
        Extract coordinates from keypoint dictionaries.
        
        Args:
            keypoints: List of keypoint dictionaries
            
        Returns:
            Array of keypoint coordinates shape (17, 2) - [x, y]
        """
        coordinates = np.zeros((len(KEYPOINT_NAMES), 2))
        
        # Create a mapping from keypoint name to its data
        keypoint_dict = {kp["name"]: kp["position"] for kp in keypoints if "name" in kp and "position" in kp}
        
        # Fill in coordinates for available keypoints
        for idx, name in enumerate(KEYPOINT_NAMES):
            if name in keypoint_dict:
                position = keypoint_dict[name]
                coordinates[idx, 0] = position["x"]
                coordinates[idx, 1] = position["y"]
                
        return coordinates
        
    def get_flat_keypoints(self, keypoints: List[Dict]) -> np.ndarray:
        """
        Convert keypoints to flat array format for RNN input.
        
        Args:
            keypoints: List of keypoint dictionaries
            
        Returns:
            Flattened array of keypoint coordinates shape (34,) - [x1, y1, x2, y2, ...]
        """
        coords = self.get_keypoint_coordinates(keypoints)
        return coords.flatten() 