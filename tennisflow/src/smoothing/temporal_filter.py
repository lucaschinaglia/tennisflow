"""
Temporal Filter Module

This module provides temporal smoothing filters for pose keypoint sequences.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class TemporalFilter:
    """
    Temporal filter for smoothing keypoint sequences.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the temporal filter.
        
        Args:
            config: Configuration for the filter
        """
        self.config = config or {}
        self.method = self.config.get("method", "savgol")
        logger.info(f"Initializing temporal filter with method: {self.method}")
    
    def smooth_keypoints(self, keypoint_sequence: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to keypoint sequence.
        
        Args:
            keypoint_sequence: Array of keypoints, shape (frames, keypoints, coordinates)
            
        Returns:
            Smoothed keypoint sequence
        """
        if len(keypoint_sequence) <= 3:
            logger.warning("Sequence too short for smoothing")
            return keypoint_sequence
        
        try:
            if self.method == "savgol":
                return self._savgol_filter(keypoint_sequence)
            elif self.method == "kalman":
                return self._kalman_filter(keypoint_sequence)
            else:
                logger.warning(f"Unknown smoothing method: {self.method}, using Savgol filter")
                return self._savgol_filter(keypoint_sequence)
        except Exception as e:
            logger.error(f"Error during smoothing: {e}")
            return keypoint_sequence
    
    def _savgol_filter(self, keypoint_sequence: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to keypoint sequence.
        
        Args:
            keypoint_sequence: Array of keypoints, shape (frames, keypoints, coordinates)
            
        Returns:
            Smoothed keypoint sequence
        """
        try:
            from scipy import signal
            
            # Get filter parameters
            savgol_config = self.config.get("savgol", {})
            window_size = savgol_config.get("window_size", 15)
            polynomial_order = savgol_config.get("polynomial_order", 3)
            
            # Ensure window size is odd
            if window_size % 2 == 0:
                window_size += 1
            
            # Ensure window size is less than sequence length
            if window_size >= len(keypoint_sequence):
                window_size = min(len(keypoint_sequence) - 2, 15)
                window_size = window_size if window_size % 2 == 1 else window_size - 1
                logger.warning(f"Adjusted window size to {window_size} due to short sequence")
            
            # Ensure polynomial order is valid
            if polynomial_order >= window_size:
                polynomial_order = window_size - 1
                logger.warning(f"Adjusted polynomial order to {polynomial_order}")
            
            # Apply filter to each keypoint coordinate
            n_frames, n_keypoints, n_coords = keypoint_sequence.shape
            smoothed = np.zeros_like(keypoint_sequence)
            
            for k in range(n_keypoints):
                for c in range(n_coords):
                    # Get trajectory for this keypoint coordinate
                    trajectory = keypoint_sequence[:, k, c]
                    
                    # Find valid points (non-zero values)
                    valid_indices = np.where(trajectory != 0)[0]
                    
                    if len(valid_indices) > window_size:
                        # Extract valid trajectory
                        valid_trajectory = trajectory[valid_indices]
                        
                        # Apply Savitzky-Golay filter
                        smoothed_valid = signal.savgol_filter(
                            valid_trajectory,
                            window_size,
                            polynomial_order
                        )
                        
                        # Put back smoothed values
                        smoothed_trajectory = np.copy(trajectory)
                        smoothed_trajectory[valid_indices] = smoothed_valid
                        smoothed[:, k, c] = smoothed_trajectory
                    else:
                        # Not enough points for filtering
                        smoothed[:, k, c] = trajectory
            
            return smoothed
            
        except ImportError:
            logger.error("SciPy not available, skipping smoothing")
            return keypoint_sequence
        except Exception as e:
            logger.error(f"Error in Savgol filtering: {e}")
            return keypoint_sequence
    
    def _kalman_filter(self, keypoint_sequence: np.ndarray) -> np.ndarray:
        """
        Apply Kalman filter to keypoint sequence.
        
        Args:
            keypoint_sequence: Array of keypoints, shape (frames, keypoints, coordinates)
            
        Returns:
            Smoothed keypoint sequence
        """
        # Get filter parameters
        kalman_config = self.config.get("kalman", {})
        process_noise = kalman_config.get("process_noise", 0.01)
        measurement_noise = kalman_config.get("measurement_noise", 0.1)
        
        # Apply Simple Kalman filter
        n_frames, n_keypoints, n_coords = keypoint_sequence.shape
        smoothed = np.zeros_like(keypoint_sequence)
        
        # State for each keypoint
        kalman_states = {}
        
        for i in range(n_frames):
            for k in range(n_keypoints):
                kp_id = f"kp_{k}"
                
                # Skip if the keypoint is not detected (zero)
                if np.all(keypoint_sequence[i, k] == 0):
                    if kp_id in kalman_states:
                        # Use predicted state if available
                        smoothed[i, k] = kalman_states[kp_id]["pos"]
                    else:
                        smoothed[i, k] = 0
                    continue
                
                # Measurement
                z = keypoint_sequence[i, k]
                
                if kp_id not in kalman_states:
                    # Initialize state
                    kalman_states[kp_id] = {
                        "pos": z,
                        "vel": np.zeros(2),
                        "cov": np.eye(4) * 100  # High initial uncertainty
                    }
                    smoothed[i, k] = z
                else:
                    # Predict
                    state = kalman_states[kp_id]
                    pred_pos = state["pos"] + state["vel"]
                    pred_vel = state["vel"]
                    
                    # Kalman gain
                    k_gain = state["cov"] / (state["cov"] + measurement_noise)
                    
                    # Update
                    new_pos = pred_pos + k_gain * (z - pred_pos)
                    new_vel = pred_vel
                    
                    if i > 0:
                        # Update velocity based on position change
                        new_vel = new_pos - state["pos"]
                    
                    # Update state
                    kalman_states[kp_id] = {
                        "pos": new_pos,
                        "vel": new_vel,
                        "cov": (1 - k_gain) * state["cov"] + process_noise
                    }
                    
                    # Store smoothed position
                    smoothed[i, k] = new_pos
        
        return smoothed 