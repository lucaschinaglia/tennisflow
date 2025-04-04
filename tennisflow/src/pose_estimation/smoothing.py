"""
Temporal Smoothing Module

This module provides functions to apply temporal smoothing to keypoint sequences,
reducing jitter and noise in pose estimations across video frames.
"""

import numpy as np
from scipy import signal
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

def savgol_filter_keypoints(
    keypoint_sequence: np.ndarray,
    window_size: int = 15,
    polynomial_order: int = 3,
    axis: int = 0
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth keypoint trajectories.
    
    Args:
        keypoint_sequence: Sequence of keypoints of shape (frames, keypoints, 2) or (frames, keypoints*2)
        window_size: Size of the smoothing window (must be odd number)
        polynomial_order: Order of the polynomial used for fitting
        axis: Axis along which to smooth (0 for frames)
        
    Returns:
        Smoothed keypoint sequence
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Ensure window size is less than sequence length
    num_frames = keypoint_sequence.shape[0]
    if window_size > num_frames:
        window_size = min(num_frames - (num_frames % 2 == 0), 3)  # Ensure it's odd and at least 3
        polynomial_order = min(polynomial_order, window_size - 1)
        logger.warning(f"Adjusted window_size to {window_size} and polynomial_order to {polynomial_order} due to short sequence")
    
    # Ensure polynomial order is valid
    if polynomial_order >= window_size:
        polynomial_order = window_size - 1
    
    try:
        # Reshape to 2D if needed for filtering
        original_shape = keypoint_sequence.shape
        if len(original_shape) == 3:  # (frames, keypoints, 2)
            num_frames, num_keypoints, num_coords = original_shape
            reshaped = keypoint_sequence.reshape(num_frames, -1)  # (frames, keypoints*2)
        else:
            reshaped = keypoint_sequence
            
        # Apply Savitzky-Golay filter to each coordinate
        smoothed = np.zeros_like(reshaped)
        
        for i in range(reshaped.shape[1]):
            # Get trajectory for this coordinate
            trajectory = reshaped[:, i]
            
            # Only filter if we have enough valid points (non-zero)
            valid_indices = np.where(trajectory != 0)[0]
            if len(valid_indices) > window_size:
                # Extract valid trajectory
                valid_trajectory = trajectory[valid_indices]
                
                # Apply filter
                smoothed_valid = signal.savgol_filter(
                    valid_trajectory, 
                    window_size, 
                    polynomial_order
                )
                
                # Put back smoothed values
                trajectory_smoothed = np.copy(trajectory)
                trajectory_smoothed[valid_indices] = smoothed_valid
                smoothed[:, i] = trajectory_smoothed
            else:
                # Not enough points for filtering
                smoothed[:, i] = trajectory
        
        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            smoothed = smoothed.reshape(original_shape)
            
        return smoothed
        
    except Exception as e:
        logger.error(f"Error in Savitzky-Golay filtering: {e}")
        return keypoint_sequence


class KalmanFilter:
    """
    Kalman filter implementation for smoothing keypoint trajectories.
    
    This implements a simple Kalman filter for each keypoint coordinate,
    treating position and velocity as the state variables.
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise variance (Q)
            measurement_noise: Measurement noise variance (R)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.states = {}  # Dict to store state for each keypoint
        
    def _initialize_state(self, keypoint_id: str):
        """
        Initialize state for a new keypoint.
        
        Args:
            keypoint_id: Unique identifier for the keypoint
        """
        # State: [x, y, vx, vy]
        self.states[keypoint_id] = {
            'x': np.zeros(4),           # State
            'P': np.eye(4) * 100,       # Covariance matrix (high uncertainty initially)
            'F': np.array([             # State transition matrix
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            'H': np.array([             # Measurement matrix
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]),
            'Q': np.eye(4) * self.process_noise,  # Process noise
            'R': np.eye(2) * self.measurement_noise,  # Measurement noise
        }
    
    def update(self, keypoint_id: str, measurement: np.ndarray) -> np.ndarray:
        """
        Update the filter with a new measurement.
        
        Args:
            keypoint_id: Unique identifier for the keypoint
            measurement: Observed position [x, y]
            
        Returns:
            Filtered state [x, y]
        """
        # Initialize if this is a new keypoint
        if keypoint_id not in self.states:
            self._initialize_state(keypoint_id)
            # Initialize state with first measurement
            self.states[keypoint_id]['x'][0] = measurement[0]
            self.states[keypoint_id]['x'][1] = measurement[1]
            return measurement
        
        state = self.states[keypoint_id]
        
        # Predict
        x_pred = state['F'] @ state['x']
        P_pred = state['F'] @ state['P'] @ state['F'].T + state['Q']
        
        # Update only if measurement is valid (non-zero)
        if not np.all(measurement == 0):
            # Calculate Kalman gain
            K = P_pred @ state['H'].T @ np.linalg.inv(state['H'] @ P_pred @ state['H'].T + state['R'])
            
            # Update state
            y = measurement - state['H'] @ x_pred
            x_new = x_pred + K @ y
            P_new = (np.eye(4) - K @ state['H']) @ P_pred
            
            # Store updated state
            state['x'] = x_new
            state['P'] = P_new
        else:
            # If no measurement, just use prediction
            state['x'] = x_pred
            state['P'] = P_pred
        
        # Return filtered position
        return state['x'][:2]


def kalman_filter_keypoints(
    keypoint_sequence: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1
) -> np.ndarray:
    """
    Apply Kalman filtering to smooth keypoint trajectories.
    
    Args:
        keypoint_sequence: Sequence of keypoints of shape (frames, keypoints, 2)
        process_noise: Process noise variance (Q)
        measurement_noise: Measurement noise variance (R)
        
    Returns:
        Smoothed keypoint sequence
    """
    try:
        # Create a Kalman filter
        kf = KalmanFilter(process_noise, measurement_noise)
        
        # Get sequence dimensions
        num_frames, num_keypoints, num_coords = keypoint_sequence.shape
        
        # Initialize output array
        smoothed = np.zeros_like(keypoint_sequence)
        
        # Process each keypoint
        for kp_idx in range(num_keypoints):
            for frame_idx in range(num_frames):
                # Get measurement
                measurement = keypoint_sequence[frame_idx, kp_idx]
                
                # Update filter
                filtered_pos = kf.update(f"kp_{kp_idx}", measurement)
                
                # Store filtered position
                smoothed[frame_idx, kp_idx] = filtered_pos
                
        return smoothed
        
    except Exception as e:
        logger.error(f"Error in Kalman filtering: {e}")
        return keypoint_sequence


def smooth_keypoint_sequence(
    keypoint_sequence: np.ndarray,
    method: str = "savgol",
    **kwargs
) -> np.ndarray:
    """
    Smooth a sequence of keypoints using the specified method.
    
    Args:
        keypoint_sequence: Sequence of keypoints
        method: Smoothing method ('savgol' or 'kalman')
        **kwargs: Additional parameters for the smoothing method
        
    Returns:
        Smoothed keypoint sequence
    """
    if method == "savgol":
        # Default parameters for Savitzky-Golay filter
        window_size = kwargs.get("window_size", 15)
        polynomial_order = kwargs.get("polynomial_order", 3)
        return savgol_filter_keypoints(
            keypoint_sequence,
            window_size=window_size,
            polynomial_order=polynomial_order
        )
    elif method == "kalman":
        # Default parameters for Kalman filter
        process_noise = kwargs.get("process_noise", 0.01)
        measurement_noise = kwargs.get("measurement_noise", 0.1)
        return kalman_filter_keypoints(
            keypoint_sequence,
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
    else:
        logger.warning(f"Unknown smoothing method: {method}, using Savitzky-Golay as default")
        return savgol_filter_keypoints(keypoint_sequence) 