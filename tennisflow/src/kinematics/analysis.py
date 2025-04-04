"""
Kinematic Analysis Module

This module implements functions to calculate biomechanical metrics from tennis swing
keypoint sequences, such as joint angles, angular velocities, and stability metrics.
"""

import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

def calculate_joint_angle(
    joint1: np.ndarray,
    joint2: np.ndarray,
    joint3: np.ndarray
) -> float:
    """
    Calculate the angle between three joints.
    
    Args:
        joint1: Position of the first joint [x, y]
        joint2: Position of the middle joint [x, y] (vertex of the angle)
        joint3: Position of the third joint [x, y]
        
    Returns:
        Angle in degrees
    """
    # Convert to numpy arrays if they aren't already
    joint1 = np.asarray(joint1)
    joint2 = np.asarray(joint2)
    joint3 = np.asarray(joint3)
    
    # Calculate vectors from the middle joint
    vector1 = joint1 - joint2
    vector2 = joint3 - joint2
    
    # Calculate the angle using the dot product formula
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # Check for zero division
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate angle in degrees, clamping to handle floating-point errors
    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)

def calculate_angular_velocity(
    angles: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Calculate angular velocity from a sequence of angles.
    
    Args:
        angles: Sequence of joint angles
        fps: Frames per second
        
    Returns:
        Angular velocity in degrees per second
    """
    if len(angles) < 2:
        return np.zeros(len(angles))
    
    # Calculate difference between consecutive angles
    # Handle angle wrapping (e.g., 359° to 1° should be a small change)
    angular_diff = np.zeros(len(angles) - 1)
    for i in range(len(angles) - 1):
        diff = angles[i+1] - angles[i]
        # Handle angle wrapping
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        angular_diff[i] = diff
    
    # Convert to degrees per second
    angular_velocity = angular_diff * fps
    
    # Add zero at the beginning to maintain array length
    return np.concatenate(([0], angular_velocity))

def calculate_hip_shoulder_separation(
    keypoint_sequence: np.ndarray,
    frame_idx: int
) -> float:
    """
    Calculate hip-shoulder separation angle.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        frame_idx: Index of the frame to analyze
        
    Returns:
        Hip-shoulder separation angle in degrees
    """
    if frame_idx >= keypoint_sequence.shape[0]:
        logger.warning(f"Frame index {frame_idx} out of bounds")
        return 0.0
    
    # Extract shoulder and hip positions
    # MoveNet keypoint indices: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    left_shoulder = keypoint_sequence[frame_idx, 5]
    right_shoulder = keypoint_sequence[frame_idx, 6]
    left_hip = keypoint_sequence[frame_idx, 11]
    right_hip = keypoint_sequence[frame_idx, 12]
    
    # Check if any keypoints are missing
    if np.any(np.isnan(left_shoulder)) or np.any(np.isnan(right_shoulder)) or \
       np.any(np.isnan(left_hip)) or np.any(np.isnan(right_hip)):
        logger.warning("Missing keypoints for hip-shoulder separation calculation")
        return 0.0
    
    # Calculate shoulder and hip angles relative to horizontal
    shoulder_vector = right_shoulder - left_shoulder
    hip_vector = right_hip - left_hip
    
    # Calculate angle with respect to horizontal (x-axis)
    shoulder_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
    hip_angle = np.degrees(np.arctan2(hip_vector[1], hip_vector[0]))
    
    # Calculate separation angle (difference between angles)
    separation = hip_angle - shoulder_angle
    
    # Normalize to -180 to 180 range
    if separation > 180:
        separation -= 360
    elif separation < -180:
        separation += 360
    
    return float(abs(separation))

def calculate_elbow_angle(
    keypoint_sequence: np.ndarray,
    frame_idx: int,
    arm: str = "right"
) -> float:
    """
    Calculate elbow angle (shoulder-elbow-wrist).
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        frame_idx: Index of the frame to analyze
        arm: Which arm to analyze ("right" or "left")
        
    Returns:
        Elbow angle in degrees
    """
    if frame_idx >= keypoint_sequence.shape[0]:
        logger.warning(f"Frame index {frame_idx} out of bounds")
        return 0.0
    
    # Define keypoint indices based on arm
    if arm == "right":
        # MoveNet keypoint indices: 6=right_shoulder, 8=right_elbow, 10=right_wrist
        shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10
    else:
        # MoveNet keypoint indices: 5=left_shoulder, 7=left_elbow, 9=left_wrist
        shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9
    
    # Extract joint positions
    shoulder = keypoint_sequence[frame_idx, shoulder_idx]
    elbow = keypoint_sequence[frame_idx, elbow_idx]
    wrist = keypoint_sequence[frame_idx, wrist_idx]
    
    # Check if any keypoints are missing
    if np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)):
        logger.warning(f"Missing keypoints for {arm} elbow angle calculation")
        return 0.0
    
    # Calculate angle
    return calculate_joint_angle(shoulder, elbow, wrist)

def calculate_knee_flexion(
    keypoint_sequence: np.ndarray,
    frame_idx: int,
    leg: str = "right"
) -> float:
    """
    Calculate knee flexion angle (hip-knee-ankle).
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        frame_idx: Index of the frame to analyze
        leg: Which leg to analyze ("right" or "left")
        
    Returns:
        Knee flexion angle in degrees
    """
    if frame_idx >= keypoint_sequence.shape[0]:
        logger.warning(f"Frame index {frame_idx} out of bounds")
        return 0.0
    
    # Define keypoint indices based on leg
    if leg == "right":
        # MoveNet keypoint indices: 12=right_hip, 14=right_knee, 16=right_ankle
        hip_idx, knee_idx, ankle_idx = 12, 14, 16
    else:
        # MoveNet keypoint indices: 11=left_hip, 13=left_knee, 15=left_ankle
        hip_idx, knee_idx, ankle_idx = 11, 13, 15
    
    # Extract joint positions
    hip = keypoint_sequence[frame_idx, hip_idx]
    knee = keypoint_sequence[frame_idx, knee_idx]
    ankle = keypoint_sequence[frame_idx, ankle_idx]
    
    # Check if any keypoints are missing
    if np.any(np.isnan(hip)) or np.any(np.isnan(knee)) or np.any(np.isnan(ankle)):
        logger.warning(f"Missing keypoints for {leg} knee flexion calculation")
        return 0.0
    
    # Calculate angle
    return calculate_joint_angle(hip, knee, ankle)

def calculate_wrist_speed(
    keypoint_sequence: np.ndarray,
    swing: Dict,
    fps: float = 30.0,
    arm: str = "right"
) -> float:
    """
    Calculate peak wrist speed during a swing.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        swing: Swing dictionary with start_frame, end_frame, etc.
        fps: Frames per second
        arm: Which arm to analyze ("right" or "left")
        
    Returns:
        Peak wrist speed in pixels per second
    """
    # Extract the swing sequence
    start_idx = swing["start_frame"]
    end_idx = min(swing["end_frame"], keypoint_sequence.shape[0] - 1)
    
    if end_idx <= start_idx:
        logger.warning("Invalid swing indices")
        return 0.0
    
    swing_sequence = keypoint_sequence[start_idx:end_idx+1]
    
    # Determine wrist keypoint index
    wrist_idx = 10 if arm == "right" else 9  # MoveNet: 10=right_wrist, 9=left_wrist
    
    # Calculate wrist velocity
    wrist_velocity = np.zeros(swing_sequence.shape[0] - 1)
    
    for i in range(len(wrist_velocity)):
        # Extract wrist positions in consecutive frames
        pos1 = swing_sequence[i, wrist_idx]
        pos2 = swing_sequence[i+1, wrist_idx]
        
        # Calculate displacement
        displacement = np.linalg.norm(pos2 - pos1)
        
        # Convert to units per second
        wrist_velocity[i] = displacement * fps
    
    # Return peak velocity
    if len(wrist_velocity) > 0:
        return float(np.max(wrist_velocity))
    else:
        return 0.0

def calculate_head_stability(
    keypoint_sequence: np.ndarray,
    swing: Dict
) -> float:
    """
    Calculate head stability during a swing.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        swing: Swing dictionary with start_frame, end_frame, etc.
        
    Returns:
        Head stability metric (lower values indicate greater stability)
    """
    # Extract the swing sequence
    start_idx = swing["start_frame"]
    end_idx = min(swing["end_frame"], keypoint_sequence.shape[0] - 1)
    
    if end_idx <= start_idx:
        logger.warning("Invalid swing indices")
        return 100.0  # High instability
    
    swing_sequence = keypoint_sequence[start_idx:end_idx+1]
    
    # Extract nose keypoint trajectory (MoveNet: 0=nose)
    nose_idx = 0
    nose_trajectory = swing_sequence[:, nose_idx, :]
    
    # Check if we have enough valid keypoints
    if np.all(np.isnan(nose_trajectory)):
        logger.warning("No valid nose keypoints for head stability calculation")
        return 100.0
    
    # Calculate variance of nose position
    valid_indices = ~np.any(np.isnan(nose_trajectory), axis=1)
    valid_nose = nose_trajectory[valid_indices]
    
    if len(valid_nose) < 3:
        logger.warning("Not enough valid nose keypoints for head stability calculation")
        return 100.0
    
    # Calculate variance in x and y directions
    var_x = np.var(valid_nose[:, 0])
    var_y = np.var(valid_nose[:, 1])
    
    # Combined variance (diagonals of covariance matrix)
    stability_metric = np.sqrt(var_x + var_y)
    
    # Normalize to 0-100 range (lower is more stable)
    # Scale factor can be adjusted based on expected range
    normalized_stability = min(100.0, stability_metric / 5.0 * 100.0)
    
    return float(normalized_stability)

def calculate_weight_transfer(
    keypoint_sequence: np.ndarray,
    swing: Dict
) -> float:
    """
    Calculate weight transfer during a swing.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        swing: Swing dictionary with start_frame, end_frame, etc.
        
    Returns:
        Weight transfer metric (0-100)
    """
    # Extract the swing sequence
    start_idx = swing["start_frame"]
    end_idx = min(swing["end_frame"], keypoint_sequence.shape[0] - 1)
    
    if end_idx <= start_idx:
        logger.warning("Invalid swing indices")
        return 0.0
    
    swing_sequence = keypoint_sequence[start_idx:end_idx+1]
    
    # Use hips as proxy for center of mass (average of left and right hip)
    left_hip_idx, right_hip_idx = 11, 12  # MoveNet: 11=left_hip, 12=right_hip
    
    # Calculate center of mass horizontal movement
    com_x = np.zeros(swing_sequence.shape[0])
    valid_frames = 0
    
    for i in range(len(com_x)):
        left_hip = swing_sequence[i, left_hip_idx]
        right_hip = swing_sequence[i, right_hip_idx]
        
        if not (np.any(np.isnan(left_hip)) or np.any(np.isnan(right_hip))):
            com_x[i] = (left_hip[0] + right_hip[0]) / 2
            valid_frames += 1
        else:
            com_x[i] = np.nan
    
    if valid_frames < 2:
        logger.warning("Not enough valid frames for weight transfer calculation")
        return 0.0
    
    # Filter out NaN values
    valid_com_x = com_x[~np.isnan(com_x)]
    
    # Calculate horizontal displacement
    displacement = np.max(valid_com_x) - np.min(valid_com_x)
    
    # Normalize to 0-100 range based on reasonable displacement range
    # Scale factor can be adjusted based on expected range
    max_expected_displacement = 100  # pixels
    normalized_transfer = min(100.0, displacement / max_expected_displacement * 100.0)
    
    return float(normalized_transfer)

def calculate_metrics(
    keypoint_sequence: np.ndarray,
    swing: Dict,
    swing_phases: Dict,
    fps: float = 30.0
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a tennis swing.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        swing: Swing dictionary with start_frame, end_frame, etc.
        swing_phases: Dictionary with phase information
        fps: Frames per second
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # Determine dominant arm (from swing info)
    dominant_arm = swing.get("dominant_hand", "right")
    
    try:
        # Get key frame indices for analysis
        backswing_frame = swing_phases.get("key_frames", {}).get("backswing_max", swing["start_frame"])
        contact_frame = swing_phases.get("key_frames", {}).get("contact", swing["peak_frame"])
        follow_through_frame = swing_phases.get("key_frames", {}).get("follow_through_max", swing["end_frame"])
        
        # 1. Wrist Speed
        metrics["racketSpeed"] = calculate_wrist_speed(
            keypoint_sequence, swing, fps, arm=dominant_arm
        )
        
        # 2. Elbow Angle at contact
        metrics["elbowAngle"] = calculate_elbow_angle(
            keypoint_sequence, contact_frame, arm=dominant_arm
        )
        
        # 3. Hip-Shoulder Separation at backswing
        metrics["hipShoulder"] = calculate_hip_shoulder_separation(
            keypoint_sequence, backswing_frame
        )
        
        # 4. Knee Flexion at contact
        metrics["kneeFlexion"] = calculate_knee_flexion(
            keypoint_sequence, contact_frame, leg=dominant_arm  # Assume same side as arm
        )
        
        # 5. Weight Transfer
        metrics["weightTransfer"] = calculate_weight_transfer(
            keypoint_sequence, swing
        )
        
        # 6. Head Stability
        metrics["headStability"] = 100 - calculate_head_stability(
            keypoint_sequence, swing
        )  # Invert so higher is better
        
        # Normalize any values that exceed 100 to better fit the UI display
        for key in metrics:
            if metrics[key] > 100:
                metrics[key] = 100.0
            
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Provide default metrics
        metrics = {
            "racketSpeed": 0.0,
            "elbowAngle": 0.0,
            "hipShoulder": 0.0,
            "kneeFlexion": 0.0,
            "weightTransfer": 0.0,
            "headStability": 0.0
        }
    
    return metrics 