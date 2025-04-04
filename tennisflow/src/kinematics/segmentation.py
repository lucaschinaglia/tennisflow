"""
Swing Segmentation Module

This module implements functions to segment tennis swings from sequences of pose keypoints
based on kinematic indicators like wrist velocity and acceleration.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.signal import find_peaks

# Configure logging
logger = logging.getLogger(__name__)

def calculate_velocity(
    keypoint_sequence: np.ndarray,
    keypoint_idx: int = 10,  # Default is right_wrist (index 10)
    fps: float = 30.0,
    smooth_window: int = 5
) -> np.ndarray:
    """
    Calculate velocity of a specific keypoint over time.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        keypoint_idx: Index of the keypoint to track
        fps: Frames per second of the video
        smooth_window: Window size for smoothing velocity
        
    Returns:
        Velocity magnitude over time, shape (frames-1,)
    """
    if keypoint_sequence.shape[0] < 2:
        logger.warning("Sequence too short to calculate velocity")
        return np.zeros(max(1, keypoint_sequence.shape[0] - 1))
        
    # Extract the specified keypoint trajectory
    keypoint_trajectory = keypoint_sequence[:, keypoint_idx, :]
    
    # Calculate displacement between consecutive frames
    displacement = np.diff(keypoint_trajectory, axis=0)
    
    # Calculate Euclidean distance (magnitude of displacement)
    velocity = np.sqrt(np.sum(displacement**2, axis=1))
    
    # Convert to units per second
    velocity = velocity * fps
    
    # Simple moving average smoothing if needed
    if smooth_window > 1 and len(velocity) > smooth_window:
        # Create a padded array
        padded = np.pad(velocity, (smooth_window//2, smooth_window//2), mode='edge')
        # Apply moving average
        smoothed = np.zeros_like(velocity)
        for i in range(len(velocity)):
            smoothed[i] = np.mean(padded[i:i+smooth_window])
        velocity = smoothed
    
    return velocity

def detect_peaks_and_valleys(
    signal: np.ndarray,
    peak_height: Optional[float] = None,
    peak_distance: int = 15,
    peak_prominence: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect peaks and valleys in a signal.
    
    Args:
        signal: Input signal (e.g., wrist velocity)
        peak_height: Minimum height of peaks, None for auto-detection
        peak_distance: Minimum distance between peaks
        peak_prominence: Minimum prominence of peaks
        
    Returns:
        Tuple of (peak_indices, valley_indices)
    """
    # Auto-determine peak height if not specified
    if peak_height is None:
        peak_height = np.mean(signal) + 0.5 * np.std(signal)
    
    # Find peaks
    peaks, _ = find_peaks(
        signal, 
        height=peak_height,
        distance=peak_distance,
        prominence=peak_prominence
    )
    
    # Find valleys (peaks in negative signal)
    neg_signal = -signal
    valleys, _ = find_peaks(
        neg_signal,
        height=-np.percentile(signal, 25),  # Use 25th percentile as threshold
        distance=peak_distance,
        prominence=peak_prominence / 2  # Lower prominence for valleys
    )
    
    return peaks, valleys

def segment_swings(
    keypoint_sequence: np.ndarray,
    fps: float = 30.0,
    min_swing_duration: float = 0.5,  # seconds
    max_swing_duration: float = 2.5,  # seconds
    velocity_threshold: float = 5.0
) -> List[Dict]:
    """
    Segment tennis swings from a sequence of keypoints.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        fps: Frames per second of the video
        min_swing_duration: Minimum duration of a swing in seconds
        max_swing_duration: Maximum duration of a swing in seconds
        velocity_threshold: Velocity threshold for swing detection
        
    Returns:
        List of detected swings, each with start_frame, end_frame, etc.
    """
    # Convert duration to frames
    min_swing_frames = int(min_swing_duration * fps)
    max_swing_frames = int(max_swing_duration * fps)
    
    # Calculate velocity for right and left wrist
    right_wrist_velocity = calculate_velocity(keypoint_sequence, keypoint_idx=10, fps=fps)  # right_wrist
    left_wrist_velocity = calculate_velocity(keypoint_sequence, keypoint_idx=9, fps=fps)   # left_wrist
    
    # Determine dominant hand (assuming right-handed player by default)
    # In a proper implementation, this would be determined by analyzing the entire sequence
    dominant_velocity = right_wrist_velocity
    non_dominant_velocity = left_wrist_velocity
    dominant_hand = "right"
    
    if np.mean(left_wrist_velocity) > np.mean(right_wrist_velocity) * 1.5:
        # If left wrist is consistently moving more, assume left-handed player
        dominant_velocity = left_wrist_velocity
        non_dominant_velocity = right_wrist_velocity
        dominant_hand = "left"
    
    # Detect peaks in dominant wrist velocity
    peaks, valleys = detect_peaks_and_valleys(
        dominant_velocity,
        peak_height=velocity_threshold,
        peak_distance=min_swing_frames // 2,
        peak_prominence=velocity_threshold / 2
    )
    
    # Segment swings using peaks and valleys
    swings = []
    
    for peak_idx in peaks:
        # Find the nearest valley before the peak
        prev_valleys = valleys[valleys < peak_idx]
        start_idx = prev_valleys[-1] if len(prev_valleys) > 0 else max(0, peak_idx - min_swing_frames)
        
        # Find the nearest valley after the peak
        next_valleys = valleys[valleys > peak_idx]
        end_idx = next_valleys[0] if len(next_valleys) > 0 else min(len(dominant_velocity), peak_idx + min_swing_frames)
        
        # Adjust to ensure minimum duration
        if end_idx - start_idx < min_swing_frames:
            padding = (min_swing_frames - (end_idx - start_idx)) // 2
            start_idx = max(0, start_idx - padding)
            end_idx = min(len(dominant_velocity), end_idx + padding)
        
        # Skip if swing is too long
        if end_idx - start_idx > max_swing_frames:
            continue
        
        # Create swing object
        swing = {
            "start_frame": int(start_idx),
            "end_frame": int(end_idx + 1),  # +1 because we lost a frame in velocity calculation
            "peak_frame": int(peak_idx + 1),  # +1 to adjust for velocity calculation
            "duration": (end_idx - start_idx) / fps,
            "peak_velocity": float(dominant_velocity[peak_idx]),
            "dominant_hand": dominant_hand
        }
        
        swings.append(swing)
    
    # Merge overlapping swings
    merged_swings = []
    if swings:
        sorted_swings = sorted(swings, key=lambda x: x["start_frame"])
        current_swing = sorted_swings[0]
        
        for swing in sorted_swings[1:]:
            # Check for overlap
            if swing["start_frame"] <= current_swing["end_frame"]:
                # Merge swings
                current_swing["end_frame"] = max(current_swing["end_frame"], swing["end_frame"])
                current_swing["peak_frame"] = swing["peak_frame"] if swing["peak_velocity"] > current_swing["peak_velocity"] else current_swing["peak_frame"]
                current_swing["peak_velocity"] = max(current_swing["peak_velocity"], swing["peak_velocity"])
                current_swing["duration"] = (current_swing["end_frame"] - current_swing["start_frame"]) / fps
            else:
                # No overlap, add current swing and move to next
                merged_swings.append(current_swing)
                current_swing = swing
        
        # Add the last swing
        merged_swings.append(current_swing)
    
    logger.info(f"Detected {len(merged_swings)} swings")
    return merged_swings

def identify_swing_phases(
    keypoint_sequence: np.ndarray,
    swing: Dict
) -> Dict:
    """
    Identify key phases within a swing.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        swing: Swing dictionary with start_frame, end_frame, etc.
        
    Returns:
        Dictionary with phase information
    """
    # Extract the segment corresponding to this swing
    start_idx = swing["start_frame"]
    end_idx = min(swing["end_frame"], keypoint_sequence.shape[0] - 1)
    
    if end_idx <= start_idx:
        logger.warning("Invalid swing indices")
        return {"phases": {}}
    
    swing_sequence = keypoint_sequence[start_idx:end_idx+1]
    
    # Calculate velocity
    keypoint_idx = 10 if swing.get("dominant_hand", "right") == "right" else 9
    velocity = calculate_velocity(swing_sequence, keypoint_idx=keypoint_idx)
    
    # Determine key points within the swing
    if len(velocity) < 3:
        logger.warning("Swing too short to identify phases")
        return {"phases": {}}
    
    # Find peak velocity
    peak_idx = np.argmax(velocity)
    
    # Phases dictionary
    phases = {}
    
    # Backswing: From start to where velocity starts increasing significantly
    # Find the point where velocity starts to increase toward the peak
    backswing_end = 0
    for i in range(min(peak_idx, len(velocity) - 1)):
        if velocity[i] > 0.3 * velocity[peak_idx]:
            backswing_end = i
            break
    
    # Forward swing: From backswing end to peak velocity
    forward_swing_start = backswing_end
    forward_swing_end = peak_idx
    
    # Follow-through: From peak to end
    follow_through_start = peak_idx
    follow_through_end = len(velocity) - 1
    
    # Store phases relative to the video frames
    phases["preparation"] = {
        "start_frame": start_idx,
        "end_frame": start_idx + backswing_end
    }
    
    phases["backswing"] = {
        "start_frame": start_idx,
        "end_frame": start_idx + backswing_end
    }
    
    phases["forward_swing"] = {
        "start_frame": start_idx + forward_swing_start,
        "end_frame": start_idx + forward_swing_end
    }
    
    # Approximate contact point at or just after peak velocity
    contact_frame = start_idx + min(peak_idx + 1, len(velocity) - 1)
    phases["contact"] = {
        "frame": contact_frame
    }
    
    phases["follow_through"] = {
        "start_frame": start_idx + follow_through_start,
        "end_frame": start_idx + follow_through_end
    }
    
    return {
        "phases": phases,
        "key_frames": {
            "backswing_max": start_idx + backswing_end,
            "contact": contact_frame,
            "follow_through_max": start_idx + follow_through_end
        }
    } 