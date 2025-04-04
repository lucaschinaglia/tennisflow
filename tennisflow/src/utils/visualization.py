"""
Visualization Module

This module provides functions to visualize pose keypoints, detected swings,
and analysis results for tennis videos.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import os

# Configure logging
logger = logging.getLogger(__name__)

def draw_keypoints(
    frame: np.ndarray,
    keypoints: List[Dict],
    connections: Optional[List[Dict]] = None,
    confidence_threshold: float = 0.3,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (0, 255, 255),
    point_size: int = 5,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw keypoints and connections on a frame.
    
    Args:
        frame: Input frame
        keypoints: List of keypoint dictionaries
        connections: List of connection dictionaries
        confidence_threshold: Minimum confidence to draw a keypoint
        point_color: Color for keypoints (BGR format)
        connection_color: Color for connections (BGR format)
        point_size: Size of keypoint circles
        line_thickness: Thickness of connection lines
        
    Returns:
        Frame with keypoints and connections drawn
    """
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # Create a dictionary for keypoint lookup
    keypoint_dict = {}
    for kp in keypoints:
        if kp.get("confidence", 0) >= confidence_threshold:
            keypoint_dict[kp["name"]] = (
                int(kp["position"]["x"]),
                int(kp["position"]["y"])
            )
    
    # Draw connections
    if connections:
        for conn in connections:
            from_name = conn.get("from", "")
            to_name = conn.get("to", "")
            
            if from_name in keypoint_dict and to_name in keypoint_dict:
                from_pos = keypoint_dict[from_name]
                to_pos = keypoint_dict[to_name]
                
                cv2.line(vis_frame, from_pos, to_pos, connection_color, line_thickness)
    
    # Draw keypoints
    for name, pos in keypoint_dict.items():
        cv2.circle(vis_frame, pos, point_size, point_color, -1)
    
    return vis_frame

def draw_swing_phase(
    frame: np.ndarray,
    phase: str,
    metrics: Optional[Dict] = None,
    position: Tuple[int, int] = (50, 50),
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw swing phase and metrics on a frame.
    
    Args:
        frame: Input frame
        phase: Current swing phase
        metrics: Dictionary of metrics to display
        position: Position for text
        font_scale: Font scale for text
        color: Text color (BGR format)
        background_color: Background color (BGR format)
        thickness: Text thickness
        
    Returns:
        Frame with swing phase and metrics drawn
    """
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # Map phase to display text
    phase_map = {
        "preparation": "PREPARATION",
        "backswing": "BACKSWING",
        "forward_swing": "FORWARD SWING",
        "contact": "CONTACT",
        "follow_through": "FOLLOW THROUGH"
    }
    
    # Get display text
    phase_text = phase_map.get(phase, phase.upper())
    
    # Draw background rectangle for phase text
    text_size = cv2.getTextSize(
        phase_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )[0]
    cv2.rectangle(
        vis_frame,
        (position[0] - 10, position[1] - text_size[1] - 10),
        (position[0] + text_size[0] + 10, position[1] + 10),
        background_color,
        -1
    )
    
    # Draw phase text
    cv2.putText(
        vis_frame,
        phase_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    # Draw metrics if provided
    if metrics:
        y_offset = position[1] + 40
        for label, value in metrics.items():
            metric_text = f"{label}: {value:.1f}"
            
            # Draw background rectangle for metric text
            text_size = cv2.getTextSize(
                metric_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness
            )[0]
            cv2.rectangle(
                vis_frame,
                (position[0] - 10, y_offset - text_size[1] - 5),
                (position[0] + text_size[0] + 10, y_offset + 5),
                (50, 50, 50),
                -1
            )
            
            # Draw metric text
            cv2.putText(
                vis_frame,
                metric_text,
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.8,
                color,
                thickness - 1
            )
            
            y_offset += 30
    
    return vis_frame

def create_swing_summary_image(
    frame: np.ndarray,
    swing_type: str,
    metrics: Dict,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a summary image for a swing, including metrics visualization.
    
    Args:
        frame: Representative frame from the swing
        swing_type: Type of swing (e.g., "forehand", "backhand")
        metrics: Dictionary of metrics
        save_path: Path to save the image (optional)
        
    Returns:
        Summary image
    """
    # Create a larger canvas with room for the metrics
    height, width = frame.shape[:2]
    metrics_height = 300  # Height of metrics section
    canvas = np.zeros((height + metrics_height, width, 3), dtype=np.uint8)
    
    # Copy the frame to the top part
    canvas[:height, :width] = frame
    
    # Fill the bottom part with a darker background
    canvas[height:, :] = (50, 50, 50)
    
    # Draw a title for the swing
    title = f"{swing_type.upper()} SWING ANALYSIS"
    cv2.putText(
        canvas,
        title,
        (width // 2 - 200, height + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )
    
    # Draw metrics as bars
    y_position = height + 80
    bar_height = 25
    max_bar_width = width - 200
    
    for i, (label, value) in enumerate(metrics.items()):
        # Normalize to 0-100 if not already
        norm_value = min(100, max(0, value))
        
        # Draw label
        cv2.putText(
            canvas,
            label,
            (50, y_position + bar_height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        
        # Draw background bar
        cv2.rectangle(
            canvas,
            (150, y_position),
            (150 + max_bar_width, y_position + bar_height),
            (100, 100, 100),
            -1
        )
        
        # Draw value bar
        bar_width = int(max_bar_width * norm_value / 100)
        
        # Color based on value
        if norm_value < 33:
            color = (50, 50, 200)  # Red (BGR)
        elif norm_value < 66:
            color = (50, 200, 200)  # Yellow (BGR)
        else:
            color = (50, 200, 50)   # Green (BGR)
            
        cv2.rectangle(
            canvas,
            (150, y_position),
            (150 + bar_width, y_position + bar_height),
            color,
            -1
        )
        
        # Draw value text
        cv2.putText(
            canvas,
            f"{norm_value:.1f}",
            (150 + max_bar_width + 10, y_position + bar_height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        
        y_position += bar_height + 10
    
    # Save the image if a path is provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, canvas)
            logger.info(f"Saved swing summary to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save swing summary: {e}")
    
    return canvas

def plot_keypoint_trajectory(
    keypoint_sequence: np.ndarray,
    keypoint_idx: int = 10,  # Default: right_wrist
    title: str = "Keypoint Trajectory",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the trajectory of a keypoint over time.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        keypoint_idx: Index of the keypoint to plot
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Extract trajectory for the specified keypoint
    trajectory = keypoint_sequence[:, keypoint_idx, :]
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trajectory
    ax.plot(x_coords, y_coords, 'b-', linewidth=2)
    ax.plot(x_coords, y_coords, 'ro', markersize=4)
    
    # Add start and end markers
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'mo', markersize=8, label='End')
    
    # Add direction arrow (at 75% of trajectory)
    idx = int(len(x_coords) * 0.75)
    if idx > 0 and idx < len(x_coords) - 1:
        dx = x_coords[idx+1] - x_coords[idx-1]
        dy = y_coords[idx+1] - y_coords[idx-1]
        arr_len = np.sqrt(dx**2 + dy**2)
        if arr_len > 0:
            dx, dy = dx / arr_len * 20, dy / arr_len * 20  # Scale arrow
            ax.arrow(x_coords[idx], y_coords[idx], dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.legend()
    
    # Invert y-axis to match image coordinates
    ax.invert_yaxis()
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure if a path is provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save trajectory plot: {e}")
    
    return fig

def plot_velocity_profile(
    keypoint_sequence: np.ndarray,
    keypoint_idx: int = 10,  # Default: right_wrist
    fps: float = 30.0,
    title: str = "Velocity Profile",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the velocity profile of a keypoint over time.
    
    Args:
        keypoint_sequence: Sequence of keypoints, shape (frames, keypoints, 2)
        keypoint_idx: Index of the keypoint to plot
        fps: Frames per second
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Extract trajectory for the specified keypoint
    trajectory = keypoint_sequence[:, keypoint_idx, :]
    
    # Calculate displacement between consecutive frames
    displacement = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
    
    # Convert to velocity in units/second
    velocity = displacement * fps
    
    # Create time points (starting from 0)
    time_points = np.arange(len(velocity)) / fps
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot velocity profile
    ax.plot(time_points, velocity, 'b-', linewidth=2)
    
    # Mark peak velocity
    peak_idx = np.argmax(velocity)
    ax.plot(time_points[peak_idx], velocity[peak_idx], 'ro', markersize=8, 
            label=f'Peak: {velocity[peak_idx]:.1f} px/s')
    
    # Add horizontal line for average velocity
    avg_velocity = np.mean(velocity)
    ax.axhline(y=avg_velocity, color='g', linestyle='--', 
               label=f'Avg: {avg_velocity:.1f} px/s')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (px/s)')
    ax.set_title(title)
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure if a path is provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved velocity profile to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save velocity profile: {e}")
    
    return fig

def visualize_results(
    output_dir: str,
    frame: np.ndarray,
    swing_type: str,
    metrics: Dict,
    keypoint_sequence: Optional[np.ndarray] = None
) -> Dict[str, str]:
    """
    Generate and save visualization results for a swing.
    
    Args:
        output_dir: Directory to save visualizations
        frame: Representative frame from the swing
        swing_type: Type of swing
        metrics: Dictionary of metrics
        keypoint_sequence: Sequence of keypoints (optional)
        
    Returns:
        Dictionary of paths to saved visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    result_paths = {}
    
    # Create and save swing summary
    summary_path = os.path.join(output_dir, f"{swing_type}_summary.jpg")
    create_swing_summary_image(frame, swing_type, metrics, summary_path)
    result_paths["summary"] = summary_path
    
    # If keypoint sequence is provided, create additional visualizations
    if keypoint_sequence is not None:
        # Plot and save trajectory
        trajectory_path = os.path.join(output_dir, f"{swing_type}_trajectory.png")
        plot_keypoint_trajectory(
            keypoint_sequence,
            title=f"{swing_type.title()} Wrist Trajectory",
            save_path=trajectory_path
        )
        result_paths["trajectory"] = trajectory_path
        
        # Plot and save velocity profile
        velocity_path = os.path.join(output_dir, f"{swing_type}_velocity.png")
        plot_velocity_profile(
            keypoint_sequence,
            title=f"{swing_type.title()} Wrist Velocity",
            save_path=velocity_path
        )
        result_paths["velocity"] = velocity_path
    
    return result_paths 