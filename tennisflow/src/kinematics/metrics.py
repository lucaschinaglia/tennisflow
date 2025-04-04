"""
This module calculates tennis-specific kinematic metrics from pose keypoint sequences.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class KinematicMetrics:
    """
    Calculate tennis-specific kinematic metrics from pose keypoint sequences.
    """
    
    # Keypoint indices based on MoveNet output
    KEYPOINT_DICT = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }
    
    def __init__(self, handedness: str = 'right'):
        """
        Initialize the kinematic metrics calculator.
        
        Args:
            handedness: Player's dominant hand ('right' or 'left')
        """
        self.handedness = handedness.lower()
        logger.info(f"Initialized kinematic metrics calculator for {handedness}-handed player")
        
        # Set dominant and non-dominant side keypoints
        if self.handedness == 'right':
            self.dom_shoulder = 'right_shoulder'
            self.dom_elbow = 'right_elbow'
            self.dom_wrist = 'right_wrist'
            self.dom_hip = 'right_hip'
            self.dom_knee = 'right_knee'
            self.dom_ankle = 'right_ankle'
            
            self.nondom_shoulder = 'left_shoulder'
            self.nondom_elbow = 'left_elbow'
            self.nondom_wrist = 'left_wrist'
            self.nondom_hip = 'left_hip'
            self.nondom_knee = 'left_knee'
            self.nondom_ankle = 'left_ankle'
        else:
            self.dom_shoulder = 'left_shoulder'
            self.dom_elbow = 'left_elbow'
            self.dom_wrist = 'left_wrist'
            self.dom_hip = 'left_hip'
            self.dom_knee = 'left_knee'
            self.dom_ankle = 'left_ankle'
            
            self.nondom_shoulder = 'right_shoulder'
            self.nondom_elbow = 'right_elbow'
            self.nondom_wrist = 'right_wrist'
            self.nondom_hip = 'right_hip'
            self.nondom_knee = 'right_knee'
            self.nondom_ankle = 'right_ankle'
    
    def calculate_metrics(self, keypoints: np.ndarray, fps: float = 30.0) -> Dict[str, Union[float, List[float]]]:
        """
        Calculate all kinematic metrics for a sequence of keypoints.
        
        Args:
            keypoints: Array of shape (sequence_length, num_keypoints, 2) containing keypoint coordinates
            fps: Frames per second of the video
        
        Returns:
            Dictionary of kinematic metrics
        """
        # Validate input
        if len(keypoints.shape) != 3 or keypoints.shape[1] != 17 or keypoints.shape[2] != 2:
            raise ValueError(f"Expected keypoints shape (seq_len, 17, 2), got {keypoints.shape}")
        
        metrics = {}
        
        # Calculate joint angles throughout the sequence
        metrics['elbow_angle'] = self.calculate_elbow_angle(keypoints)
        metrics['shoulder_angle'] = self.calculate_shoulder_angle(keypoints)
        metrics['knee_angle'] = self.calculate_knee_angle(keypoints)
        metrics['hip_rotation'] = self.calculate_hip_rotation(keypoints)
        
        # Find peak values and their timing
        metrics['max_elbow_angle'] = np.max(metrics['elbow_angle'])
        metrics['max_shoulder_angle'] = np.max(metrics['shoulder_angle'])
        metrics['max_knee_angle'] = np.max(metrics['knee_angle'])
        metrics['max_hip_rotation'] = np.max(metrics['hip_rotation'])
        
        # Calculate velocities
        dt = 1.0 / fps  # Time step
        metrics['wrist_velocity'] = self.calculate_velocity(keypoints[:, self.KEYPOINT_DICT[self.dom_wrist]], dt)
        metrics['racket_head_speed'] = np.max(metrics['wrist_velocity']) if len(metrics['wrist_velocity']) > 0 else 0
        
        # Calculate weight transfer
        metrics['weight_transfer'] = self.calculate_weight_transfer(keypoints)
        
        # Calculate kinetic chain metrics
        metrics['kinetic_chain_score'] = self.calculate_kinetic_chain_score(keypoints, fps)
        
        return metrics
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate the angle between three points in degrees.
        
        Args:
            point1, point2, point3: Points in [x, y] format, where point2 is the vertex
        
        Returns:
            Angle in degrees between the three points
        """
        # Convert to vectors
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate angle in radians and convert to degrees
        cos_angle = dot_product / (magnitude1 * magnitude2)
        # Clamp to [-1, 1] to handle floating point errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_elbow_angle(self, keypoints: np.ndarray) -> List[float]:
        """
        Calculate the elbow angle across a sequence of frames.
        
        Args:
            keypoints: Array of shape (sequence_length, num_keypoints, 2)
        
        Returns:
            List of elbow angles for each frame
        """
        angles = []
        
        for frame in keypoints:
            shoulder = frame[self.KEYPOINT_DICT[self.dom_shoulder]]
            elbow = frame[self.KEYPOINT_DICT[self.dom_elbow]]
            wrist = frame[self.KEYPOINT_DICT[self.dom_wrist]]
            
            # Check if any keypoint is missing
            if np.any(shoulder == 0) or np.any(elbow == 0) or np.any(wrist == 0):
                angles.append(0.0)  # Missing data
            else:
                angle = self.calculate_angle(shoulder, elbow, wrist)
                angles.append(angle)
        
        return angles
    
    def calculate_shoulder_angle(self, keypoints: np.ndarray) -> List[float]:
        """
        Calculate the shoulder angle across a sequence of frames.
        
        Args:
            keypoints: Array of shape (sequence_length, num_keypoints, 2)
        
        Returns:
            List of shoulder angles for each frame
        """
        angles = []
        
        for frame in keypoints:
            hip = frame[self.KEYPOINT_DICT[self.dom_hip]]
            shoulder = frame[self.KEYPOINT_DICT[self.dom_shoulder]]
            elbow = frame[self.KEYPOINT_DICT[self.dom_elbow]]
            
            # Check if any keypoint is missing
            if np.any(hip == 0) or np.any(shoulder == 0) or np.any(elbow == 0):
                angles.append(0.0)  # Missing data
            else:
                angle = self.calculate_angle(hip, shoulder, elbow)
                angles.append(angle)
        
        return angles
    
    def calculate_knee_angle(self, keypoints: np.ndarray) -> List[float]:
        """
        Calculate the knee angle across a sequence of frames.
        
        Args:
            keypoints: Array of shape (sequence_length, num_keypoints, 2)
        
        Returns:
            List of knee angles for each frame
        """
        angles = []
        
        for frame in keypoints:
            hip = frame[self.KEYPOINT_DICT[self.dom_hip]]
            knee = frame[self.KEYPOINT_DICT[self.dom_knee]]
            ankle = frame[self.KEYPOINT_DICT[self.dom_ankle]]
            
            # Check if any keypoint is missing
            if np.any(hip == 0) or np.any(knee == 0) or np.any(ankle == 0):
                angles.append(0.0)  # Missing data
            else:
                angle = self.calculate_angle(hip, knee, ankle)
                angles.append(angle)
        
        return angles
    
    def calculate_hip_rotation(self, keypoints: np.ndarray) -> List[float]:
        """
        Calculate hip rotation across a sequence of frames.
        
        Args:
            keypoints: Array of shape (sequence_length, num_keypoints, 2)
        
        Returns:
            List of hip rotation values for each frame
        """
        rotation_values = []
        
        for frame in keypoints:
            left_hip = frame[self.KEYPOINT_DICT['left_hip']]
            right_hip = frame[self.KEYPOINT_DICT['right_hip']]
            
            # Check if hip keypoints are missing
            if np.any(left_hip == 0) or np.any(right_hip == 0):
                rotation_values.append(0.0)  # Missing data
            else:
                # Calculate hip vector angle with respect to horizontal
                hip_vector = right_hip - left_hip
                horizontal = np.array([1.0, 0.0])
                
                # Calculate angle between hip vector and horizontal
                dot_product = np.dot(hip_vector, horizontal)
                hip_magnitude = np.linalg.norm(hip_vector)
                
                if hip_magnitude == 0:
                    rotation_values.append(0.0)
                else:
                    cos_angle = dot_product / hip_magnitude
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle_rad = np.arccos(cos_angle)
                    
                    # Determine direction of rotation based on y-coordinate
                    if hip_vector[1] < 0:
                        angle_rad = -angle_rad
                    
                    rotation_values.append(np.degrees(angle_rad))
        
        return rotation_values
    
    def calculate_velocity(self, points: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate velocity magnitude for a sequence of 2D points.
        
        Args:
            points: Array of shape (sequence_length, 2) containing point coordinates
            dt: Time step between frames
            
        Returns:
            Array of velocity magnitudes
        """
        if len(points) < 2:
            return np.array([])
        
        # Calculate displacement between consecutive frames
        displacement = np.diff(points, axis=0)
        
        # Calculate velocity magnitude
        velocity = np.linalg.norm(displacement, axis=1) / dt
        
        return velocity
    
    def calculate_weight_transfer(self, keypoints: np.ndarray) -> List[float]:
        """
        Calculate weight transfer metric across a sequence of frames.
        
        Args:
            keypoints: Array of shape (sequence_length, num_keypoints, 2)
            
        Returns:
            List of weight transfer values for each frame
        """
        weight_transfer = []
        
        for frame in keypoints:
            left_ankle = frame[self.KEYPOINT_DICT['left_ankle']]
            right_ankle = frame[self.KEYPOINT_DICT['right_ankle']]
            left_hip = frame[self.KEYPOINT_DICT['left_hip']]
            right_hip = frame[self.KEYPOINT_DICT['right_hip']]
            
            # Skip if any keypoint is missing
            if (np.any(left_ankle == 0) or np.any(right_ankle == 0) or 
                np.any(left_hip == 0) or np.any(right_hip == 0)):
                weight_transfer.append(0.0)
                continue
            
            # Calculate center of mass for lower body (simplified)
            com = (left_ankle + right_ankle + left_hip + right_hip) / 4
            
            # Calculate the position relative to feet midpoint
            feet_midpoint = (left_ankle + right_ankle) / 2
            lateral_position = com[0] - feet_midpoint[0]
            
            weight_transfer.append(lateral_position)
        
        return weight_transfer
    
    def calculate_kinetic_chain_score(self, keypoints: np.ndarray, fps: float) -> float:
        """
        Calculate a score for kinetic chain efficiency.
        
        Args:
            keypoints: Array of shape (sequence_length, num_keypoints, 2)
            fps: Frames per second
            
        Returns:
            Kinetic chain efficiency score (0-100)
        """
        dt = 1.0 / fps
        
        # Calculate velocities for relevant joints
        hip_vel = self.calculate_velocity(keypoints[:, self.KEYPOINT_DICT[self.dom_hip]], dt)
        shoulder_vel = self.calculate_velocity(keypoints[:, self.KEYPOINT_DICT[self.dom_shoulder]], dt)
        elbow_vel = self.calculate_velocity(keypoints[:, self.KEYPOINT_DICT[self.dom_elbow]], dt)
        wrist_vel = self.calculate_velocity(keypoints[:, self.KEYPOINT_DICT[self.dom_wrist]], dt)
        
        # Find peak times for each joint
        if len(hip_vel) == 0 or len(shoulder_vel) == 0 or len(elbow_vel) == 0 or len(wrist_vel) == 0:
            return 0.0
            
        try:
            hip_peak_time = np.argmax(hip_vel)
            shoulder_peak_time = np.argmax(shoulder_vel)
            elbow_peak_time = np.argmax(elbow_vel)
            wrist_peak_time = np.argmax(wrist_vel)
            
            # Check if sequence is in correct order
            correct_sequence = (hip_peak_time <= shoulder_peak_time <= elbow_peak_time <= wrist_peak_time)
            
            # Calculate timing ratios
            if correct_sequence:
                # Time gaps between peak velocities
                hip_to_shoulder = shoulder_peak_time - hip_peak_time
                shoulder_to_elbow = elbow_peak_time - shoulder_peak_time
                elbow_to_wrist = wrist_peak_time - elbow_peak_time
                
                # Total sequence time
                total_time = wrist_peak_time - hip_peak_time
                
                if total_time == 0:
                    return 0.0
                
                # Calculate timing ratio score (higher is better)
                timing_score = min(50, 50 * (total_time / fps) / 0.5)  # Optimal swing is ~0.5 seconds
                
                # Calculate sequence order score
                order_score = 50
                
                return timing_score + order_score
            else:
                # Incorrect sequence order - calculate partial score based on which parts are correct
                partial_score = 0
                if hip_peak_time <= shoulder_peak_time:
                    partial_score += 10
                if shoulder_peak_time <= elbow_peak_time:
                    partial_score += 10
                if elbow_peak_time <= wrist_peak_time:
                    partial_score += 10
                
                return partial_score
                
        except Exception as e:
            logger.warning(f"Error calculating kinetic chain score: {e}")
            return 0.0
    
    def generate_metrics_report(self, metrics: Dict[str, Union[float, List[float]]]) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Generate a report with insights from the calculated metrics.
        
        Args:
            metrics: Dictionary of kinematic metrics
            
        Returns:
            Dictionary with insights and recommendations
        """
        report = {
            'elbow_angle': {
                'value': metrics.get('max_elbow_angle', 0),
                'interpretation': '',
                'recommendation': ''
            },
            'shoulder_angle': {
                'value': metrics.get('max_shoulder_angle', 0),
                'interpretation': '',
                'recommendation': ''
            },
            'knee_angle': {
                'value': metrics.get('max_knee_angle', 0),
                'interpretation': '',
                'recommendation': ''
            },
            'hip_rotation': {
                'value': metrics.get('max_hip_rotation', 0),
                'interpretation': '',
                'recommendation': ''
            },
            'racket_head_speed': {
                'value': metrics.get('racket_head_speed', 0),
                'interpretation': '',
                'recommendation': ''
            },
            'kinetic_chain_score': {
                'value': metrics.get('kinetic_chain_score', 0),
                'interpretation': '',
                'recommendation': ''
            }
        }
        
        # Add interpretations and recommendations based on values
        
        # Elbow angle
        elbow = report['elbow_angle']['value']
        if elbow > 170:
            report['elbow_angle']['interpretation'] = 'Excessive elbow extension'
            report['elbow_angle']['recommendation'] = 'Maintain slight bend in elbow for better control'
        elif elbow < 120:
            report['elbow_angle']['interpretation'] = 'Insufficient elbow extension'
            report['elbow_angle']['recommendation'] = 'Extend elbow more to generate power'
        else:
            report['elbow_angle']['interpretation'] = 'Good elbow extension'
            report['elbow_angle']['recommendation'] = 'Maintain current technique'
        
        # Shoulder angle
        shoulder = report['shoulder_angle']['value']
        if shoulder > 160:
            report['shoulder_angle']['interpretation'] = 'High shoulder rotation'
            report['shoulder_angle']['recommendation'] = 'Good power generation, ensure proper timing'
        elif shoulder < 100:
            report['shoulder_angle']['interpretation'] = 'Limited shoulder rotation'
            report['shoulder_angle']['recommendation'] = 'Increase shoulder turn to generate more power'
        else:
            report['shoulder_angle']['interpretation'] = 'Moderate shoulder rotation'
            report['shoulder_angle']['recommendation'] = 'Consider increasing shoulder turn for more power'
        
        # Knee angle
        knee = report['knee_angle']['value']
        if knee < 120:
            report['knee_angle']['interpretation'] = 'Good knee bend'
            report['knee_angle']['recommendation'] = 'Maintain this knee bend for optimal power transfer'
        elif knee > 160:
            report['knee_angle']['interpretation'] = 'Limited knee bend'
            report['knee_angle']['recommendation'] = 'Bend knees more to improve stability and power generation'
        else:
            report['knee_angle']['interpretation'] = 'Moderate knee bend'
            report['knee_angle']['recommendation'] = 'Consider slightly deeper knee bend for better power transfer'
        
        # Hip rotation
        hip = abs(report['hip_rotation']['value'])
        if hip > 45:
            report['hip_rotation']['interpretation'] = 'Good hip rotation'
            report['hip_rotation']['recommendation'] = 'Maintain this hip rotation for optimal power generation'
        elif hip < 20:
            report['hip_rotation']['interpretation'] = 'Limited hip rotation'
            report['hip_rotation']['recommendation'] = 'Increase hip rotation to improve power and consistency'
        else:
            report['hip_rotation']['interpretation'] = 'Moderate hip rotation'
            report['hip_rotation']['recommendation'] = 'Consider increasing hip rotation for better power'
        
        # Racket head speed
        speed = report['racket_head_speed']['value']
        if speed > 20:
            report['racket_head_speed']['interpretation'] = 'Excellent racket speed'
            report['racket_head_speed']['recommendation'] = 'Focus on consistency while maintaining speed'
        elif speed < 10:
            report['racket_head_speed']['interpretation'] = 'Low racket speed'
            report['racket_head_speed']['recommendation'] = 'Improve kinetic chain to increase racket speed'
        else:
            report['racket_head_speed']['interpretation'] = 'Moderate racket speed'
            report['racket_head_speed']['recommendation'] = 'Work on timing and kinetic chain to improve speed'
        
        # Kinetic chain score
        kc_score = report['kinetic_chain_score']['value']
        if kc_score > 80:
            report['kinetic_chain_score']['interpretation'] = 'Excellent kinetic chain efficiency'
            report['kinetic_chain_score']['recommendation'] = 'Maintain current technique'
        elif kc_score > 60:
            report['kinetic_chain_score']['interpretation'] = 'Good kinetic chain efficiency'
            report['kinetic_chain_score']['recommendation'] = 'Minor improvements in timing would help'
        elif kc_score > 40:
            report['kinetic_chain_score']['interpretation'] = 'Moderate kinetic chain efficiency'
            report['kinetic_chain_score']['recommendation'] = 'Work on sequential timing of joint movements'
        else:
            report['kinetic_chain_score']['interpretation'] = 'Poor kinetic chain efficiency'
            report['kinetic_chain_score']['recommendation'] = 'Focus on proper sequencing from ground up'
        
        return report 