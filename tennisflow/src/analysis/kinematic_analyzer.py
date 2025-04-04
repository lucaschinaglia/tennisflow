"""
Kinematic Analyzer Module

This module provides functionality for analyzing the kinematics of tennis shots,
including joint angles, velocities, and other biomechanical metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KinematicAnalyzer:
    """Analyzer for tennis shot kinematics."""
    
    def __init__(self):
        """Initialize the kinematic analyzer."""
        # Define key joint indices for MoveNet model
        self.joint_indices = {
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
    
    def analyze_shot(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the kinematics of a tennis shot.
        
        Args:
            keypoints: Array of keypoints, shape (frames, keypoints, coordinates)
            
        Returns:
            Dictionary of kinematic analysis results
        """
        logger.info("Analyzing shot kinematics")
        
        # Check input shape
        if len(keypoints.shape) != 3:
            logger.error(f"Invalid keypoints shape: {keypoints.shape}")
            return {}
        
        n_frames, n_keypoints, n_coords = keypoints.shape
        
        if n_keypoints != 17 or n_coords != 2:
            logger.error(f"Expected keypoints shape (frames, 17, 2), got {keypoints.shape}")
            return {}
        
        logger.info(f"Analyzing sequence of {n_frames} frames")
        
        # Calculate kinematics
        joint_angles = self._calculate_joint_angles(keypoints)
        joint_velocities = self._calculate_joint_velocities(keypoints)
        peak_velocities = self._calculate_peak_velocities(joint_velocities)
        
        # Calculate shot phases
        phases = self._identify_shot_phases(keypoints, joint_angles, joint_velocities)
        
        # Calculate center of mass motion
        com_motion = self._calculate_center_of_mass_motion(keypoints)
        
        # Calculate pose ratings
        pose_ratings = self._rate_pose_quality(keypoints, joint_angles, phases)
        
        # Compile results
        results = {
            'joint_angles': joint_angles,
            'joint_velocities': peak_velocities,
            'phases': phases,
            'com_motion': com_motion,
            'pose_ratings': pose_ratings,
            'summary': self._generate_summary(joint_angles, peak_velocities, phases, pose_ratings)
        }
        
        logger.info("Kinematic analysis completed")
        return results
    
    def _calculate_joint_angles(self, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate joint angles throughout the sequence.
        
        Args:
            keypoints: Array of keypoints, shape (frames, keypoints, coordinates)
            
        Returns:
            Dictionary of joint angles (in degrees)
        """
        n_frames = keypoints.shape[0]
        
        # Initialize angle dictionaries
        joint_angles = {
            'right_elbow': np.zeros(n_frames),
            'left_elbow': np.zeros(n_frames),
            'right_shoulder': np.zeros(n_frames),
            'left_shoulder': np.zeros(n_frames),
            'right_hip': np.zeros(n_frames),
            'left_hip': np.zeros(n_frames),
            'right_knee': np.zeros(n_frames),
            'left_knee': np.zeros(n_frames),
            'trunk': np.zeros(n_frames)
        }
        
        # Calculate angles for each frame
        for i in range(n_frames):
            # Extract keypoints for current frame
            kp = keypoints[i]
            
            # Right elbow angle (right shoulder - right elbow - right wrist)
            joint_angles['right_elbow'][i] = self._calculate_angle(
                kp[self.joint_indices['right_shoulder']],
                kp[self.joint_indices['right_elbow']],
                kp[self.joint_indices['right_wrist']]
            )
            
            # Left elbow angle (left shoulder - left elbow - left wrist)
            joint_angles['left_elbow'][i] = self._calculate_angle(
                kp[self.joint_indices['left_shoulder']],
                kp[self.joint_indices['left_elbow']],
                kp[self.joint_indices['left_wrist']]
            )
            
            # Right shoulder angle (right hip - right shoulder - right elbow)
            joint_angles['right_shoulder'][i] = self._calculate_angle(
                kp[self.joint_indices['right_hip']],
                kp[self.joint_indices['right_shoulder']],
                kp[self.joint_indices['right_elbow']]
            )
            
            # Left shoulder angle (left hip - left shoulder - left elbow)
            joint_angles['left_shoulder'][i] = self._calculate_angle(
                kp[self.joint_indices['left_hip']],
                kp[self.joint_indices['left_shoulder']],
                kp[self.joint_indices['left_elbow']]
            )
            
            # Right hip angle (right shoulder - right hip - right knee)
            joint_angles['right_hip'][i] = self._calculate_angle(
                kp[self.joint_indices['right_shoulder']],
                kp[self.joint_indices['right_hip']],
                kp[self.joint_indices['right_knee']]
            )
            
            # Left hip angle (left shoulder - left hip - left knee)
            joint_angles['left_hip'][i] = self._calculate_angle(
                kp[self.joint_indices['left_shoulder']],
                kp[self.joint_indices['left_hip']],
                kp[self.joint_indices['left_knee']]
            )
            
            # Right knee angle (right hip - right knee - right ankle)
            joint_angles['right_knee'][i] = self._calculate_angle(
                kp[self.joint_indices['right_hip']],
                kp[self.joint_indices['right_knee']],
                kp[self.joint_indices['right_ankle']]
            )
            
            # Left knee angle (left hip - left knee - left ankle)
            joint_angles['left_knee'][i] = self._calculate_angle(
                kp[self.joint_indices['left_hip']],
                kp[self.joint_indices['left_knee']],
                kp[self.joint_indices['left_ankle']]
            )
            
            # Trunk angle (vertical alignment)
            mid_shoulder = (kp[self.joint_indices['left_shoulder']] + kp[self.joint_indices['right_shoulder']]) / 2
            mid_hip = (kp[self.joint_indices['left_hip']] + kp[self.joint_indices['right_hip']]) / 2
            vertical = np.array([mid_shoulder[0], 0])  # Vertical reference
            
            trunk_vector = mid_shoulder - mid_hip
            joint_angles['trunk'][i] = self._calculate_angle_between_vectors(trunk_vector, vertical)
        
        return joint_angles
    
    def _calculate_joint_velocities(self, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate joint velocities throughout the sequence.
        
        Args:
            keypoints: Array of keypoints, shape (frames, keypoints, coordinates)
            
        Returns:
            Dictionary of joint velocities
        """
        n_frames = keypoints.shape[0]
        
        # Skip if too few frames
        if n_frames < 2:
            return {}
        
        # Initialize velocities
        velocities = {
            'right_wrist': np.zeros(n_frames - 1),
            'left_wrist': np.zeros(n_frames - 1),
            'right_elbow': np.zeros(n_frames - 1),
            'left_elbow': np.zeros(n_frames - 1),
            'right_shoulder': np.zeros(n_frames - 1),
            'left_shoulder': np.zeros(n_frames - 1),
            'right_hip': np.zeros(n_frames - 1),
            'left_hip': np.zeros(n_frames - 1)
        }
        
        # Calculate velocities (frame-to-frame displacement)
        for joint, index in [
            ('right_wrist', self.joint_indices['right_wrist']),
            ('left_wrist', self.joint_indices['left_wrist']),
            ('right_elbow', self.joint_indices['right_elbow']),
            ('left_elbow', self.joint_indices['left_elbow']),
            ('right_shoulder', self.joint_indices['right_shoulder']),
            ('left_shoulder', self.joint_indices['left_shoulder']),
            ('right_hip', self.joint_indices['right_hip']),
            ('left_hip', self.joint_indices['left_hip'])
        ]:
            # Calculate velocity as the norm of the displacement between consecutive frames
            displacements = np.linalg.norm(np.diff(keypoints[:, index, :], axis=0), axis=1)
            velocities[joint] = displacements
        
        return velocities
    
    def _calculate_peak_velocities(self, velocities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate peak velocities for each joint.
        
        Args:
            velocities: Dictionary of joint velocities
            
        Returns:
            Dictionary of peak velocities
        """
        peak_velocities = {}
        
        for joint, vel in velocities.items():
            if len(vel) > 0:
                peak_velocities[joint] = float(np.max(vel))
            else:
                peak_velocities[joint] = 0.0
        
        return peak_velocities
    
    def _identify_shot_phases(
        self, 
        keypoints: np.ndarray, 
        joint_angles: Dict[str, np.ndarray], 
        velocities: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, int]]:
        """
        Identify the phases of the tennis shot.
        
        Args:
            keypoints: Array of keypoints
            joint_angles: Dictionary of joint angles
            velocities: Dictionary of joint velocities
            
        Returns:
            Dictionary of shot phases with frame indices
        """
        n_frames = keypoints.shape[0]
        
        # Skip if too few frames
        if n_frames < 5:
            return {'preparation': {'start': 0, 'end': 0}, 
                    'backswing': {'start': 0, 'end': 0}, 
                    'forward_swing': {'start': 0, 'end': 0}, 
                    'impact': {'start': 0, 'end': 0}, 
                    'follow_through': {'start': 0, 'end': n_frames - 1}}
        
        # Simplistic phase detection based on wrist velocity
        # Would be more sophisticated in a real system
        
        # Use dominant wrist velocity (assuming right-handed player)
        wrist_vel = velocities.get('right_wrist', np.zeros(n_frames - 1))
        
        if len(wrist_vel) < 3:
            return {'preparation': {'start': 0, 'end': 0}, 
                    'backswing': {'start': 0, 'end': 0}, 
                    'forward_swing': {'start': 0, 'end': 0}, 
                    'impact': {'start': 0, 'end': 0}, 
                    'follow_through': {'start': 0, 'end': n_frames - 1}}
        
        # Smooth velocity
        smoothed_vel = np.convolve(wrist_vel, np.ones(3)/3, mode='valid')
        
        # Find peak velocity (approximate impact)
        impact_idx = np.argmax(smoothed_vel) + 1  # +1 for the smoothing offset
        
        # Approximate other phases
        prep_end = max(0, int(n_frames * 0.2))
        backswing_end = max(prep_end, min(impact_idx - 1, int(n_frames * 0.4)))
        forward_swing_start = backswing_end
        impact_start = max(forward_swing_start, impact_idx - 2)
        impact_end = min(n_frames - 1, impact_idx + 2)
        follow_through_start = impact_end
        
        phases = {
            'preparation': {'start': 0, 'end': prep_end},
            'backswing': {'start': prep_end, 'end': backswing_end},
            'forward_swing': {'start': forward_swing_start, 'end': impact_start},
            'impact': {'start': impact_start, 'end': impact_end},
            'follow_through': {'start': follow_through_start, 'end': n_frames - 1}
        }
        
        return phases
    
    def _calculate_center_of_mass_motion(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Calculate center of mass motion metrics.
        
        Args:
            keypoints: Array of keypoints
            
        Returns:
            Dictionary of COM motion metrics
        """
        n_frames = keypoints.shape[0]
        com_motion = {}
        
        if n_frames < 2:
            return com_motion
        
        # Approximate center of mass using hip midpoint
        com_trajectory = np.zeros((n_frames, 2))
        
        for i in range(n_frames):
            left_hip = keypoints[i, self.joint_indices['left_hip']]
            right_hip = keypoints[i, self.joint_indices['right_hip']]
            com_trajectory[i] = (left_hip + right_hip) / 2
        
        # Calculate displacement
        total_displacement = np.linalg.norm(com_trajectory[-1] - com_trajectory[0])
        
        # Calculate path length
        path_length = np.sum(np.linalg.norm(np.diff(com_trajectory, axis=0), axis=1))
        
        # Calculate linear vs. non-linear motion ratio
        linearity = total_displacement / max(path_length, 1e-6)
        
        # Calculate vertical and horizontal components
        vertical_displacement = abs(com_trajectory[-1, 1] - com_trajectory[0, 1])
        horizontal_displacement = abs(com_trajectory[-1, 0] - com_trajectory[0, 0])
        
        com_motion = {
            'total_displacement': float(total_displacement),
            'path_length': float(path_length),
            'linearity': float(linearity),
            'vertical_displacement': float(vertical_displacement),
            'horizontal_displacement': float(horizontal_displacement)
        }
        
        return com_motion
    
    def _rate_pose_quality(
        self, 
        keypoints: np.ndarray, 
        joint_angles: Dict[str, np.ndarray], 
        phases: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """
        Rate the quality of the pose during different shot phases.
        
        Args:
            keypoints: Array of keypoints
            joint_angles: Dictionary of joint angles
            phases: Dictionary of shot phases
            
        Returns:
            Dictionary of pose quality ratings (0-10 scale)
        """
        # Skip if missing data
        if not joint_angles or not phases:
            return {'overall': 5.0, 'preparation': 5.0, 'backswing': 5.0, 
                    'forward_swing': 5.0, 'impact': 5.0, 'follow_through': 5.0}
        
        ratings = {}
        
        # Preparation phase rating
        prep_phase = phases.get('preparation', {'start': 0, 'end': 0})
        if prep_phase['end'] > prep_phase['start']:
            prep_frames = slice(prep_phase['start'], prep_phase['end'])
            
            # Check knee bend (should be slightly bent)
            knee_angle_avg = np.mean(joint_angles['right_knee'][prep_frames])
            knee_rating = 10 - min(10, abs(knee_angle_avg - 160) / 3)
            
            # Check trunk angle (should be upright)
            trunk_angle_avg = np.mean(joint_angles['trunk'][prep_frames])
            trunk_rating = 10 - min(10, abs(trunk_angle_avg) / 3)
            
            # Combine ratings
            ratings['preparation'] = float((knee_rating + trunk_rating) / 2)
        else:
            ratings['preparation'] = 5.0
        
        # Backswing phase rating
        bs_phase = phases.get('backswing', {'start': 0, 'end': 0})
        if bs_phase['end'] > bs_phase['start']:
            bs_frames = slice(bs_phase['start'], bs_phase['end'])
            
            # Check shoulder rotation
            shoulder_angle_avg = np.mean(joint_angles['right_shoulder'][bs_frames])
            shoulder_rating = min(10, max(0, shoulder_angle_avg - 100) / 5)
            
            # Check elbow angle (should be bent)
            elbow_angle_avg = np.mean(joint_angles['right_elbow'][bs_frames])
            elbow_rating = 10 - min(10, abs(elbow_angle_avg - 90) / 5)
            
            # Combine ratings
            ratings['backswing'] = float((shoulder_rating + elbow_rating) / 2)
        else:
            ratings['backswing'] = 5.0
        
        # Forward swing phase rating
        fs_phase = phases.get('forward_swing', {'start': 0, 'end': 0})
        if fs_phase['end'] > fs_phase['start']:
            fs_frames = slice(fs_phase['start'], fs_phase['end'])
            
            # Check hip-shoulder separation
            hip_shoulder_sep = abs(np.mean(joint_angles['trunk'][fs_frames]))
            hip_shoulder_rating = min(10, hip_shoulder_sep / 5)
            
            # Check knee extension
            knee_extension = np.mean(np.diff(joint_angles['right_knee'][fs_frames]))
            knee_rating = min(10, max(0, knee_extension + 5) / 2)
            
            # Combine ratings
            ratings['forward_swing'] = float((hip_shoulder_rating + knee_rating) / 2)
        else:
            ratings['forward_swing'] = 5.0
        
        # Impact phase rating
        impact_phase = phases.get('impact', {'start': 0, 'end': 0})
        if impact_phase['end'] > impact_phase['start']:
            impact_frames = slice(impact_phase['start'], impact_phase['end'])
            
            # Check arm extension
            elbow_angle_avg = np.mean(joint_angles['right_elbow'][impact_frames])
            extension_rating = min(10, max(0, elbow_angle_avg - 120) / 6)
            
            # Combine ratings
            ratings['impact'] = float(extension_rating)
        else:
            ratings['impact'] = 5.0
        
        # Follow-through phase rating
        ft_phase = phases.get('follow_through', {'start': 0, 'end': 0})
        if ft_phase['end'] > ft_phase['start']:
            ft_frames = slice(ft_phase['start'], ft_phase['end'])
            
            # Check follow-through extension
            shoulder_angle_avg = np.mean(joint_angles['right_shoulder'][ft_frames])
            ft_rating = min(10, max(0, 180 - shoulder_angle_avg) / 8)
            
            # Combine ratings
            ratings['follow_through'] = float(ft_rating)
        else:
            ratings['follow_through'] = 5.0
        
        # Overall rating (weighted average)
        weights = {
            'preparation': 0.1,
            'backswing': 0.2,
            'forward_swing': 0.3,
            'impact': 0.3,
            'follow_through': 0.1
        }
        
        overall_rating = sum(ratings.get(phase, 5.0) * weight 
                             for phase, weight in weights.items())
        
        ratings['overall'] = float(overall_rating)
        
        return ratings
    
    def _generate_summary(
        self, 
        joint_angles: Dict[str, np.ndarray],
        peak_velocities: Dict[str, float],
        phases: Dict[str, Dict[str, int]],
        pose_ratings: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate a summary of the kinematic analysis.
        
        Args:
            joint_angles: Dictionary of joint angles
            peak_velocities: Dictionary of peak velocities
            phases: Dictionary of shot phases
            pose_ratings: Dictionary of pose quality ratings
            
        Returns:
            Dictionary of summary metrics
        """
        summary = {}
        
        # Key joint angles at impact
        impact_phase = phases.get('impact', {'start': 0, 'end': 0})
        if impact_phase['end'] > impact_phase['start']:
            impact_frame = impact_phase['start']
            
            summary['impact_angle_elbow'] = float(joint_angles['right_elbow'][impact_frame]) if len(joint_angles['right_elbow']) > impact_frame else None
            summary['impact_angle_shoulder'] = float(joint_angles['right_shoulder'][impact_frame]) if len(joint_angles['right_shoulder']) > impact_frame else None
            summary['impact_angle_knee'] = float(joint_angles['right_knee'][impact_frame]) if len(joint_angles['right_knee']) > impact_frame else None
        else:
            summary['impact_angle_elbow'] = None
            summary['impact_angle_shoulder'] = None
            summary['impact_angle_knee'] = None
        
        # Peak wrist and elbow velocities
        summary['peak_velocity_wrist'] = peak_velocities.get('right_wrist', 0.0)
        summary['peak_velocity_elbow'] = peak_velocities.get('right_elbow', 0.0)
        
        # Range of motion
        for joint in ['right_elbow', 'right_shoulder', 'right_knee']:
            if joint in joint_angles and len(joint_angles[joint]) > 1:
                summary[f'range_of_motion_{joint}'] = float(np.max(joint_angles[joint]) - np.min(joint_angles[joint]))
            else:
                summary[f'range_of_motion_{joint}'] = 0.0
        
        # Pose quality ratings
        summary['pose_rating_overall'] = pose_ratings.get('overall', 5.0)
        summary['pose_rating_backswing'] = pose_ratings.get('backswing', 5.0)
        summary['pose_rating_impact'] = pose_ratings.get('impact', 5.0)
        
        # Areas for improvement
        improvements = []
        
        if summary['pose_rating_backswing'] < 6.0:
            improvements.append("Improve backswing position for better shot preparation")
        
        if summary['pose_rating_impact'] < 6.0:
            improvements.append("Work on impact position for better power transfer")
        
        if summary.get('range_of_motion_right_shoulder', 0.0) < 60.0:
            improvements.append("Increase shoulder rotation for more power")
        
        if summary.get('range_of_motion_right_knee', 0.0) < 30.0:
            improvements.append("Use more leg drive for better power generation")
        
        summary['areas_for_improvement'] = improvements
        
        return summary
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            p1: First point coordinates
            p2: Second point coordinates (vertex)
            p3: Third point coordinates
            
        Returns:
            Angle in degrees
        """
        # Skip if any points are missing
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3)):
            return 0.0
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # Clip to handle numerical errors
        cosine = np.clip(cosine, -1.0, 1.0)
        
        # Return angle in degrees
        return float(np.degrees(np.arccos(cosine)))
    
    def _calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate the angle between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle in degrees
        """
        # Skip if any vectors are missing
        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
            return 0.0
        
        # Ensure non-zero vectors
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return 0.0
        
        # Calculate angle using dot product
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # Clip to handle numerical errors
        cosine = np.clip(cosine, -1.0, 1.0)
        
        # Return angle in degrees
        return float(np.degrees(np.arccos(cosine))) 