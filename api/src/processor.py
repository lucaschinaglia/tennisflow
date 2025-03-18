import os
import sys
import time
import json
import logging
import tempfile
import shutil
import uuid
import urllib.request
from datetime import datetime
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

# Try importing MediaPipe as a fallback
try:
    import mediapipe as mp
    mediapipe_available = True
except ImportError:
    mediapipe_available = False
    logging.warning("MediaPipe not available. Using fallback pose estimation.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe if available
if mediapipe_available:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_model = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

# Helper Functions
def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()

def download_video(url: str, target_path: str) -> str:
    """Download video from URL to target path"""
    urllib.request.urlretrieve(url, target_path)
    return target_path

def extract_frames(video_path: str, output_dir: str, fps: int = 30) -> List[str]:
    """Extract frames from video at specified FPS"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval to achieve target FPS
    interval = max(1, round(video_fps / fps))
    
    # Extract frames
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    
    return frame_paths

def analyze_frames(frame_paths: List[str]) -> List[Dict[str, Any]]:
    """Analyze frames to extract pose data"""
    frame_poses = []
    
    if mediapipe_available:
        # Use MediaPipe for pose estimation
        for i, frame_path in enumerate(frame_paths):
            try:
                # Load image
                image = cv2.imread(frame_path)
                if image is None:
                    logger.warning(f"Could not read image: {frame_path}")
                    continue
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape
                
                # Get pose
                results = pose_model.process(image_rgb)
                
                if results.pose_landmarks:
                    # Convert landmarks to our format
                    keypoints = []
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        name = mp_pose.PoseLandmark(idx).name.lower()
                        keypoints.append({
                            "name": name,
                            "position": {
                                "x": landmark.x,  # Normalized 0-1
                                "y": landmark.y   # Normalized 0-1
                            },
                            "confidence": landmark.visibility
                        })
                    
                    # Define connections between keypoints
                    connections = []
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        start_name = mp_pose.PoseLandmark(start_idx).name.lower()
                        end_name = mp_pose.PoseLandmark(end_idx).name.lower()
                        connections.append({
                            "from": start_name,
                            "to": end_name
                        })
                    
                    frame_poses.append({
                        "frame_number": i,
                        "keypoints": keypoints,
                        "connections": connections
                    })
                else:
                    logger.warning(f"No pose detected in frame: {frame_path}")
            except Exception as e:
                logger.error(f"Error analyzing frame {frame_path}: {e}")
    else:
        # Fallback: Generate mock pose data
        logger.info("Using mock pose data generator (MediaPipe not available)")
        for i, frame_path in enumerate(frame_paths):
            try:
                # Load image to get dimensions
                image = cv2.imread(frame_path)
                if image is None:
                    logger.warning(f"Could not read image: {frame_path}")
                    continue
                
                h, w, _ = image.shape
                
                # Generate mock keypoints
                # In a real implementation, this would use OpenPose or another model
                keypoints = generate_mock_keypoints(i, len(frame_paths))
                
                connections = [
                    {"from": "nose", "to": "left_shoulder"},
                    {"from": "nose", "to": "right_shoulder"},
                    {"from": "left_shoulder", "to": "right_shoulder"},
                    {"from": "left_shoulder", "to": "left_elbow"},
                    {"from": "right_shoulder", "to": "right_elbow"},
                    {"from": "left_elbow", "to": "left_wrist"},
                    {"from": "right_elbow", "to": "right_wrist"},
                    {"from": "left_shoulder", "to": "left_hip"},
                    {"from": "right_shoulder", "to": "right_hip"},
                    {"from": "left_hip", "to": "right_hip"},
                    {"from": "left_hip", "to": "left_knee"},
                    {"from": "right_hip", "to": "right_knee"},
                    {"from": "left_knee", "to": "left_ankle"},
                    {"from": "right_knee", "to": "right_ankle"}
                ]
                
                frame_poses.append({
                    "frame_number": i,
                    "keypoints": keypoints,
                    "connections": connections
                })
            except Exception as e:
                logger.error(f"Error generating mock data for frame {frame_path}: {e}")
    
    return frame_poses

def generate_mock_keypoints(frame_idx: int, total_frames: int) -> List[Dict[str, Any]]:
    """Generate mock keypoints for testing"""
    progress = frame_idx / total_frames
    
    # Common keypoint names
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    keypoints = []
    
    # Initial positions
    base_positions = {
        "nose": (0.5, 0.15),
        "left_eye": (0.45, 0.13),
        "right_eye": (0.55, 0.13),
        "left_ear": (0.4, 0.15),
        "right_ear": (0.6, 0.15),
        "left_shoulder": (0.4, 0.3),
        "right_shoulder": (0.6, 0.3),
        "left_elbow": (0.3, 0.4),
        "right_elbow": (0.7, 0.4),
        "left_wrist": (0.25, 0.5),
        "right_wrist": (0.75, 0.5),
        "left_hip": (0.45, 0.6),
        "right_hip": (0.55, 0.6),
        "left_knee": (0.45, 0.75),
        "right_knee": (0.55, 0.75),
        "left_ankle": (0.45, 0.9),
        "right_ankle": (0.55, 0.9)
    }
    
    # Simulate tennis swing motion based on progress
    # 0.0-0.2: preparation, 0.2-0.4: backswing, 0.4-0.6: forward swing, 
    # 0.6-0.7: contact, 0.7-1.0: follow through
    for name in keypoint_names:
        x, y = base_positions[name]
        confidence = 0.8 + 0.2 * np.random.random()
        
        # Add motion based on phase
        if "right" in name:
            if progress < 0.2:  # Preparation
                pass  # No significant movement
            elif progress < 0.4:  # Backswing
                if "elbow" in name or "wrist" in name:
                    x += 0.1 * (progress - 0.2) / 0.2
                    y -= 0.15 * (progress - 0.2) / 0.2
                elif "shoulder" in name:
                    x += 0.05 * (progress - 0.2) / 0.2
                    y -= 0.05 * (progress - 0.2) / 0.2
            elif progress < 0.6:  # Forward swing
                if "elbow" in name or "wrist" in name:
                    x -= 0.3 * (progress - 0.4) / 0.2
                    y += 0.1 * (progress - 0.4) / 0.2
                elif "shoulder" in name:
                    x -= 0.1 * (progress - 0.4) / 0.2
            elif progress < 0.7:  # Contact
                if "wrist" in name:
                    x -= 0.1 * (progress - 0.6) / 0.1
                    y += 0.05 * (progress - 0.6) / 0.1
            else:  # Follow through
                if "elbow" in name or "wrist" in name:
                    x -= 0.05 * (progress - 0.7) / 0.3
                    y += 0.1 * (progress - 0.7) / 0.3
        
        # Add slight noise for natural movement
        x += 0.01 * np.random.randn()
        y += 0.01 * np.random.randn()
        
        # Ensure values are within range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        keypoints.append({
            "name": name,
            "position": {"x": x, "y": y},
            "confidence": confidence
        })
    
    return keypoints

def identify_swing_phase(frame_idx: int, total_frames: int) -> str:
    """Identify the tennis swing phase based on frame index"""
    progress = frame_idx / total_frames
    
    if progress < 0.2:
        return "preparation"
    elif progress < 0.4:
        return "backswing"
    elif progress < 0.6:
        return "forward-swing"
    elif progress < 0.7:
        return "contact"
    else:
        return "follow-through"

def calculate_metrics(poses: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate performance metrics from pose data"""
    # In a real implementation, this would analyze the poses to extract metrics
    # For now, we'll return mock metrics
    return {
        "racketSpeed": 80 + 15 * np.random.random(),      # mph
        "hipRotation": 40 + 15 * np.random.random(),      # degrees
        "shoulderRotation": 70 + 20 * np.random.random(), # degrees
        "kneeFlexion": 25 + 15 * np.random.random(),      # degrees
        "weightTransfer": 60 + 30 * np.random.random(),   # percentage
        "balanceScore": 50 + 40 * np.random.random(),     # 0-100
        "followThrough": 50 + 40 * np.random.random(),    # 0-100
        "consistency": 40 + 50 * np.random.random(),      # 0-100
    }

def generate_annotations(frame_idx: int, total_frames: int, keypoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate annotations for a frame based on swing phase"""
    phase = identify_swing_phase(frame_idx, total_frames)
    annotations = []
    
    if phase == "preparation":
        # Check knee bend
        right_knee = next((kp for kp in keypoints if kp["name"] == "right_knee"), None)
        right_hip = next((kp for kp in keypoints if kp["name"] == "right_hip"), None)
        right_ankle = next((kp for kp in keypoints if kp["name"] == "right_ankle"), None)
        
        if right_knee and right_hip and right_ankle:
            # Calculate knee angle (simplified)
            knee_angle = 160 + 20 * np.random.random()  # Mock angle
            annotations.append({
                "type": "angle",
                "text": f"Knee Flexion: {knee_angle:.1f}°",
                "position": right_knee["position"],
                "value": knee_angle,
                "color": "#FFC107" if knee_angle > 150 else "#4CAF50"
            })
    
    elif phase == "backswing":
        # Check hip rotation
        right_hip = next((kp for kp in keypoints if kp["name"] == "right_hip"), None)
        left_hip = next((kp for kp in keypoints if kp["name"] == "left_hip"), None)
        
        if right_hip and left_hip:
            hip_rotation = 30 + 20 * np.random.random()  # Mock angle
            mid_x = (right_hip["position"]["x"] + left_hip["position"]["x"]) / 2
            mid_y = (right_hip["position"]["y"] + left_hip["position"]["y"]) / 2
            
            annotations.append({
                "type": "angle",
                "text": f"Hip Rotation: {hip_rotation:.1f}°",
                "position": {"x": mid_x, "y": mid_y},
                "value": hip_rotation,
                "color": "#4CAF50" if hip_rotation > 40 else "#FFC107"
            })
    
    elif phase == "forward-swing":
        # Check racket path
        right_wrist = next((kp for kp in keypoints if kp["name"] == "right_wrist"), None)
        
        if right_wrist:
            annotations.append({
                "type": "movement",
                "text": "Racket Path",
                "position": right_wrist["position"],
                "color": "#2196F3"
            })
    
    elif phase == "contact":
        # Check contact point
        right_wrist = next((kp for kp in keypoints if kp["name"] == "right_wrist"), None)
        
        if right_wrist:
            annotations.append({
                "type": "position",
                "text": "Contact Point",
                "position": right_wrist["position"],
                "color": "#F44336"
            })
    
    elif phase == "follow-through":
        # Check follow-through extension
        right_elbow = next((kp for kp in keypoints if kp["name"] == "right_elbow"), None)
        right_wrist = next((kp for kp in keypoints if kp["name"] == "right_wrist"), None)
        
        if right_elbow and right_wrist:
            extension = 80 + 20 * np.random.random()  # Mock value
            annotations.append({
                "type": "angle",
                "text": f"Extension: {extension:.1f}%",
                "position": {"x": right_elbow["position"]["x"], "y": right_elbow["position"]["y"]},
                "value": extension,
                "color": "#4CAF50" if extension > 85 else "#FFC107"
            })
    
    return annotations

def generate_analysis_results(video_id: str, frame_poses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate complete analysis results from pose data"""
    total_frames = len(frame_poses)
    
    # Process each frame
    frames = []
    for i, pose in enumerate(frame_poses):
        frame_number = pose.get("frame_number", i)
        timestamp = frame_number / 30.0  # Assuming 30fps
        
        # Identify swing phase
        swing_phase = identify_swing_phase(i, total_frames)
        
        # Generate annotations
        annotations = generate_annotations(i, total_frames, pose.get("keypoints", []))
        
        frames.append({
            "frameNumber": frame_number,
            "timestamp": timestamp,
            "poseData": {
                "keypoints": pose.get("keypoints", []),
                "connections": pose.get("connections", [])
            },
            "swingPhase": swing_phase,
            "annotations": annotations
        })
    
    # Calculate metrics
    metrics = calculate_metrics(frame_poses)
    
    # Find key frames (one for each phase)
    key_frames = []
    phase_indices = {
        "preparation": int(0.1 * total_frames),
        "backswing": int(0.3 * total_frames),
        "forward-swing": int(0.5 * total_frames),
        "contact": int(0.65 * total_frames),
        "follow-through": int(0.85 * total_frames)
    }
    
    for phase, idx in phase_indices.items():
        key_frames.append(idx)
    
    # Generate strengths and weaknesses
    strengths = [
        "Good follow-through extension",
        "Proper weight transfer timing",
        "Consistent racket path"
    ]
    
    weaknesses = [
        "Incomplete hip rotation during backswing",
        "Slight loss of balance during follow-through",
        "Racket face angle could be improved at contact"
    ]
    
    # Generate improvement suggestions
    suggestions = [
        "Focus on increasing hip rotation during backswing phase",
        "Practice maintaining balance through the entire swing",
        "Work on racket face control at contact point"
    ]
    
    # Build the complete analysis
    analysis_results = {
        "videoId": video_id,
        "duration": total_frames / 30.0,  # Assuming 30fps
        "frames": frames,
        "summary": {
            "swingType": "forehand",  # In real app, this would be detected
            "swingCount": 1,          # In real app, this would be detected
            "averageMetrics": metrics,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvementSuggestions": suggestions
        },
        "swings": [
            {
                "id": str(uuid.uuid4()),
                "startTime": 0,
                "endTime": total_frames / 30.0,
                "swingType": "forehand",
                "metrics": metrics,
                "keyFrames": key_frames,
                "score": 70 + 20 * np.random.random()  # 0-100 score
            }
        ]
    }
    
    return analysis_results