"""
API integration for TennisFlow models.

This module provides functions to integrate the models with the FastAPI backend.
"""
import os
import sys
import torch
import numpy as np
import cv2
import logging
import tempfile
import base64
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

# Configure logging
logger = logging.getLogger("tennisflow.models.api_integration")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import INFERENCE_CONFIG
from inference import TennisAnalyzer

# Global analyzer instance (lazy initialization)
_analyzer = None

def initialize_analyzer(
    pose_model_path: Optional[str] = None,
    event_model_path: Optional[str] = None,
    device: Optional[str] = None
) -> None:
    """
    Initialize the TennisAnalyzer with specified models.
    
    Args:
        pose_model_path: Path to pose model weights
        event_model_path: Path to event model weights
        device: Device to run inference on ('cpu' or 'cuda')
    """
    global _analyzer
    
    # Use specified paths or fallback to config
    pose_path = pose_model_path or INFERENCE_CONFIG.get('pose_model_path')
    event_path = event_model_path or INFERENCE_CONFIG.get('event_model_path')
    inference_device = device or INFERENCE_CONFIG.get('device')
    
    logger.info(f"Initializing TennisAnalyzer with device: {inference_device}")
    logger.info(f"Pose model path: {pose_path}")
    logger.info(f"Event model path: {event_path}")
    
    try:
        _analyzer = TennisAnalyzer(
            pose_model_path=pose_path,
            event_model_path=event_path,
            device=inference_device
        )
        logger.info("TennisAnalyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TennisAnalyzer: {e}")
        _analyzer = None

def is_analyzer_ready() -> bool:
    """
    Check if the analyzer is initialized and ready.
    
    Returns:
        True if analyzer is ready, False otherwise
    """
    global _analyzer
    
    if _analyzer is None:
        return False
    
    return _analyzer.is_ready()

def analyze_video_from_path(video_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze a video from a file path.
    
    Args:
        video_path: Path to the video file
        options: Additional analysis options
    
    Returns:
        Analysis results as a dictionary
    """
    global _analyzer
    
    if _analyzer is None or not is_analyzer_ready():
        initialize_analyzer()
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return {"error": "Video file not found"}
    
    try:
        results = _analyzer.analyze_video(video_path, options=options or {})
        return results
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        return {"error": str(e)}

def analyze_video_from_url(url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze a video from a URL.
    
    Args:
        url: URL of the video
        options: Additional analysis options
    
    Returns:
        Analysis results as a dictionary
    """
    global _analyzer
    
    if _analyzer is None or not is_analyzer_ready():
        initialize_analyzer()
    
    try:
        results = _analyzer.analyze_video_url(url, options=options or {})
        return results
    except Exception as e:
        logger.error(f"Error analyzing video from URL: {e}")
        return {"error": str(e)}

def analyze_video_from_base64(base64_data: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze a video from base64 encoded data.
    
    Args:
        base64_data: Base64 encoded video data
        options: Additional analysis options
    
    Returns:
        Analysis results as a dictionary
    """
    global _analyzer
    
    if _analyzer is None or not is_analyzer_ready():
        initialize_analyzer()
    
    try:
        # Decode base64 data
        video_data = base64.b64decode(base64_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(video_data)
        
        # Analyze the video
        results = _analyzer.analyze_video(temp_path, options=options or {})
        
        # Clean up
        os.unlink(temp_path)
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing video from base64: {e}")
        return {"error": str(e)}

def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Analyze a single image for pose classification.
    
    Args:
        image_path: Path to the image
    
    Returns:
        Analysis results as a dictionary
    """
    global _analyzer
    
    if _analyzer is None or not is_analyzer_ready():
        initialize_analyzer()
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return {"error": "Image file not found"}
    
    try:
        results = _analyzer.analyze_image(image_path)
        return results
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {"error": str(e)}

def get_last_analysis_results() -> Dict[str, Any]:
    """
    Get the results of the last analysis.
    
    Returns:
        Last analysis results or empty dict if none available
    """
    global _analyzer
    
    if _analyzer is None or not is_analyzer_ready():
        return {"error": "Analyzer not initialized"}
    
    return _analyzer.get_last_results() or {} 