"""
Configuration settings for model training and inference.
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = PROJECT_ROOT / "models"
POSE_MODEL_DIR = MODELS_DIR / "output" / "pose_models"
EVENT_MODEL_DIR = MODELS_DIR / "output" / "event_models"
TENSORBOARD_DIR = MODELS_DIR / "tensorboard"

# Ensure output directories exist
POSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Pose model configuration
POSE_CONFIG = {
    # Dataset config
    'data_dir': str(PROJECT_ROOT.parent / "training-set" / "poses"),
    'classes': ['forehand', 'backhand', 'serve', 'ready_position'],
    'img_size': 224,
    'augmentation_level': 'medium',  # 'none', 'light', 'medium', 'heavy'
    
    # Model config
    'backbone': 'resnet50',  # 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
    'pretrained': True,
    'dropout': 0.5,
    
    # Training config
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'seed': 42,
    'patience': 15,  # Early stopping patience
    
    # Output config
    'output_dir': str(POSE_MODEL_DIR),
    'export_formats': ['onnx', 'torchscript']
}

# Event model configuration
EVENT_CONFIG = {
    # Dataset config
    'data_dir': str(PROJECT_ROOT.parent / "training-set" / "events"),
    'classes': ['serve', 'hit', 'bounce', 'net'],
    'clip_len': 16,
    'frame_size': 224,
    'temporal_stride': 2,
    'augmentation_level': 'medium',  # 'none', 'light', 'medium', 'heavy'
    
    # Model config
    'model_type': 'slowfast',  # 'slowfast', 'r2plus1d', 'i3d'
    'pretrained': True,
    'dropout': 0.5,
    
    # Training config
    'batch_size': 8,
    'epochs': 30,
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'seed': 42,
    'patience': 10,  # Early stopping patience
    
    # Output config
    'output_dir': str(EVENT_MODEL_DIR),
    'export_formats': ['onnx', 'torchscript']
}

# Inference configuration
INFERENCE_CONFIG = {
    'pose_model_path': str(POSE_MODEL_DIR / "best_model.pth"),
    'event_model_path': str(EVENT_MODEL_DIR / "best_model.pth"),
    'device': 'cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu',
    'confidence_threshold': 0.7,
    'frame_sample_rate': 5,  # Sample every N frames
    'batch_size': 16,
    'temporal_window': 16,   # Number of frames for event detection
    'use_mock': False        # Set to True for mock implementation
} 