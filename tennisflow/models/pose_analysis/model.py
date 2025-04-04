"""
Model architecture for tennis pose analysis.
"""
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import logging
from pathlib import Path
import json

# Configure logging
logger = logging.getLogger("tennisflow.models.pose_analysis")

class TennisPoseModel(nn.Module):
    """
    Tennis pose classification model with various backbone options.
    
    Args:
        num_classes: Number of classes to predict
        backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50', 'efficientnet_b0')
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
    """
    def __init__(self, num_classes=4, backbone='resnet50', pretrained=True, dropout=0.5):
        super(TennisPoseModel, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Create backbone
        if backbone.startswith('resnet'):
            if backbone == 'resnet18':
                base_model = models.resnet18(pretrained=pretrained)
                feature_dim = 512
            elif backbone == 'resnet34':
                base_model = models.resnet34(pretrained=pretrained)
                feature_dim = 512
            elif backbone == 'resnet50':
                base_model = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet backbone: {backbone}")
            
            # Remove the classification head
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        elif backbone.startswith('efficientnet'):
            if backbone == 'efficientnet_b0':
                base_model = models.efficientnet_b0(pretrained=pretrained)
                feature_dim = 1280
            else:
                raise ValueError(f"Unsupported EfficientNet backbone: {backbone}")
            
            # Remove the classification head
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        logger.info(f"Created TennisPoseModel with {backbone} backbone, {num_classes} classes")
    
    def forward(self, x):
        """Forward pass through the model."""
        # Pass through backbone
        features = self.backbone(x)
        # Flatten the features
        features = torch.flatten(features, 1)
        # Classification head
        output = self.classifier(features)
        return output
    
    def save(self, path):
        """
        Save the model weights and configuration.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save configuration
        config = {
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
            'model_type': 'TennisPoseModel'
        }
        
        # Save the model
        torch.save({
            'config': config,
            'state_dict': self.state_dict()
        }, path)
        
        logger.info(f"Model saved to {path}")
        return path
    
    @classmethod
    def load(cls, path, device=None):
        """
        Load a model from a saved checkpoint.
        
        Args:
            path: Path to the saved model
            device: Device to load the model onto
        
        Returns:
            Loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint.get('config', {})
        num_classes = config.get('num_classes', 4)
        backbone = config.get('backbone', 'resnet50')
        
        # Create model
        model = cls(num_classes=num_classes, backbone=backbone)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model

def create_tennis_pose_model(config, device=None):
    """
    Create a tennis pose model based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to move the model to
    
    Returns:
        TennisPoseModel
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract parameters from config
    num_classes = len(config.get('classes', ['forehand', 'backhand', 'serve', 'ready_position']))
    backbone = config.get('backbone', 'resnet50')
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.5)
    
    # Create model
    model = TennisPoseModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    )
    
    # Move to device
    model = model.to(device)
    
    return model

def export_model(model, output_path, format='onnx', input_shape=(1, 3, 224, 224)):
    """
    Export model to different formats.
    
    Args:
        model: The model to export
        output_path: Path to save the exported model
        format: Export format ('onnx', 'torchscript')
        input_shape: Input tensor shape for tracing
        
    Returns:
        Path to the exported model
    """
    model.eval()
    
    if format == 'onnx':
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
    elif format == 'torchscript':
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
        
        # Export to TorchScript
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(output_path)
        
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    logger.info(f"Model exported to {output_path} in {format} format")
    return output_path 