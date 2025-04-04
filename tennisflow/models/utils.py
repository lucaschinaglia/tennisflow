"""
Utility functions for model training and inference.
"""
import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger("tennisflow.models")

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def load_json(file_path):
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Loaded JSON data as dictionary
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait after min has been hit
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' for monitoring metrics that should decrease or increase
    """
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric):
        score = metric
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            return True
        return False

def plot_learning_curves(train_values, val_values, y_label, save_path=None):
    """
    Plot training and validation learning curves.
    
    Args:
        train_values: List of training metrics
        val_values: List of validation metrics
        y_label: Label for the y-axis
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_values, label=f'Train {y_label}')
    plt.plot(val_values, label=f'Validation {y_label}')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(f'Training and Validation {y_label}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Learning curve saved to {save_path}")
    
    plt.close() 