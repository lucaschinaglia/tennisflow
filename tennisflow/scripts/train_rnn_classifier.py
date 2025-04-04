#!/usr/bin/env python
"""
Train the RNN classifier using prepared training data.
Saves the trained model and metadata for use in the tennis analysis pipeline.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
from pathlib import Path
import tensorflow as tf

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.rnn_classifier import RNNShotClassifier

def setup_logging(log_level):
    """Set up logging configuration."""
    logging_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }
    
    logging.basicConfig(
        level=logging_levels.get(log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Suppress TensorFlow logging except errors
    tf.get_logger().setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_training_data(data_dir):
    """
    Load prepared training data from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing training data
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, metadata)
    """
    logger = logging.getLogger(__name__)
    
    train_data_path = os.path.join(data_dir, 'train_data.npz')
    val_data_path = os.path.join(data_dir, 'val_data.npz')
    metadata_path = os.path.join(data_dir, 'metadata.json')
    
    if not os.path.exists(train_data_path):
        logger.error(f"Training data not found at {train_data_path}")
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    logger.info(f"Loading training data from {train_data_path}")
    train_data = np.load(train_data_path)
    X_train = train_data['sequences']
    y_train = train_data['labels']
    
    if os.path.exists(val_data_path):
        logger.info(f"Loading validation data from {val_data_path}")
        val_data = np.load(val_data_path)
        X_val = val_data['sequences']
        y_val = val_data['labels']
    else:
        logger.warning(f"No validation data found at {val_data_path}")
        # Split training data to create validation set
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        logger.info(f"Split training data to create validation set (80/20 split)")
    
    # Load metadata
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    else:
        logger.warning(f"No metadata found at {metadata_path}")
    
    # Log dataset statistics
    logger.info(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")
    
    # Log label distribution
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    logger.info(f"Training label distribution: {dict(zip(train_labels, train_counts))}")
    val_labels, val_counts = np.unique(y_val, return_counts=True)
    logger.info(f"Validation label distribution: {dict(zip(val_labels, val_counts))}")
    
    return X_train, y_train, X_val, y_val, metadata

def main():
    parser = argparse.ArgumentParser(description='Train the RNN classifier for tennis shot classification')
    parser.add_argument('--data-dir', required=True, help='Directory containing prepared training data')
    parser.add_argument('--output-model', help='Path to save the trained model')
    parser.add_argument('--config', help='Path to the model configuration file')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default='info', 
                        help='Logging level')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate for optimizer')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    if args.output_model:
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    else:
        default_model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(default_model_dir, exist_ok=True)
        args.output_model = os.path.join(default_model_dir, 'rnn_classifier.h5')
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Load training data
    try:
        X_train, y_train, X_val, y_val, metadata = load_training_data(args.data_dir)
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return 1
    
    # Extract shot types from metadata
    shot_types = None
    if metadata and 'shot_types' in metadata:
        shot_types = metadata['shot_types']
        logger.info(f"Using shot types from metadata: {shot_types}")
    
    # Update config with command line arguments
    if config is None:
        config = {}
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if shot_types:
        config['shot_types'] = shot_types
    
    # Get sequence length and number of features from data
    seq_length, features_per_frame = X_train.shape[1], X_train.shape[2]
    config['sequence_length'] = seq_length
    config['features_per_frame'] = features_per_frame
    
    # Initialize and train the model
    try:
        logger.info("Initializing the RNN classifier...")
        classifier = RNNShotClassifier(config)
        
        logger.info("Building the model...")
        classifier.build_model()
        
        logger.info("Starting training...")
        history = classifier.train(
            X_train, y_train, 
            X_val, y_val,
            model_path=args.output_model
        )
        
        # Save the model and metadata
        logger.info(f"Saving model to {args.output_model}")
        classifier.save(args.output_model)
        
        # Log training results
        val_acc = max(history.history['val_accuracy'])
        logger.info(f"Training completed with best validation accuracy: {val_acc:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 