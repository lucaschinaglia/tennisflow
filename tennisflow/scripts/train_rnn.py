#!/usr/bin/env python
"""
RNN Classifier Training Script

This script trains an RNN (GRU or LSTM) classifier on processed keypoint sequences
for tennis shot type classification.
"""

import os
import sys
import numpy as np
import logging
import argparse
import yaml
import json
import glob
from pathlib import Path
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import local modules
from src.classification.rnn_classifier import RNNClassifier, create_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_rnn")

def load_data(data_dir: str) -> tuple:
    """
    Load and preprocess data for training.
    
    Args:
        data_dir: Directory containing processed sequences
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, class_names)
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Find all subdirectories (shot types)
    shot_types = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    
    if not shot_types:
        logger.error(f"No shot type directories found in {data_dir}")
        return None
    
    logger.info(f"Found shot types: {shot_types}")
    
    # Load sequences and labels
    sequences = []
    labels = []
    
    for i, shot_type in enumerate(shot_types):
        shot_dir = os.path.join(data_dir, shot_type)
        
        # Find all .npy files
        sequence_files = glob.glob(os.path.join(shot_dir, "*.npy"))
        
        if not sequence_files:
            logger.warning(f"No sequence files found for shot type '{shot_type}'")
            continue
        
        logger.info(f"Loading {len(sequence_files)} sequences for shot type '{shot_type}'")
        
        # Load each sequence
        for seq_file in sequence_files:
            try:
                # Load the sequence
                sequence = np.load(seq_file)
                
                # Check if the sequence has the expected shape
                if len(sequence.shape) != 3:
                    logger.warning(f"Skipping {seq_file}: unexpected shape {sequence.shape}")
                    continue
                
                # Add to dataset
                sequences.append(sequence)
                labels.append(i)  # Use index as label
                
            except Exception as e:
                logger.error(f"Error loading {seq_file}: {e}")
    
    if not sequences:
        logger.error("No valid sequences loaded")
        return None
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    # Reshape sequences to match model input (flatten keypoints)
    n_samples, n_frames, n_keypoints, n_coords = X.shape
    X = X.reshape(n_samples, n_frames, n_keypoints * n_coords)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert labels to one-hot encoding
    n_classes = len(shot_types)
    y_train_onehot = to_categorical(y_train, num_classes=n_classes)
    y_val_onehot = to_categorical(y_val, num_classes=n_classes)
    
    logger.info(f"Dataset prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    logger.info(f"Input shape: {X_train.shape[1:]} (sequence_length, features)")
    
    return X_train, y_train_onehot, X_val, y_val_onehot, shot_types, y_train, y_val

def train_model(
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    class_names, 
    config_path: str,
    output_dir: str
):
    """
    Train the RNN classifier.
    
    Args:
        X_train: Training data
        y_train: Training labels (one-hot encoded)
        X_val: Validation data
        y_val: Validation labels (one-hot encoded)
        class_names: Names of classes
        config_path: Path to configuration file
        output_dir: Directory to save the model and results
    """
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = {}
    
    # Get model parameters
    rnn_config = config.get("rnn_classifier", {})
    model_type = rnn_config.get("model_type", "gru")
    hidden_size = rnn_config.get("hidden_size", 128)
    num_layers = rnn_config.get("num_layers", 2)
    dropout = rnn_config.get("dropout", 0.2)
    learning_rate = rnn_config.get("learning_rate", 0.001)
    
    # Get training parameters
    batch_size = rnn_config.get("batch_size", 32)
    epochs = rnn_config.get("epochs", 50)
    patience = rnn_config.get("patience", 10)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for this training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_type}_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create tensorboard log directory
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get model dimensions
    sequence_length, feature_dim = X_train.shape[1:]
    num_classes = len(class_names)
    
    # Create model
    classifier = RNNClassifier(
        model_path=None,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        num_classes=num_classes,
        model_type=model_type,
        classes=class_names
    )
    
    # Build the model
    model = classifier.build_model()
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, "model_{epoch:02d}_{val_accuracy:.4f}.h5"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    logger.info(f"Starting training with {model_type} model")
    logger.info(f"Model parameters: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
    logger.info(f"Training parameters: batch_size={batch_size}, epochs={epochs}, patience={patience}")
    
    history = classifier.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Save the final model
    final_model_path = os.path.join(run_dir, "final_model.h5")
    classifier.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save class names
    class_names_path = os.path.join(run_dir, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    
    # Create and save training plots
    create_training_plots(history, run_dir)
    
    # Evaluate model on the validation set
    evaluate_model(classifier, X_val, y_val, class_names, run_dir)
    
    return final_model_path

def create_training_plots(history, output_dir):
    """
    Create and save training history plots.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
    """
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'), dpi=300)
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'), dpi=300)

def evaluate_model(classifier, X_val, y_val, class_names, output_dir):
    """
    Evaluate the model and save the results.
    
    Args:
        classifier: Trained classifier
        X_val: Validation data
        y_val: Validation labels (one-hot encoded)
        class_names: Names of classes
        output_dir: Directory to save results
    """
    # Get predictions
    y_pred_proba = classifier.predict(X_val)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=class_names)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Save text report
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f"Classification Report:\n\n{report}\n\n")
        f.write(f"Confusion Matrix:\n\n{conf_matrix}\n")
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    logger.info(f"Classification Report:\n{report}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Train RNN classifier for tennis shot classification")
    parser.add_argument("--data", "-d", type=str, required=True, help="Directory containing processed sequences")
    parser.add_argument("--config", "-c", type=str, default="../config.yaml", help="Path to configuration file")
    parser.add_argument("--output", "-o", type=str, default="../models", help="Output directory for models and results")
    args = parser.parse_args()
    
    # Load data
    data_result = load_data(args.data)
    
    if data_result is None:
        logger.error("Failed to load data")
        return
    
    X_train, y_train, X_val, y_val, class_names, y_train_raw, y_val_raw = data_result
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train model
    model_path = train_model(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        class_names, 
        args.config,
        args.output
    )
    
    logger.info(f"Training completed. Model saved to {model_path}")
    
if __name__ == "__main__":
    main() 