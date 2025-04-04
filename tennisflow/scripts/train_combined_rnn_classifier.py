#!/usr/bin/env python3

import os
import argparse
import logging
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import sys
import pandas as pd
from sklearn import metrics

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tennisflow.src.classification.rnn_classifier import RNNShotClassifier

def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train RNN classifier on combined dataset')
    parser.add_argument('--data-dir', required=True, 
                        help='Directory containing the combined training data')
    parser.add_argument('--output-model', required=True, 
                        help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights to handle class imbalance')
    parser.add_argument('--use-augmentation', action='store_true',
                        help='Use data augmentation for underrepresented classes')
    parser.add_argument('--lstm-units', type=int, default=64,
                        help='Number of units in LSTM layers')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional LSTM')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level')
    return parser.parse_args()

def load_data(data_dir):
    """
    Load training and validation data from the given directory.
    
    Args:
        data_dir: Directory containing training_data.npz, validation_data.npz, and metadata.json
        
    Returns:
        X_train, y_train, X_val, y_val, metadata
    """
    # Training data
    train_data_path = os.path.join(data_dir, 'training_data.npz')
    train_data = np.load(train_data_path)
    X_train, y_train = train_data['sequences'], train_data['labels']
    
    # Validation data
    val_data_path = os.path.join(data_dir, 'validation_data.npz')
    val_data = np.load(val_data_path)
    X_val, y_val = val_data['sequences'], val_data['labels']
    
    # Metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Remap labels to ensure they are consecutive integers starting from 0
    unique_labels = np.unique(np.concatenate([y_train, y_val]))
    logging.info(f"Original unique labels: {unique_labels}")
    
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
    logging.info(f"Label mapping: {label_map}")
    
    # Apply mapping
    y_train = np.array([label_map[y] for y in y_train])
    y_val = np.array([label_map[y] for y in y_val])
    
    # Update class names in metadata if needed
    if 'class_names' in metadata:
        original_class_names = metadata['class_names']
        remapped_class_names = [original_class_names[i] if i < len(original_class_names) else f"class_{i}" 
                               for i in sorted(unique_labels)]
        metadata['class_names'] = remapped_class_names
        metadata['label_map'] = label_map
    
    logging.info(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples")
    logging.info(f"Training class distribution: {np.bincount(y_train)}")
    logging.info(f"Validation class distribution: {np.bincount(y_val)}")
    
    return X_train, y_train, X_val, y_val, metadata

def plot_training_history(history, output_dir):
    """
    Plot and save training history.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Save history to JSON
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)

def evaluate_model(model, X_val, y_val, class_names, output_dir):
    """
    Evaluate model on validation set and save results.
    
    Args:
        model: Trained RNN classifier
        X_val: Validation data
        y_val: Validation labels
        class_names: Names of classes
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred, y_probs = model.predict(X_val)
    
    # Calculate metrics
    report = classification_report(
        y_val, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report
    logging.info("Classification Report:")
    logging.info(classification_report(y_val, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add labels
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save confusion matrix
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
    
    # Plot ROC curve for multiclass
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_val == i).astype(int)
        y_score = y_probs[:, i]
        
        fpr, tpr, _ = metrics.roc_curve(y_true_binary, y_score)
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, 
                 label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    
    # Class distribution bar chart
    plt.figure(figsize=(10, 6))
    class_counts = np.bincount(y_val)
    plt.bar(range(len(class_counts)), class_counts)
    plt.xticks(range(len(class_counts)), class_names)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Validation Class Distribution')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_val,
        'predicted_label': y_pred,
        'true_class': [class_names[y] for y in y_val],
        'predicted_class': [class_names[y] for y in y_pred]
    })
    
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = y_probs[:, i]
    
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

def main():
    args = parse_args()
    setup_logging(args.log_level.upper())
    
    # Load data
    X_train, y_train, X_val, y_val, metadata = load_data(args.data_dir)
    
    # Get class names
    class_names = metadata.get('class_names', ['forehand', 'backhand', 'serve', 'volley', 'neutral'])
    
    # Configure model
    config = {
        'rnn_units': args.lstm_units,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'bidirectional': args.bidirectional,
        'shot_types': class_names,
        'use_class_weights': args.use_class_weights,
        'augment_data': args.use_augmentation,
    }
    
    # Create output directories
    model_dir = os.path.dirname(args.output_model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create evaluation directory
    eval_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize model
    model = RNNShotClassifier(config)
    
    # Train model
    logging.info("Starting model training")
    history = model.train(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        model_path=args.output_model
    )
    
    # Plot training history
    plot_training_history(history, eval_dir)
    
    # Evaluate model
    logging.info("Evaluating model on validation set")
    evaluate_model(model, X_val, y_val, class_names, eval_dir)
    
    # Save model
    model.save(args.output_model)
    logging.info(f"Model saved to {args.output_model}")
    
    # Print out the best validation accuracy
    best_val_acc = max(history.history['val_accuracy'])
    logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Log class distribution
    train_class_dist = {class_names[i]: int(count) for i, count in enumerate(np.bincount(y_train))}
    val_class_dist = {class_names[i]: int(count) for i, count in enumerate(np.bincount(y_val))}
    
    logging.info(f"Training class distribution: {train_class_dist}")
    logging.info(f"Validation class distribution: {val_class_dist}")

if __name__ == '__main__':
    main() 