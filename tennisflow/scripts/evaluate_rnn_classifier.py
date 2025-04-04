#!/usr/bin/env python
"""
Evaluate the trained RNN classifier on test data.
Generates evaluation metrics and visualizations for model performance.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

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

def load_test_data(test_data_path, metadata_path=None):
    """
    Load test data from a specified path.
    
    Args:
        test_data_path: Path to the test data file (.npz)
        metadata_path: Path to the metadata file (.json)
        
    Returns:
        X_test: Test data (sequences)
        y_test: Test labels
        metadata: Metadata dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Load test data
    test_data = np.load(test_data_path)
    X_test, y_test = test_data['sequences'], test_data['labels']
    
    # Load metadata if provided
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    
    logger.info(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    logger.info(f"Test label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_test, y_test, metadata

def convert_to_native_types(obj):
    """
    Convert numpy data types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    else:
        return obj

def save_evaluation_results(y_true, y_pred, class_names, output_dir, y_pred_probs):
    """
    Save evaluation results to a JSON file.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): List of class names
        output_dir (str): Directory to save results
        y_pred_probs (np.ndarray): Predicted probabilities
    """
    logger = logging.getLogger(__name__)
    
    # Get unique labels present in the test data
    present_labels = np.unique(y_true).astype(int).tolist()
    present_class_names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in present_labels]
    
    logger.info(f"Note: Test data contains labels {present_labels} (out of {len(class_names)} possible classes)")
    logger.info(f"Using class names for present labels: {present_class_names}")
    
    # Calculate metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    # Classification report with only present labels
    report = classification_report(y_true, y_pred, labels=present_labels, 
                                  target_names=present_class_names, output_dict=True)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, labels=present_labels, target_names=present_class_names)}")
    
    # Prepare results dictionary
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'present_labels': present_labels,
        'present_class_names': present_class_names,
    }
    
    # Convert numpy types to native types for JSON serialization
    results = convert_to_native_types(results)
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved evaluation results to {results_path}")

def create_visualizations(X_test, y_true, y_pred, class_names, output_dir):
    """
    Create visualizations for evaluation results.
    
    Args:
        X_test (np.ndarray): Test data (sequences)
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): List of class names
        output_dir (str): Directory to save visualizations
    """
    logger = logging.getLogger(__name__)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get present labels
    present_labels = np.unique(np.concatenate([y_true, y_pred]))
    present_labels = [int(label) for label in present_labels]
    
    # Limit class_names to those present
    used_class_names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in present_labels]
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=used_class_names,
                yticklabels=used_class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_true, order=present_labels)
    plt.xticks(ticks=range(len(present_labels)), labels=used_class_names)
    plt.title('Class Distribution in Test Data')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # Plot per-class accuracy
    class_acc = {}
    for i in present_labels:
        mask = (y_true == i)
        if np.sum(mask) > 0:
            class_acc[i] = accuracy_score(y_true[mask], y_pred[mask])
    
    plt.figure(figsize=(10, 6))
    classes = list(class_acc.keys())
    accuracies = list(class_acc.values())
    
    class_name_mapping = {i: name for i, name in zip(present_labels, used_class_names)}
    class_names_for_plot = [class_name_mapping.get(c, f"class_{c}") for c in classes]
    
    sns.barplot(x=classes, y=accuracies)
    plt.xticks(ticks=range(len(classes)), labels=class_names_for_plot)
    plt.title('Per-class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate the RNN classifier for tennis shot classification')
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--test-data', required=True, help='Path to the test data (.npz file)')
    parser.add_argument('--output-dir', help='Directory to save evaluation results')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default='info', 
                        help='Logging level')
    parser.add_argument('--metadata', help='Path to the metadata file (.json)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), 'evaluation_results')
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load test data
        logger.info(f"Loading test data from {args.test_data}")
        metadata_path = None
        if args.metadata:
            metadata_path = args.metadata
        else:
            # Try to find metadata in the same directory as test data
            test_dir = os.path.dirname(args.test_data)
            potential_metadata = os.path.join(test_dir, 'metadata.json')
            if os.path.exists(potential_metadata):
                metadata_path = potential_metadata
                logger.info(f"Loaded metadata from {metadata_path}")
        
        X_test, y_test, metadata = load_test_data(args.test_data, metadata_path)
        
        # Load model
        logger.info(f"Loading model from {args.model}")
        model = RNNShotClassifier()
        model.load(args.model)
        
        # Get shot types
        if hasattr(model, 'config') and 'shot_types' in model.config:
            class_names = model.config['shot_types']
        else:
            if metadata and 'class_names' in metadata:
                class_names = metadata['class_names']
            else:
                class_names = ['forehand', 'backhand', 'serve', 'volley', 'neutral']
        
        logger.info(f"Using shot types: {class_names}")
        
        # Remap test labels to match model labels
        unique_test_labels = np.unique(y_test)
        logger.info(f"Original test labels: {unique_test_labels}")
        
        # Check if label mapping is needed
        if 4 in unique_test_labels and len(class_names) < 5:
            # Create a mapping from original labels to model labels
            label_map = {}
            for i, label in enumerate(sorted(unique_test_labels)):
                if i < len(class_names):
                    label_map[label] = i
                else:
                    # Skip labels that don't have a corresponding class_name
                    logger.warning(f"Label {label} doesn't have a corresponding class name, mapping to 0")
                    label_map[label] = 0
            
            logger.info(f"Mapping test labels using: {label_map}")
            y_test_mapped = np.array([label_map.get(y, 0) for y in y_test])
            
            # Update test labels
            y_test = y_test_mapped
            logger.info(f"After mapping, test label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Predict on test data
        logger.info("Predicting on test data...")
        y_pred_indices, y_pred_probs = model.predict(X_test)
        
        # Generate evaluation metrics
        logger.info("Generating evaluation metrics...")
        # Limit class_names to those actually present in the test data
        present_labels = np.unique(y_test)
        used_class_names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in range(len(class_names))]
        
        # Save evaluation results
        save_evaluation_results(y_test, y_pred_indices, used_class_names, args.output_dir, y_pred_probs)
        
        # Create visualizations
        create_visualizations(X_test, y_test, y_pred_indices, used_class_names, args.output_dir)
        
        logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
        return 0
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 