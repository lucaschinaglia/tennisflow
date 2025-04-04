#!/usr/bin/env python3

import os
import argparse
import numpy as np
import json
import logging
import glob
import csv
from sklearn.model_selection import train_test_split

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
    parser = argparse.ArgumentParser(description='Combine training data from multiple sources')
    parser.add_argument('--input-dirs', nargs='+', required=True, 
                        help='Input directories containing training data')
    parser.add_argument('--serve-data-dir', type=str,
                        help='Directory containing serve-specific data to augment training')
    parser.add_argument('--output-dir', required=True, 
                        help='Output directory for combined training data')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--balance-classes', action='store_true',
                        help='Balance classes in the training set')
    parser.add_argument('--max-samples-per-class', type=int, default=None,
                        help='Maximum number of samples per class')
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level')
    return parser.parse_args()

def load_training_data(npz_file):
    """Load training data from NPZ file."""
    data = np.load(npz_file)
    return data['sequences'], data['labels']

def load_metadata(json_file):
    """Load metadata from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def load_csv_serves(csv_files, window_size=30):
    """
    Load serve data from CSV files.
    
    Args:
        csv_files: List of CSV files containing serve data
        window_size: Number of frames per sequence
        
    Returns:
        sequences: Array of shape (n_sequences, window_size, n_keypoints, 2)
        labels: Array of shape (n_sequences,) with class label (2 for serves)
    """
    sequences = []
    labels = []
    
    keypoint_names = ["nose", "left_shoulder", "right_shoulder", "left_elbow", 
                     "right_elbow", "left_wrist", "right_wrist", "left_hip", 
                     "right_hip", "left_knee", "right_knee", "left_ankle", 
                     "right_ankle"]
    
    n_keypoints = len(keypoint_names)
    
    for csv_file in csv_files:
        # Read CSV file
        logging.info(f"Loading {csv_file}")
        rows = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                rows.append(row)
        
        # Get number of frames
        n_frames = len(rows)
        
        if n_frames < window_size:
            logging.warning(f"Skipping {csv_file}: insufficient frames ({n_frames} < {window_size})")
            continue
        
        # Extract keypoints
        keypoints = np.zeros((n_frames, n_keypoints, 2))
        
        for i, row in enumerate(rows):
            # Last element is the shot type
            shot_type = row[-1]
            
            # Extract keypoints
            for j in range(n_keypoints):
                keypoints[i, j, 0] = float(row[j*2])    # y coordinate
                keypoints[i, j, 1] = float(row[j*2+1])  # x coordinate
        
        # Create sequences of window_size frames
        for i in range(0, n_frames - window_size + 1, window_size):
            sequence = keypoints[i:i+window_size]
            if len(sequence) == window_size:
                sequences.append(sequence)
                # Serve is class label 2
                labels.append(2)
    
    if sequences:
        return np.array(sequences), np.array(labels)
    else:
        return np.empty((0, window_size, n_keypoints, 2)), np.empty((0,), dtype=int)

def balance_classes(X, y, max_samples_per_class=None):
    """
    Balance classes by random sampling.
    
    Args:
        X: Input sequences
        y: Labels
        max_samples_per_class: Maximum number of samples per class
        
    Returns:
        Balanced X and y
    """
    unique_classes = np.unique(y)
    class_counts = {c: np.sum(y == c) for c in unique_classes}
    
    logging.info(f"Original class distribution: {class_counts}")
    
    if max_samples_per_class is None:
        # Use the minimum count as the target
        max_samples_per_class = min(class_counts.values())
    
    balanced_X = []
    balanced_y = []
    
    for c in unique_classes:
        indices = np.where(y == c)[0]
        
        # Skip if no samples for this class
        if len(indices) == 0:
            continue
            
        # Limit samples per class
        if len(indices) > max_samples_per_class:
            indices = np.random.choice(indices, max_samples_per_class, replace=False)
        
        balanced_X.append(X[indices])
        balanced_y.append(y[indices])
    
    # Concatenate and shuffle
    balanced_X = np.concatenate(balanced_X, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)
    
    # Shuffle
    indices = np.arange(len(balanced_y))
    np.random.shuffle(indices)
    balanced_X = balanced_X[indices]
    balanced_y = balanced_y[indices]
    
    updated_class_counts = {c: np.sum(balanced_y == c) for c in unique_classes}
    logging.info(f"Balanced class distribution: {updated_class_counts}")
    
    return balanced_X, balanced_y

def combine_datasets(input_dirs, serve_data_dir=None):
    """
    Combine datasets from multiple directories.
    
    Args:
        input_dirs: List of input directories
        serve_data_dir: Directory containing serve-specific data
        
    Returns:
        sequences: Combined sequences
        labels: Combined labels
        metadata: Combined metadata
    """
    all_sequences = []
    all_labels = []
    all_metadata = {}
    
    # Load data from input directories
    for input_dir in input_dirs:
        npz_file = os.path.join(input_dir, 'training_data.npz')
        json_file = os.path.join(input_dir, 'metadata.json')
        
        if not os.path.exists(npz_file) or not os.path.exists(json_file):
            logging.warning(f"Skipping {input_dir}: missing training_data.npz or metadata.json")
            continue
        
        sequences, labels = load_training_data(npz_file)
        metadata = load_metadata(json_file)
        
        logging.info(f"Loaded {len(sequences)} sequences from {input_dir}")
        logging.info(f"Class distribution: {np.bincount(labels)}")
        
        all_sequences.append(sequences)
        all_labels.append(labels)
        
        # Update metadata
        if not all_metadata:
            all_metadata = metadata
        else:
            # Ensure class_names are consistent
            if 'class_names' in metadata and 'class_names' in all_metadata:
                all_metadata['class_names'] = list(set(all_metadata['class_names'] + metadata['class_names']))
    
    # Load serve data if provided
    if serve_data_dir and os.path.exists(serve_data_dir):
        csv_files = glob.glob(os.path.join(serve_data_dir, '*.csv'))
        
        if csv_files:
            logging.info(f"Loading {len(csv_files)} serve CSV files from {serve_data_dir}")
            serve_sequences, serve_labels = load_csv_serves(csv_files)
            
            if len(serve_sequences) > 0:
                logging.info(f"Loaded {len(serve_sequences)} serve sequences")
                all_sequences.append(serve_sequences)
                all_labels.append(serve_labels)
    
    # Combine all datasets
    if all_sequences:
        combined_sequences = np.concatenate(all_sequences, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        logging.info(f"Combined dataset: {len(combined_sequences)} sequences")
        logging.info(f"Combined class distribution: {np.bincount(combined_labels)}")
        
        return combined_sequences, combined_labels, all_metadata
    else:
        logging.error("No valid datasets found")
        return None, None, None

def save_datasets(train_sequences, train_labels, val_sequences, val_labels, metadata, output_dir):
    """
    Save training and validation datasets.
    
    Args:
        train_sequences: Training sequences
        train_labels: Training labels
        val_sequences: Validation sequences
        val_labels: Validation labels
        metadata: Metadata dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    np.savez(
        os.path.join(output_dir, 'training_data.npz'),
        sequences=train_sequences,
        labels=train_labels
    )
    
    # Save validation data
    np.savez(
        os.path.join(output_dir, 'validation_data.npz'),
        sequences=val_sequences,
        labels=val_labels
    )
    
    # Update metadata
    metadata['num_training'] = len(train_sequences)
    metadata['num_validation'] = len(val_sequences)
    metadata['class_distribution_train'] = {i: int(count) for i, count in enumerate(np.bincount(train_labels))}
    metadata['class_distribution_val'] = {i: int(count) for i, count in enumerate(np.bincount(val_labels))}
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved dataset to {output_dir}")
    logging.info(f"Training sequences: {len(train_sequences)}")
    logging.info(f"Validation sequences: {len(val_sequences)}")

def main():
    args = parse_args()
    setup_logging(args.log_level.upper())
    
    # Combine datasets
    sequences, labels, metadata = combine_datasets(args.input_dirs, args.serve_data_dir)
    
    if sequences is None:
        return
    
    # Balance classes if requested
    if args.balance_classes:
        sequences, labels = balance_classes(sequences, labels, args.max_samples_per_class)
    
    # Split into training and validation sets
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=args.validation_split, stratify=labels, random_state=42
    )
    
    # Save datasets
    save_datasets(train_sequences, train_labels, val_sequences, val_labels, metadata, args.output_dir)

if __name__ == '__main__':
    main() 