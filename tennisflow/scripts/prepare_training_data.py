#!/usr/bin/env python
"""
Prepare training data by combining CSV files from multiple shot sequences
into a single dataset for training the RNN classifier.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Fix the import path to use the correct path
from tennisflow.scripts.import_keller_data import reshape_keller_format

def setup_logging(log_level="info"):
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

def load_sequence_data(npz_file):
    """
    Load sequence data from .npz file.
    
    Args:
        npz_file (str): Path to the .npz file
        
    Returns:
        tuple: (sequences, labels, metadata)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {npz_file}")
    
    data = np.load(npz_file)
    sequences = data['sequences']
    labels = data['labels']
    
    # Load metadata if available
    metadata_file = os.path.join(os.path.dirname(npz_file), 'metadata.json')
    metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    return sequences, labels, metadata

def combine_datasets(input_dirs):
    """
    Combine multiple datasets into a single dataset.
    
    Args:
        input_dirs (list): List of directories containing training data
        
    Returns:
        tuple: (combined_sequences, combined_labels, combined_metadata)
    """
    logger = logging.getLogger(__name__)
    combined_sequences = []
    combined_labels = []
    shot_types = set()
    
    # Keep track of label mappings
    label_mappings = {}
    
    # Process each input directory
    for input_dir in input_dirs:
        train_file = os.path.join(input_dir, 'training_data.npz')
        val_file = os.path.join(input_dir, 'validation_data.npz')
        metadata_file = os.path.join(input_dir, 'metadata.json')
        
        # Load metadata
        metadata = None
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if 'shot_types' in metadata:
                    for shot_type in metadata['shot_types']:
                        shot_types.add(shot_type)
        
        # Load training data
        if os.path.exists(train_file):
            sequences, labels, _ = load_sequence_data(train_file)
            combined_sequences.append(sequences)
            combined_labels.append(labels)
            logger.info(f"Added {len(sequences)} training sequences from {train_file}")
        
        # Load validation data
        if os.path.exists(val_file):
            sequences, labels, _ = load_sequence_data(val_file)
            combined_sequences.append(sequences)
            combined_labels.append(labels)
            logger.info(f"Added {len(sequences)} validation sequences from {val_file}")
    
    # Concatenate all sequences and labels
    all_sequences = np.concatenate(combined_sequences, axis=0)
    all_labels = np.concatenate(combined_labels, axis=0)
    
    # Create combined metadata
    combined_metadata = {
        'shot_types': sorted(list(shot_types)),
        'label_mapping': label_mappings,
        'num_sequences': len(all_sequences)
    }
    
    return all_sequences, all_labels, combined_metadata

def save_prepared_data(sequences, labels, metadata, output_dir, test_split=0.2):
    """
    Split and save the combined data into training and validation sets.
    
    Args:
        sequences (np.ndarray): Combined sequence data
        labels (np.ndarray): Combined labels
        metadata (dict): Combined metadata
        output_dir (str): Directory to save the prepared data
        test_split (float): Fraction of data to use for validation
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=test_split, random_state=42, stratify=labels
    )
    
    # Log dataset statistics
    logger.info(f"Training data: {len(X_train)} sequences")
    logger.info(f"Validation data: {len(X_val)} sequences")
    
    # Log label distribution
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    logger.info(f"Training label distribution: {dict(zip(train_labels, train_counts))}")
    val_labels, val_counts = np.unique(y_val, return_counts=True)
    logger.info(f"Validation label distribution: {dict(zip(val_labels, val_counts))}")
    
    # Save training data
    train_file = os.path.join(output_dir, 'train_data.npz')
    np.savez(train_file, sequences=X_train, labels=y_train)
    logger.info(f"Saved training data to {train_file}")
    
    # Save validation data
    val_file = os.path.join(output_dir, 'val_data.npz')
    np.savez(val_file, sequences=X_val, labels=y_val)
    logger.info(f"Saved validation data to {val_file}")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.json')
    metadata.update({
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'train_distribution': {str(k): int(v) for k, v in zip(train_labels, train_counts)},
        'val_distribution': {str(k): int(v) for k, v in zip(val_labels, val_counts)},
    })
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare training data for the RNN classifier')
    parser.add_argument('--input-dirs', nargs='+', required=True, help='Directories containing data to combine')
    parser.add_argument('--output-dir', required=True, help='Directory to save prepared data')
    parser.add_argument('--test-split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default='info',
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Combine datasets
        logger.info(f"Combining data from {len(args.input_dirs)} directories")
        sequences, labels, metadata = combine_datasets(args.input_dirs)
        
        # Save prepared data
        logger.info(f"Saving prepared data to {args.output_dir}")
        save_prepared_data(sequences, labels, metadata, args.output_dir, args.test_split)
        
        logger.info("Data preparation completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 