#!/usr/bin/env python3
"""
Import and convert data from the tennis_shot_recognition repository.
This script converts the CSV-based pose data to TennisFlow's format for training.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from glob import glob
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reshape_keller_format(df):
    """
    Reshape the flat CSV format from Keller's repository to our 3D format.
    
    Args:
        df: Pandas DataFrame with pose keypoints
        
    Returns:
        numpy array of shape (sequence_length, num_keypoints, 2)
    """
    # Get keypoint columns (exclude shot column)
    keypoint_cols = [col for col in df.columns if col != 'shot']
    
    # Extract data
    sequence = df[keypoint_cols].values
    
    # Count number of keypoints
    num_keypoints = len(keypoint_cols) // 2
    
    # Reshape from (frames, keypoints*2) to (frames, keypoints, 2)
    frames = sequence.shape[0]
    reshaped = np.zeros((frames, num_keypoints, 2))
    
    for i in range(num_keypoints):
        # Get x and y columns
        x_col = keypoint_cols[i*2 + 1]  # x is usually second
        y_col = keypoint_cols[i*2]      # y is usually first
        
        # Extract x and y coordinates
        x_idx = df.columns.get_loc(x_col)
        y_idx = df.columns.get_loc(y_col)
        
        reshaped[:, i, 0] = sequence[:, x_idx]
        reshaped[:, i, 1] = sequence[:, y_idx]
    
    return reshaped

def convert_keller_data(input_dir, output_dir, sequence_length=30):
    """
    Convert Keller dataset to TennisFlow format.
    
    Args:
        input_dir: Directory containing the CSV files
        output_dir: Directory to save the converted data
        sequence_length: Length of sequences to use (default: 30 frames)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob(os.path.join(input_dir, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        return False
    
    sequences = []
    labels = []
    class_mapping = {
        'forehand': 0,
        'backhand': 1,
        'serve': 2,
        'volley': 3,
        'neutral': 4
    }
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        # Extract shot type from filename
        filename = os.path.basename(csv_file)
        shot_type = filename.split('_')[0]
        
        # Skip unknown shot types
        if shot_type not in class_mapping:
            logger.warning(f"Unknown shot type in {filename}, skipping")
            continue
            
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Skip files with wrong format or no data
        if df.empty or 'shot' not in df.columns:
            logger.warning(f"File {filename} has invalid format, skipping")
            continue
        
        # Check if the sequence length is sufficient
        if len(df) < sequence_length:
            logger.warning(f"File {filename} has insufficient frames ({len(df)}), padding")
            # Pad the sequence to reach the desired length
            last_row = df.iloc[-1:].values
            padding = pd.DataFrame(np.repeat(last_row, sequence_length - len(df), axis=0), 
                                  columns=df.columns)
            df = pd.concat([df, padding], ignore_index=True)
        elif len(df) > sequence_length:
            logger.info(f"File {filename} has extra frames ({len(df)}), truncating")
            # Take the middle portion of the sequence
            start = (len(df) - sequence_length) // 2
            df = df.iloc[start:start+sequence_length]
        
        # Convert to our format
        sequence = reshape_keller_format(df)
        
        # Add to dataset
        sequences.append(sequence)
        labels.append(class_mapping[shot_type])
    
    if not sequences:
        logger.error("No valid sequences were found")
        return False
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    logger.info(f"Created dataset with {len(X)} sequences of shape {X.shape[1:]}")
    
    # Split into training and validation
    indices = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Save to npz files
    train_path = os.path.join(output_dir, 'training_data.npz')
    val_path = os.path.join(output_dir, 'validation_data.npz')
    
    np.savez(train_path, sequences=X_train, labels=y_train)
    np.savez(val_path, sequences=X_val, labels=y_val)
    
    # Save metadata
    metadata = {
        'class_names': ['forehand', 'backhand', 'serve', 'volley', 'neutral'],
        'num_classes': 5,
        'dataset_source': 'Keller Tennis Shot Recognition',
        'original_format': 'CSV',
        'sequence_length': sequence_length,
        'num_keypoints': X.shape[2],
        'num_training_samples': len(X_train),
        'num_validation_samples': len(X_val)
    }
    
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training set: {len(X_train)} sequences")
    logger.info(f"Validation set: {len(X_val)} sequences")
    logger.info(f"Data saved to {output_dir}")
    logger.info(f"  - Training data: {train_path}")
    logger.info(f"  - Validation data: {val_path}")
    logger.info(f"  - Metadata: {meta_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert tennis_shot_recognition data to TennisFlow format")
    parser.add_argument("--input-dir", required=True, help="Path to directory containing CSV files")
    parser.add_argument("--output-dir", required=True, help="Path to save converted data")
    parser.add_argument("--sequence-length", type=int, default=30, help="Sequence length (frames)")
    
    args = parser.parse_args()
    
    result = convert_keller_data(
        args.input_dir, 
        args.output_dir,
        args.sequence_length
    )
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 