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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate, GlobalAveragePooling1D, Reshape

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
    parser = argparse.ArgumentParser(description='Train Model 4 - Enhanced RNN with Static Poses')
    parser.add_argument('--data-dir', required=True, 
                        help='Directory containing the combined training data')
    parser.add_argument('--static-poses-dir', required=True,
                        help='Directory containing static pose data')
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
    parser.add_argument('--lstm-units', type=int, default=96,
                        help='Number of units in LSTM layers (default from best Model 3)')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional LSTM')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level')
    parser.add_argument('--comparison-report', type=str, default='model_comparison_report.txt',
                        help='Path to output comparison report')
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

def load_static_poses(static_poses_dir, shot_types):
    """
    Load static poses for each shot type.
    
    Args:
        static_poses_dir: Directory containing static pose data
        shot_types: List of shot type names
    
    Returns:
        Dictionary mapping shot types to representative keypoint data
    """
    static_poses = {}
    
    for shot_type in shot_types:
        # Convert to lowercase and handle 'neutral' special case
        folder_name = shot_type.lower()
        if folder_name == 'neutral':
            folder_name = 'ready_position'
            
        json_path = os.path.join(static_poses_dir, f"{folder_name}.json")
        
        if not os.path.exists(json_path):
            logging.warning(f"Static pose data not found for {shot_type} at {json_path}")
            continue
            
        logging.info(f"Loading static pose data for {shot_type} from {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                pose_data = json.load(f)
                
            # Extract keypoints from each annotation
            keypoints_list = []
            
            for annotation in pose_data:
                if 'keypoints' in annotation:
                    # Keypoints are stored as [x1, y1, v1, x2, y2, v2, ...]
                    keypoints = annotation['keypoints']
                    
                    # Reshape to [x, y, v] format and filter to only include points with visibility
                    kp_array = np.array(keypoints).reshape(-1, 3)
                    visible_kp = kp_array[kp_array[:, 2] > 0][:, :2]  # Only x,y for visible points
                    
                    if len(visible_kp) > 0:
                        # Normalize coordinates
                        x_min, y_min = visible_kp.min(axis=0)
                        x_max, y_max = visible_kp.max(axis=0)
                        
                        width = max(1, x_max - x_min)
                        height = max(1, y_max - y_min)
                        
                        normalized_kp = (visible_kp - [x_min, y_min]) / [width, height]
                        keypoints_list.append(normalized_kp)
            
            if keypoints_list:
                # Store representative keypoints
                static_poses[shot_type] = keypoints_list
                logging.info(f"Loaded {len(keypoints_list)} static poses for {shot_type}")
            else:
                logging.warning(f"No valid keypoints found for {shot_type}")
                
        except Exception as e:
            logging.error(f"Error loading static pose data for {shot_type}: {e}")
    
    return static_poses

def preprocess_static_poses(static_poses, n_keypoints=13):
    """
    Preprocess static poses to ensure consistent format.
    
    Args:
        static_poses: Dictionary of static poses by shot type
        n_keypoints: Number of keypoints to standardize to
        
    Returns:
        Processed static poses
    """
    processed_poses = {}
    
    for shot_type, poses_list in static_poses.items():
        # Process each pose to ensure it has exactly n_keypoints
        standardized_poses = []
        
        for pose in poses_list:
            # If pose has fewer than n_keypoints, pad with zeros
            if len(pose) < n_keypoints:
                padded_pose = np.zeros((n_keypoints, 2))
                padded_pose[:len(pose)] = pose
                standardized_poses.append(padded_pose)
            # If pose has more than n_keypoints, truncate
            elif len(pose) > n_keypoints:
                standardized_poses.append(pose[:n_keypoints])
            else:
                standardized_poses.append(pose)
                
        # Store a sample of standardized poses (up to 10)
        n_samples = min(10, len(standardized_poses))
        processed_poses[shot_type] = np.array(standardized_poses[:n_samples])
    
    return processed_poses

def compute_pose_similarity_features(sequence, static_poses, shot_types):
    """
    Compute similarity features between a sequence and static poses.
    
    Args:
        sequence: Sequence of poses (seq_length, n_keypoints, coords)
        static_poses: Dictionary of static poses by shot type
        shot_types: List of shot type names
        
    Returns:
        Similarity features (seq_length, n_shot_types)
    """
    seq_length = sequence.shape[0]
    n_shot_types = len(shot_types)
    similarity_features = np.zeros((seq_length, n_shot_types))
    
    # For each frame in the sequence
    for i in range(seq_length):
        frame = sequence[i]  # (n_keypoints, coords)
        
        # For each shot type
        for j, shot_type in enumerate(shot_types):
            if shot_type not in static_poses:
                continue
                
            # Compute similarity with each reference pose for this shot type
            reference_poses = static_poses[shot_type]
            
            similarities = []
            for ref_pose in reference_poses:
                # Simple Euclidean distance-based similarity
                try:
                    # Reshape if necessary to match dimensions
                    if frame.shape != ref_pose.shape:
                        min_keypoints = min(frame.shape[0], ref_pose.shape[0])
                        frame_subset = frame[:min_keypoints]
                        ref_pose_subset = ref_pose[:min_keypoints]
                        dist = np.mean(np.sqrt(np.sum((frame_subset - ref_pose_subset)**2, axis=1)))
                    else:
                        dist = np.mean(np.sqrt(np.sum((frame - ref_pose)**2, axis=1)))
                    
                    # Convert distance to similarity (1 for identical, 0 for very different)
                    similarity = np.exp(-dist * 5.0)  # Scale factor to make differences more pronounced
                    similarities.append(similarity)
                except Exception as e:
                    logging.warning(f"Error computing similarity: {e}")
                    similarities.append(0.0)
            
            # Use the highest similarity among all reference poses
            if similarities:
                similarity_features[i, j] = max(similarities)
    
    return similarity_features

def augment_data_with_static_poses(X, static_poses, shot_types):
    """
    Augment input sequences with pose similarity features.
    
    Args:
        X: Input sequences (n_samples, seq_length, n_keypoints, coords)
        static_poses: Dictionary of static poses by shot type
        shot_types: List of shot type names
        
    Returns:
        Similarity features for each sequence (n_samples, seq_length, n_shot_types)
    """
    n_samples = X.shape[0]
    seq_length = X.shape[1]
    n_shot_types = len(shot_types)
    
    similarity_features = np.zeros((n_samples, seq_length, n_shot_types))
    
    for i in range(n_samples):
        sequence = X[i]
        similarity_features[i] = compute_pose_similarity_features(sequence, static_poses, shot_types)
    
    return similarity_features

def build_model_4(input_shape, num_classes, similarity_feature_shape, config):
    """
    Build Model 4 - Enhanced RNN with static pose similarity features.
    
    Args:
        input_shape: Shape of the input sequences
        num_classes: Number of classes to predict
        similarity_feature_shape: Shape of the similarity features
        config: Model configuration parameters
        
    Returns:
        Compiled Model 4
    """
    # Parameters
    lstm_units = config['lstm_units']
    dropout_rate = config['dropout_rate']
    use_bidirectional = config.get('bidirectional', False)
    learning_rate = config['learning_rate']
    
    # Main sequence input
    sequence_input = Input(shape=input_shape, name='sequence_input')
    
    # Reshape if needed
    if len(input_shape) == 3:  # (frames, keypoints, coords)
        frames, keypoints, coords = input_shape
        flat_dim = keypoints * coords
        x = Reshape((frames, flat_dim))(sequence_input)
    else:
        x = sequence_input
    
    # Similarity features input
    similarity_input = Input(shape=similarity_feature_shape, name='similarity_input')
    
    # Process sequence with LSTM
    if use_bidirectional:
        x = tf.keras.layers.Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    else:
        x = LSTM(lstm_units, return_sequences=True)(x)
    
    x = Dropout(dropout_rate)(x)
    
    # Concatenate similarity features with sequence features
    x = Concatenate(axis=2)([x, similarity_input])
    
    # Second LSTM layer
    if use_bidirectional:
        x = tf.keras.layers.Bidirectional(LSTM(lstm_units))(x)
    else:
        x = LSTM(lstm_units)(x)
    
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=[sequence_input, similarity_input], outputs=output)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary(print_fn=logging.info)
    
    return model

def train_model_4(model, X_train, similarity_features_train, y_train, 
                 X_val, similarity_features_val, y_val, config, model_path):
    """
    Train Model 4 with sequence data and similarity features.
    
    Args:
        model: Model 4 instance
        X_train, similarity_features_train, y_train: Training data
        X_val, similarity_features_val, y_val: Validation data
        config: Training configuration
        model_path: Path to save the best model
        
    Returns:
        Training history
    """
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=config['patience'],
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Calculate class weights if enabled
    class_weight_dict = None
    if config['use_class_weights']:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        logging.info(f"Using class weights: {class_weight_dict}")
    
    # Train model
    history = model.fit(
        {'sequence_input': X_train, 'similarity_input': similarity_features_train},
        y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(
            {'sequence_input': X_val, 'similarity_input': similarity_features_val},
            y_val
        ),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=2
    )
    
    return history

def evaluate_model_4(model, X_test, similarity_features_test, y_test, class_names, output_dir):
    """
    Evaluate Model 4 and save results.
    
    Args:
        model: Trained Model 4
        X_test, similarity_features_test, y_test: Test data
        class_names: Names of the classes
        output_dir: Directory to save evaluation results
    
    Returns:
        Evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_probs = model.predict({'sequence_input': X_test, 'similarity_input': similarity_features_test})
    y_pred = np.argmax(y_probs, axis=1)
    
    # Calculate metrics
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=class_names))
    
    # Calculate accuracy
    accuracy = report['accuracy']
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
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
    
    # Save predictions for further analysis
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'true_class': [class_names[y] for y in y_test],
        'predicted_class': [class_names[y] for y in y_pred]
    })
    
    # Add prediction probabilities for each class
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = y_probs[:, i]
    
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Calculate F1 scores for specific classes
    f1_scores = {}
    for i, class_name in enumerate(class_names):
        if class_name.lower() in ['volley', 'backhand', 'forehand', 'serve', 'neutral']:
            f1_scores[class_name] = report[class_name]['f1-score']
    
    # Return key metrics
    return {
        'accuracy': accuracy,
        'f1_scores': f1_scores,
        'report': report,
        'confusion_matrix': cm
    }

def update_comparison_report(metrics, model_params, epochs, report_path):
    """
    Update the model comparison report with Model 4 results.
    
    Args:
        metrics: Evaluation metrics
        model_params: Model parameters count
        epochs: Number of training epochs
        report_path: Path to the report file
    """
    # Read existing report if it exists
    report_text = ""
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_text = f.read()
    
    # Extract F1 scores for specific classes
    f1_scores = metrics['f1_scores']
    volley_f1 = f1_scores.get('volley', 0)
    backhand_f1 = f1_scores.get('backhand', 0)
    
    # Create model 4 entry
    model4_entry = f"""
Model 4 (Static Pose Enhanced LSTM):
  - Accuracy: {metrics['accuracy']*100:.2f}%
  - Volley F1-score: {volley_f1:.2f}
  - Backhand F1-score: {backhand_f1:.2f}
  - Training epochs: {epochs} (early stopping)
  - Parameters: {model_params//1000}K
"""
    
    # Check if we need to add Model 4 or if it already exists
    if "Model 4" not in report_text:
        # Find the position to insert Model 4
        conclusion_pos = report_text.find("Conclusion:")
        if conclusion_pos > -1:
            # Insert before conclusion
            updated_report = report_text[:conclusion_pos] + model4_entry + "\n" + report_text[conclusion_pos:]
        else:
            # Append to the end
            updated_report = report_text + model4_entry
    else:
        # Replace existing Model 4 entry
        import re
        updated_report = re.sub(r"Model 4.*?Parameters:.*?K\n", model4_entry, report_text, flags=re.DOTALL)
    
    # Write the updated report
    with open(report_path, 'w') as f:
        f.write(updated_report)
    
    logging.info(f"Updated model comparison report at {report_path}")

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
    plt.title('Model 4 Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model 4 Loss')
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

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logging.info("Starting training of Model 4 with static pose integration")
    
    # Load sequence data
    X_train, y_train, X_val, y_val, metadata = load_data(args.data_dir)
    class_names = metadata.get('class_names', [f'class_{i}' for i in range(len(np.unique(y_train)))])
    logging.info(f"Class names: {class_names}")
    
    # Load static poses
    static_poses = load_static_poses(args.static_poses_dir, class_names)
    logging.info(f"Loaded static poses for {len(static_poses)} shot types")
    
    # Preprocess static poses
    processed_poses = preprocess_static_poses(static_poses)
    
    # Generate similarity features
    similarity_features_train = augment_data_with_static_poses(X_train, processed_poses, class_names)
    similarity_features_val = augment_data_with_static_poses(X_val, processed_poses, class_names)
    
    logging.info(f"Generated similarity features with shape: {similarity_features_train.shape}")
    
    # Model configuration
    config = {
        'lstm_units': args.lstm_units,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'use_class_weights': args.use_class_weights,
        'bidirectional': args.bidirectional,
        'augment_data': args.use_augmentation,
    }
    
    # Build Model 4
    input_shape = X_train.shape[1:]
    similarity_feature_shape = similarity_features_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    model = build_model_4(input_shape, num_classes, similarity_feature_shape, config)
    
    # Train Model 4
    logging.info("Training Model 4...")
    history = train_model_4(
        model, X_train, similarity_features_train, y_train,
        X_val, similarity_features_val, y_val,
        config, args.output_model
    )
    
    # Plot training history
    output_dir = os.path.dirname(args.output_model)
    plot_training_history(history, output_dir)
    
    # Load the best model for evaluation
    best_model = tf.keras.models.load_model(args.output_model)
    
    # Evaluate on validation set
    metrics = evaluate_model_4(
        best_model, X_val, similarity_features_val, y_val,
        class_names, os.path.join(output_dir, 'validation')
    )
    
    # Count parameters
    model_params = best_model.count_params()
    
    # Get number of epochs actually trained
    epochs_trained = len(history.history['accuracy'])
    
    # Update comparison report
    update_comparison_report(metrics, model_params, epochs_trained, args.comparison_report)
    
    logging.info(f"Model 4 training and evaluation complete. Model saved to {args.output_model}")
    logging.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    
    # Save an overall summary
    summary = {
        'model_name': 'Model 4 - Static Pose Enhanced LSTM',
        'accuracy': float(metrics['accuracy']),
        'f1_scores': {k: float(v) for k, v in metrics['f1_scores'].items()},
        'parameters': int(model_params),
        'epochs_trained': epochs_trained,
        'config': config
    }
    
    with open(os.path.join(output_dir, 'model4_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main() 