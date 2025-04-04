"""
This module implements an RNN classifier for tennis shot classification based on pose keypoint sequences.
Inspired by the tennis_shot_recognition project by Antoine Keller.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, Dropout, BatchNormalization, Reshape, Input, Flatten, LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import logging
import json
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

class RNNShotClassifier:
    """RNN-based classifier for tennis shots using pose keypoint sequences."""
    
    def __init__(self, config=None):
        """
        Initialize the RNN classifier.
        
        Args:
            config (dict, optional): Configuration parameters for the classifier.
                If None, default parameters are used.
        """
        self.config = {
            'rnn_units': 64,
            'dense_units': 32,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'shot_types': ['forehand', 'backhand', 'serve', 'volley', 'neutral'],
            'sequence_length': 30,
            'features_per_frame': 26,  # 13 keypoints * 2 coordinates
            'use_class_weights': True,
            'augment_data': False,
            'noise_level': 0.02
        }
        
        if config:
            self.config.update(config)
        
        self.model = None
        self.input_shape = None
        self.num_classes = None
        logger.info("Initialized RNN classifier with config: %s", self.config)
    
    def build_model(self, input_shape, num_classes):
        """
        Build and compile the RNN model architecture.
        
        Args:
            input_shape (tuple): Shape of the input data
            num_classes (int): Number of classes to classify
            
        Returns:
            The compiled Keras model.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Get model parameters from config
        lstm_units = self.config['rnn_units']
        dropout_rate = self.config['dropout_rate']
        use_bidirectional = self.config.get('bidirectional', True)
        
        # Create model
        model = Sequential()
        
        # Handle different input shapes
        if len(input_shape) == 3:  # (frames, keypoints, coords)
            frames, keypoints, coords = input_shape
            flat_dim = keypoints * coords
            # Reshape layer to combine keypoints and coordinates
            model.add(Reshape((frames, flat_dim), input_shape=input_shape))
        elif len(input_shape) == 2:  # (frames, features)
            # No need for reshape, use input shape directly
            model.add(Input(shape=input_shape))
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # LSTM layers
        if use_bidirectional:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
            model.add(Dropout(dropout_rate))
            model.add(Bidirectional(LSTM(lstm_units)))
        else:
            model.add(LSTM(lstm_units, return_sequences=True))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(lstm_units))
        
        # Output layer
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Use CPU only to avoid GPU issues
        with tf.device('/cpu:0'):
            optimizer = self.config['learning_rate']
            if isinstance(optimizer, float):
                opt = tf.keras.optimizers.Adam(learning_rate=optimizer)
            elif optimizer.lower() == 'adam':
                opt = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
            elif optimizer.lower() == 'rmsprop':
                opt = tf.keras.optimizers.RMSprop(learning_rate=self.config['learning_rate'])
            else:
                logger.warning(f"Unknown optimizer: {optimizer}, using Adam")
                opt = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
            
            model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        self.model = model
        logger.info("Model built with %d parameters", model.count_params())
        model.summary(print_fn=logger.info)
        
        return model
    
    def preprocess_input(self, X):
        """
        Preprocess input data for the model.
        
        Args:
            X (np.ndarray): Input data, can be:
                - (n_samples, sequence_length, n_keypoints, coords)
                - (n_samples, sequence_length, flattened_features)
                
        Returns:
            np.ndarray: Preprocessed data with consistent format
        """
        if X is None:
            logger.warning("Input is None, returning empty array")
            return np.array([])
            
        # Check if X is empty or has invalid dimensions
        if len(X) == 0 or X.size == 0:
            logger.warning(f"Input is empty with shape {X.shape}, returning empty array")
            return np.array([])
        
        # Get expected input shape based on model configuration
        expected_input_shape = None
        if hasattr(self.model, 'inputs') and self.model.inputs:
            try:
                # Try to get shape as a list if available
                if hasattr(self.model.inputs[0].shape, 'as_list'):
                    expected_input_shape = self.model.inputs[0].shape.as_list()[1:]  # Remove batch dimension
                # Otherwise, use string parsing or type conversion
                else:
                    # Get as string and parse
                    shape_str = str(self.model.inputs[0].shape)
                    logger.info(f"Model input shape string: {shape_str}")
                    
                    # Check for special 4D shape (batch, seq, keypoints, coords)
                    if '13, 2' in shape_str:
                        expected_input_shape = [30, 13, 2]  # Standard pose shape
                    
                logger.info(f"Model expects input shape: {expected_input_shape}")
            except Exception as e:
                logger.warning(f"Error determining expected input shape: {e}")
            
        # Handle single sequence (no batch dimension)
        if len(X.shape) == 2 or (len(X.shape) == 3 and X.shape[-1] == 2):
            X = np.expand_dims(X, axis=0)
            logger.info(f"Added batch dimension, new shape: {X.shape}")
        
        # Check if model expects 4D input (batch, seq_len, n_keypoints, coords)
        if expected_input_shape and len(expected_input_shape) == 3:
            # Model expects (batch, seq_len, n_keypoints, coords)
            if len(X.shape) == 3:
                # Input is (batch, seq_len, flattened_features)
                # Need to reshape to (batch, seq_len, n_keypoints, coords)
                n_samples, seq_len, flat_dim = X.shape
                n_keypoints = expected_input_shape[1]  # From expected shape
                coords = expected_input_shape[2]  # From expected shape (should be 2)
                
                # Check if we have enough features to create the expected keypoints
                if flat_dim >= n_keypoints * coords:
                    # Reshape the features to (batch, seq_len, n_keypoints, coords)
                    try:
                        n_features_to_use = n_keypoints * coords
                        # Create new array with expected shape
                        reshaped_X = np.zeros((n_samples, seq_len, n_keypoints, coords))
                        for i in range(n_samples):
                            for j in range(seq_len):
                                # Reshape flattened features to (n_keypoints, coords)
                                features = X[i, j, :n_features_to_use]
                                reshaped_X[i, j] = features.reshape(n_keypoints, coords)
                        
                        logger.info(f"Reshaped input from {X.shape} to {reshaped_X.shape}")
                        return reshaped_X
                    except Exception as e:
                        logger.warning(f"Error reshaping input: {e}")
            elif len(X.shape) == 4 and X.shape[2:] == (expected_input_shape[1], expected_input_shape[2]):
                # Input already has the right shape
                logger.info(f"Input already has the expected shape: {X.shape}")
                return X
        
        # Handle batch of sequences with points and coords (n_samples, sequence_length, n_keypoints, 2)
        if len(X.shape) == 4 and X.shape[-1] == 2:
            logger.info(f"Batch input has shape {X.shape}, flattening keypoints")
            n_samples, seq_len, n_keypoints, coords = X.shape
            X = X.reshape(n_samples, seq_len, n_keypoints * coords)
            
        # Ensure we have a valid shape before continuing
        if len(X.shape) < 3 or X.shape[1] == 0 or X.shape[2] == 0:
            logger.warning(f"Invalid input shape after preprocessing: {X.shape}")
            # Create a dummy input with zeros of the expected shape
            expected_shape = (X.shape[0], self.config.get('sequence_length', 30), 
                             self.config.get('input_dim', 34))
            logger.info(f"Creating empty array with expected shape {expected_shape}")
            return np.zeros(expected_shape)
        
        # Attempt to reshape based on expected features_per_frame or keypoints/coords
        features_per_frame = self.config.get('features_per_frame', self.input_shape[2] if len(self.input_shape) == 3 else None) # Target features
        num_keypoints_model = self.config.get('num_keypoints', 13) # Expected keypoints
        num_coords_model = self.config.get('num_coords', 2) # Expected coords

        batch_size, seq_len, features_input = X.shape # Get shape from actual input

        if features_input == features_per_frame:
            # Shape is already correct (e.g., (batch, seq, 34) or (batch, seq, 26))
            logger.debug(f"Input shape {X.shape} matches expected features_per_frame {features_per_frame}. No reshape needed.")
            X_processed = X
        elif features_input == num_keypoints_model * num_coords_model:
            # Shape is (batch, seq, 26) - needs reshape to 4D if model expects it
            if expected_input_shape and len(expected_input_shape) == 3:
                logger.info(f"Input shape {X.shape} matches expected keypoints*coords. Reshaping to {expected_input_shape}.")
                X_processed = X.reshape(batch_size, seq_len, num_keypoints_model, num_coords_model)
            else:
                logger.debug(f"Input shape {X.shape} matches expected keypoints*coords. Model expects 3D. No reshape needed.")
                X_processed = X
        elif features_input == self.config.get('num_keypoints', 17) * self.config.get('num_coords', 2): # Input has 17 keypoints, model expects 13?
             logger.warning(f"Input features ({features_input}) do not directly reshape to model's keypoints*coords ({num_keypoints_model}*{num_coords_model}). Attempting to select relevant keypoints.")
             # Assume the input X has 17 keypoints and we need to select the 13 the model expects
             keypoint_indices = self.config.get('keypoint_indices')
             logger.info(f"Attempting keypoint selection. Found keypoint_indices in config: {keypoint_indices}")
 
             if keypoint_indices and len(keypoint_indices) > 0:
                 # --- Perform Keypoint Selection and Padding --- 
                 logger.info(f"Selecting {len(keypoint_indices)} keypoints based on indices: {keypoint_indices}")
                 input_kpts = self.config.get('num_keypoints', 17) # Get the original number of keypoints (e.g., 17)
                 input_coords = self.config.get('num_coords', 2)

                 # Reshape input to (batch, seq, original_kpts, original_coords) temporarily
                 # Ensure the original number of keypoints/coords allows this reshape
                 if X.size != batch_size * seq_len * input_kpts * input_coords:
                      logger.error(f"Input data size {X.size} does not match expected size based on config {batch_size}*{seq_len}*{input_kpts}*{input_coords}. Cannot reshape for keypoint selection.")
                      # Fallback if reshape is impossible
                      target_shape_fallback = (batch_size, seq_len, num_keypoints_model, num_coords_model) if expected_input_shape[-1] == 4 else (batch_size, seq_len, features_per_frame)
                      X_processed = np.zeros(target_shape_fallback, dtype=X.dtype)
                 else: 
                     X_temp_4d = X.reshape(batch_size, seq_len, input_kpts, input_coords)
                     # Select the required keypoints using the indices
                     X_selected_kpts = X_temp_4d[:, :, keypoint_indices, :]
                     logger.debug(f"Selected keypoints shape: {X_selected_kpts.shape}")

                     # Now, check if padding is needed to match the model's expected keypoint count
                     target_kpts = num_keypoints_model # Should be 13 from config/default
                     target_coords = num_coords_model  # Should be 2
                     target_shape = (batch_size, seq_len, target_kpts, target_coords)

                     if X_selected_kpts.shape[2] == target_kpts:
                         logger.debug(f"Selected keypoints ({X_selected_kpts.shape[2]}) match target ({target_kpts}). No padding needed.")
                         X_processed_4d = X_selected_kpts
                     elif X_selected_kpts.shape[2] < target_kpts:
                         logger.warning(f"Padding selected {X_selected_kpts.shape[2]} keypoints to match model's expected {target_kpts} keypoints.")
                         X_processed_4d = np.zeros(target_shape, dtype=X_selected_kpts.dtype)
                         X_processed_4d[:, :, :X_selected_kpts.shape[2], :] = X_selected_kpts
                     else: # X_selected_kpts.shape[2] > target_kpts - Should not happen with current indices, but handle defensively
                         logger.warning(f"Selected keypoints ({X_selected_kpts.shape[2]}) exceed target ({target_kpts}). Truncating.")
                         X_processed_4d = X_selected_kpts[:, :, :target_kpts, :]

                     # Ensure final shape matches the expected dimensionality (3D or 4D)
                     expected_dim = 3 if expected_input_shape[-1] == 3 else 4
                     if expected_dim == 3 and len(X_processed_4d.shape) == 4:
                         X_processed = X_processed_4d.reshape(batch_size, seq_len, -1)
                         logger.info(f"Selected/padded/truncated keypoints and reshaped to 3D: {X_processed.shape}")
                     elif expected_dim == 4 and len(X_processed_4d.shape) == 4:
                         X_processed = X_processed_4d
                         logger.info(f"Selected/padded/truncated keypoints, kept 4D shape: {X_processed.shape}")
                     # Handle case where expected is 3D but result is already 3D (unlikely here but for completeness)
                     elif expected_dim == 3 and len(X_processed_4d.shape) == 3:
                          X_processed = X_processed_4d 
                          logger.info(f"Selected/padded/truncated keypoints, kept 3D shape: {X_processed.shape}")
                     else:
                         logger.warning(f"Unexpected dimension mismatch after processing. Processed shape: {X_processed_4d.shape}, expected model dim: {expected_dim}. Falling back.")
                         target_shape_fallback = (batch_size, seq_len, num_keypoints_model, num_coords_model) if expected_dim == 4 else (batch_size, seq_len, features_per_frame)
                         X_processed = np.zeros(target_shape_fallback, dtype=X.dtype)
             else:
                 logger.error(f"Cannot select keypoints: 'keypoint_indices' missing, empty, or length mismatch (expected {num_keypoints_model}, got {len(keypoint_indices) if keypoint_indices else 'None'}). Falling back to zeros.")
                 target_shape = (batch_size, seq_len, num_keypoints_model, num_coords_model) if expected_input_shape[-1] == 4 else (batch_size, seq_len, features_per_frame)
                 X_processed = np.zeros(target_shape, dtype=X.dtype)
        else:
             logger.warning(f"Unhandled input feature size {features_input}. Expected {features_per_frame} or {num_keypoints_model * num_coords_model}. Falling back to zeros.")
             target_shape = (batch_size, seq_len, num_keypoints_model, num_coords_model) if expected_input_shape[-1] == 4 else (batch_size, seq_len, features_per_frame)
             X_processed = np.zeros(target_shape, dtype=X.dtype)

        # Final shape check
        final_shape = list(X_processed.shape)
        target_shape_list = list(self.input_shape)
        
        if final_shape != target_shape_list:
            logger.warning(f"Final shape {final_shape} does not match expected shape {target_shape_list}")
        
        self.input_shape = X_processed.shape[1:]
        logger.info(f"Using input with shape {X_processed.shape}")
        return X_processed
    
    def train(self, X_train, y_train, X_val=None, y_val=None, model_path=None):
        """
        Train the RNN model on the given data.
        
        Args:
            X_train (np.ndarray): Training data with shape (n_samples, sequence_length, features_per_frame)
                or (n_samples, sequence_length, n_keypoints, coords)
            y_train (np.ndarray): Training labels with shape (n_samples,)
            X_val (np.ndarray, optional): Validation data
            y_val (np.ndarray, optional): Validation labels
            model_path (str, optional): Path to save the best model

        Returns:
            History object from model training
        """
        # Force TensorFlow to use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Preprocess input data
        X_train_processed = self.preprocess_input(X_train)
        X_val_processed = self.preprocess_input(X_val) if X_val is not None else None
        
        # Apply data augmentation if enabled
        if self.config['augment_data']:
            logger.info("Applying data augmentation")
            X_train_processed, y_train = self._augment_data(X_train_processed, y_train)
            logger.info(f"After augmentation, training data shape: {X_train_processed.shape}")
        
        # Build model if it doesn't exist
        if self.model is None:
            logger.info(f"Building model with input shape {self.input_shape}")
            unique_classes = np.unique(y_train)
            logger.info(f"Unique classes in training data: {unique_classes}")
            self.build_model(self.input_shape, len(unique_classes))
        
        callbacks = []
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['patience'],
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        callbacks.append(early_stopping)
        
        validation_data = None
        if X_val_processed is not None and y_val is not None:
            validation_data = (X_val_processed, y_val)
        
        logger.info("Training model on %d samples", len(X_train_processed))
        
        # Calculate class weights if enabled
        use_class_weights = self.config['use_class_weights']
        if use_class_weights:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            logger.info(f"Using class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
        
        # Run training with CPU
        with tf.device('/cpu:0'):
            history = self.model.fit(
                X_train_processed, y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
            )
        
        return history
