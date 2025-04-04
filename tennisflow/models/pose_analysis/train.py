"""
Training script for TennisFlow pose analysis model.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import yaml
import time
import logging

# Configure logging
logger = logging.getLogger("tennisflow.models.pose_analysis")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import POSE_CONFIG, POSE_MODEL_DIR, TENSORBOARD_DIR
from utils import set_seed, EarlyStopping, plot_learning_curves
from pose_analysis.datasets import prepare_pose_datasets, custom_collate_fn
from pose_analysis.model import create_tennis_pose_model, export_model

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    logger.warning("TensorBoard not available, install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (Train)")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get the inputs and labels
        inputs = batch['image'].to(device)
        labels = batch['category_id'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average validation loss
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get the inputs and labels
            inputs = batch['image'].to(device)
            labels = batch['category_id'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def get_class_accuracy(model, dataloader, device):
    """
    Get accuracy for each class.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device to evaluate on
    
    Returns:
        Dictionary of class accuracies
    """
    model.eval()
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Class Evaluation"):
            inputs = batch['image'].to(device)
            labels = batch['category_id'].to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                correct = (predicted[i] == label)
                
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                
                class_correct[label] += correct.item()
                class_total[label] += 1
    
    # Calculate accuracies
    class_accuracy = {}
    for class_id in class_total:
        class_accuracy[class_id] = 100 * class_correct[class_id] / class_total[class_id]
    
    return class_accuracy

def train(train_loader=None, val_loader=None, test_loader=None, config=None):
    """
    Train the model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Training configuration
    """
    if config is None:
        config = POSE_CONFIG
        
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create the save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.get('output_dir', POSE_MODEL_DIR)) / f"pose_model_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize TensorBoard
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / f"pose_train_{timestamp}"))
    else:
        writer = None
    
    # Prepare datasets and loaders if not provided
    if train_loader is None or val_loader is None or test_loader is None:
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, test_dataset = prepare_pose_datasets(
            data_dir=config.get('data_dir'),
            classes=config.get('classes'),
            img_size=config.get('img_size', 224),
            augmentation_level=config.get('augmentation_level', 'medium')
        )
        
        if not train_dataset or len(train_dataset) == 0:
            logger.error("No training data available.")
            return
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    
    logger.info(f"Training with {len(train_loader.dataset)} samples, validating with {len(val_loader.dataset)} samples")
    
    # Create model
    model = create_tennis_pose_model(config, device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 0.0005)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 15),
        min_delta=0.001,
        mode='min'
    )
    
    # Training loop
    epochs = config.get('epochs', 100)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    logger.info(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        logger.info(f"Epoch {epoch}/{epochs}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            # Save the model
            model.save(save_dir / 'best_model.pth')
            logger.info(f"Saved best model (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        
        # Check early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds.")
    logger.info(f"Best validation loss: {best_val_loss:.4f}, accuracy: {best_val_acc:.2f}%")
    
    # Load the best model
    try:
        model.load(save_dir / 'best_model.pth', device=device)
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
        logger.info("Using the last trained model")
    
    # Final evaluation on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Get class-wise accuracy
    class_accuracy = get_class_accuracy(model, test_loader, device)
    for class_id, accuracy in class_accuracy.items():
        logger.info(f"Class {class_id}: {accuracy:.2f}%")
        if writer:
            writer.add_scalar(f'TestAccuracy/class_{class_id}', accuracy, 0)
    
    # Plot learning curves
    plot_learning_curves(
        train_losses, val_losses, 'Loss',
        save_path=save_dir / 'loss_curve.png'
    )
    
    plot_learning_curves(
        train_accs, val_accs, 'Accuracy',
        save_path=save_dir / 'accuracy_curve.png'
    )
    
    # Export model for inference
    for export_format in config.get('export_formats', ['onnx']):
        export_path = save_dir / f"model.{export_format}"
        export_model(model, export_path, format=export_format)
    
    # Copy best model to the parent directory for easy access
    import shutil
    best_model_path = save_dir / 'best_model.pth'
    output_dir = Path(config.get('output_dir', POSE_MODEL_DIR))
    shutil.copy(best_model_path, output_dir / 'best_model.pth')
    logger.info(f"Copied best model to {output_dir / 'best_model.pth'}")
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    # Save results summary
    results = {
        'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'class_accuracy': {str(k): float(v) for k, v in class_accuracy.items()},
        'training_time': float(training_time),
        'epochs_trained': epoch,
    }
    
    with open(save_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Training results saved to {save_dir}")
    return results 