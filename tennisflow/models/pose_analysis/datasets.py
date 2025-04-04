"""
Dataset classes for pose analysis model training.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import logging

# Configure logging
logger = logging.getLogger("tennisflow.models.pose_analysis")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_json

class TennisPostureDataset(Dataset):
    """
    Dataset for tennis pose estimation with COCO format annotations.
    
    Args:
        image_dir: Directory containing images
        annotations_file: Path to COCO format annotations JSON file
        transform: Albumentations transforms to apply
        class_name: Filter by class name (optional)
        class_mapping: Mapping from category IDs to class indices (0-based)
    """
    def __init__(self, image_dir, annotations_file, transform=None, class_name=None, class_mapping=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load annotations
        self.annotations = load_json(annotations_file)
        
        # Create lookup dictionaries
        self.images = {img['id']: img for img in self.annotations['images']}
        
        # Filter annotations by category if specified
        self.categories = self.annotations.get('categories', [])
        self.category_map = {cat['id']: cat['name'] for cat in self.categories}
        
        # Create a mapping from category IDs to class indices (0-based)
        if class_mapping is None:
            # Create default mapping based on category order
            self.class_mapping = {cat['id']: i for i, cat in enumerate(self.categories)}
        else:
            self.class_mapping = class_mapping
            
        logger.info(f"Category to class mapping: {self.class_mapping}")
        
        self.annotations_list = self.annotations['annotations']
        
        if class_name:
            # Get category IDs matching the class name
            category_ids = [cat['id'] for cat in self.categories if cat['name'] == class_name]
            if not category_ids:
                logger.warning(f"Class name '{class_name}' not found in annotations.")
            else:
                self.annotations_list = [ann for ann in self.annotations_list 
                                      if ann['category_id'] in category_ids]
        
        logger.info(f"Loaded {len(self.annotations_list)} annotations from {annotations_file}")
    
    def __len__(self):
        return len(self.annotations_list)
    
    def __getitem__(self, idx):
        annotation = self.annotations_list[idx]
        image_id = annotation['image_id']
        image_info = self.images[image_id]
        image_path = self.image_dir / image_info['file_name']
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            # Return a placeholder image and data
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            keypoints = np.zeros((17, 3), dtype=np.float32)  # Default COCO keypoints
            bbox = np.array([0, 0, 224, 224], dtype=np.float32)
            category_id = 0
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get keypoints
            keypoints = annotation.get('keypoints', [])
            keypoints = np.array(keypoints).reshape(-1, 3)  # x, y, visibility
            
            # Get bounding box
            bbox = annotation.get('bbox', [0, 0, image.shape[1], image.shape[0]])
            if len(bbox) == 4:
                x, y, w, h = bbox
                bbox = [x, y, x + w, y + h]  # Convert to x1, y1, x2, y2 format
            
            # Get category
            category_id = annotation.get('category_id', 0)
        
        # Map to class index (0-based) using the class mapping
        class_idx = self.class_mapping.get(category_id, 0)
        
        # Prepare sample
        sample = {
            'image': image,
            'keypoints': keypoints,
            'bbox': np.array(bbox),
            'category_id': class_idx,  # Store as integer
            'image_id': image_id,
            'annotation_id': annotation['id'],
            'image_path': str(image_path)
        }
        
        # Apply transformations
        if self.transform:
            # Convert keypoints to list of (x,y) tuples for albumentations
            keypoints_list = [(kp[0], kp[1]) for kp in keypoints]
            
            transformed = self.transform(
                image=image,
                keypoints=keypoints_list,
                bbox=bbox
            )
            
            sample['image'] = transformed['image']
            
            # Update other transformed data if available
            if 'keypoints' in transformed:
                # Convert back to numpy array with visibility
                transformed_keypoints = np.zeros_like(keypoints)
                for i, (kp, orig_kp) in enumerate(zip(transformed['keypoints'], keypoints)):
                    transformed_keypoints[i] = [kp[0], kp[1], orig_kp[2]]
                sample['keypoints'] = transformed_keypoints
                
            if 'bbox' in transformed:
                sample['bbox'] = np.array(transformed['bbox'])
        
        # Convert category_id to Long tensor (this must be an integer type)
        sample['category_id'] = torch.tensor(sample['category_id'], dtype=torch.long)
        
        return sample

def create_tennis_pose_transforms(config, mode='train'):
    """
    Create transformations for tennis pose dataset.
    
    Args:
        config: Configuration dictionary
        mode: 'train', 'val', or 'test'
    
    Returns:
        Albumentations transforms
    """
    img_size = config['img_size']
    
    if mode == 'train':
        # Training transforms with augmentation
        level = config.get('augmentation_level', 'medium')
        
        if level == 'none':
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        elif level == 'light':
            return A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        elif level == 'medium':
            return A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        elif level == 'heavy':
            return A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(p=0.3),
                A.OneOf([
                    A.MotionBlur(p=0.5),
                    A.GaussianBlur(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.ISONoise(p=0.5),
                ], p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        else:
            logger.warning(f"Unknown augmentation level: {level}, using medium")
            return create_tennis_pose_transforms(config, mode='train')
    
    else:
        # Validation/Test transforms (no augmentation)
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def create_coco_format_dataset(images_dir, annotations_path, class_name=None, transform=None):
    """
    Create a dataset from COCO format annotations.
    
    Args:
        images_dir: Directory containing images
        annotations_path: Path to COCO format annotations JSON file
        class_name: Filter by class name (optional)
        transform: Albumentations transforms to apply
    
    Returns:
        TennisPostureDataset
    """
    return TennisPostureDataset(
        image_dir=images_dir,
        annotations_file=annotations_path,
        transform=transform,
        class_name=class_name
    )

def prepare_pose_datasets(data_dir, classes=None, img_size=224, augmentation_level='medium', train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Prepare train, validation, and test datasets for pose analysis.
    
    Args:
        data_dir: Directory containing class subdirectories with images and annotations
        classes: List of class names to include
        img_size: Image size for resizing
        augmentation_level: Level of data augmentation ('none', 'light', 'medium', 'heavy')
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Default classes if none provided
    if classes is None:
        classes = ['forehand', 'backhand', 'serve', 'ready_position']
    
    # Convert data_dir to Path
    data_dir = Path(data_dir)
    
    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Classes: {classes}")
    
    # Configuration for transforms
    config = {
        'img_size': img_size,
        'augmentation_level': augmentation_level
    }
    
    # Create a consistent class mapping for all datasets
    class_mapping = {i+1: i for i in range(len(classes))}
    logger.info(f"Class mapping: {class_mapping}")
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for i, class_name in enumerate(classes):
        logger.info(f"Preparing datasets for {class_name}")
        
        # Prepare transforms
        train_transform = create_tennis_pose_transforms(config, mode='train')
        val_transform = create_tennis_pose_transforms(config, mode='val')
        test_transform = create_tennis_pose_transforms(config, mode='test')
        
        # Get class directory and annotation file
        class_dir = data_dir / class_name
        annotation_file = data_dir / f"{class_name}.json"
        
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
            
        if not annotation_file.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            continue
        
        # Create datasets
        dataset = TennisPostureDataset(
            image_dir=class_dir,
            annotations_file=annotation_file,
            transform=None,  # No transform yet
            class_name=class_name,
            class_mapping={cat_id: i for cat_id in range(1, 10)}  # Map all category IDs to this class index
        )
        
        # Split dataset
        total_size = len(dataset)
        test_size = int(total_size * test_split)
        val_size = int(total_size * val_split)
        train_size = total_size - test_size - val_size
        
        if total_size == 0:
            logger.warning(f"No samples for class {class_name}")
            continue
            
        logger.info(f"Class {class_name}: {total_size} total samples")
        
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))
        
        # Apply transforms
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        train_dataset.dataset.transform = train_transform
        
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        val_dataset.dataset.transform = val_transform
        
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        test_dataset.dataset.transform = test_transform
        
        # Append to lists
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
        
        logger.info(f"Created datasets for {class_name}: {train_size} train, {val_size} val, {test_size} test")
    
    # Combine datasets
    if not train_datasets:
        logger.error("No datasets created")
        return None, None, None
        
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
    logger.info(f"Final datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_dataset, val_dataset, test_dataset

# Custom collate function to handle different data types
def custom_collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch)
    elif isinstance(elem, (int, float)):
        return torch.tensor(batch)
    elif isinstance(elem, dict):
        return {key: custom_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, (list, tuple)):
        return elem_type([custom_collate_fn(samples) for samples in zip(*batch)])
    else:
        return batch 