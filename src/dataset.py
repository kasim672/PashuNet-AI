"""
⚠️ DEPRECATED - Use src/dataset_hybrid.py instead

Dataset handling and preprocessing for Buffalo Breed Recognition (Legacy Single-Stage)
This file is maintained for backward compatibility only.

For new projects, use:
- src/dataset_hybrid.py for two-stage hybrid classification (cattle + buffalo)

Enhanced with production-grade augmentation and class imbalance handling
"""

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class HybridAnimalDataset(Dataset):
    """
    Enhanced Dataset for Hybrid Cattle+Buffalo Classification
    Returns: (image, breed_label, animal_type_label)
    animal_type_label: 0=buffalo, 1=cattle
    """

    def __init__(self, image_paths: List[str], breed_labels: List[int],
                 animal_type_labels: List[int], transform=None,
                 class_names: List[str] = None, mode: str = 'hybrid'):
        self.image_paths = image_paths
        self.breed_labels = breed_labels
        self.animal_type_labels = animal_type_labels
        self.transform = transform
        self.class_names = class_names
        self.mode = mode  # 'hybrid' or 'single'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        breed_label = self.breed_labels[idx]
        animal_type_label = self.animal_type_labels[idx]

        # Read image with error handling
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.mode == 'hybrid':
            return image, breed_label, animal_type_label
        else:
            return image, breed_label


class BuffaloBreedDataset(Dataset):
    """
    Legacy Dataset for Buffalo-only Classification (Backward Compatibility)
    """

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform=None, class_names: List[str] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


def get_transforms(config: Dict, mode: str = 'train'):
    """
    Get image transformations with production-grade augmentation
    Enhanced for real-world field conditions
    """
    img_size = config['image']['input_size']
    mean = config['image']['mean']
    std = config['image']['std']

    if mode == 'train':
        aug_config = config['augmentation']['train']
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.Rotate(limit=aug_config['rotation_limit'], p=0.5),

            # Lighting variations (field conditions)
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness_contrast'],
                contrast_limit=aug_config['brightness_contrast'],
                p=0.5
            ),
            # Low light simulation
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomToneCurve(scale=0.1, p=0.3),

            # Motion and blur (camera shake, movement)
            A.MotionBlur(blur_limit=7, p=0.3),
            A.Blur(blur_limit=aug_config['blur_limit'], p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),

            # Noise (sensor noise, compression artifacts)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),

            # Weather and environmental conditions
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20,
                         drop_width=1, drop_color=(200, 200, 200), p=0.1),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2,
                           shadow_dimension=5, p=0.2),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0,
                             angle_upper=1, p=0.1),

            # Occlusion (partial visibility, obstacles)
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                            min_holes=1, min_height=8, min_width=8, p=0.25),

            # Color variations
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                          hue=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                 val_shift_limit=20, p=0.3),

            # Normalize and convert to tensor
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    return transform


def analyze_dataset(root_dir: str, mode: str = 'single') -> Dict:
    """
    Analyze dataset structure and class distribution

    Args:
        root_dir: Root directory path
        mode: 'single' (buffalo only) or 'hybrid' (cattle + buffalo)

    Returns:
        Dictionary with dataset statistics and class distribution
    """
    if mode == 'hybrid':
        # Analyze hybrid dataset (cattle + buffalo)
        analysis = {
            'mode': 'hybrid',
            'categories': {},
            'total_images': 0,
            'imbalanced_classes': [],
            'class_distribution': {}
        }

        for category in ['buffalo', 'cattle']:
            category_path = Path(root_dir) / category
            if not category_path.exists():
                logger.warning(
                    f"⚠ Category '{category}' not found at {category_path}")
                continue

            class_dirs = [d for d in category_path.iterdir() if d.is_dir()]
            category_analysis = {
                'num_classes': len(class_dirs),
                'class_names': [],
                'class_counts': {},
                'total_images': 0
            }

            for class_dir in sorted(class_dirs):
                class_name = class_dir.name
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                    list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + \
                    list(class_dir.glob('*.JPEG'))

                count = len(images)
                category_analysis['class_names'].append(class_name)
                category_analysis['class_counts'][class_name] = count
                category_analysis['total_images'] += count

                # Track imbalanced classes
                if count < 50:
                    analysis['imbalanced_classes'].append(
                        (f"{category}/{class_name}", count))

                # Global class distribution
                analysis['class_distribution'][f"{category}/{class_name}"] = count

            analysis['categories'][category] = category_analysis
            analysis['total_images'] += category_analysis['total_images']

        # Calculate overall imbalance ratio
        if analysis['class_distribution']:
            counts = list(analysis['class_distribution'].values())
            analysis['imbalance_ratio'] = max(counts) / max(min(counts), 1)
        else:
            analysis['imbalance_ratio'] = 1.0

        return analysis
    else:
        # Original single-category analysis (buffalo only)
        class_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]

        analysis = {
            'mode': 'single',
            'num_classes': len(class_dirs),
            'class_names': [],
            'class_counts': {},
            'total_images': 0,
            'imbalanced_classes': [],
            'class_distribution': {}
        }

        for class_dir in sorted(class_dirs):
            class_name = class_dir.name
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + \
                list(class_dir.glob('*.JPEG'))

            count = len(images)
            analysis['class_names'].append(class_name)
            analysis['class_counts'][class_name] = count
            analysis['total_images'] += count
            analysis['class_distribution'][class_name] = count

            if count < 50:
                analysis['imbalanced_classes'].append((class_name, count))

        # Calculate imbalance ratio (max/min class size)
        if analysis['class_counts']:
            counts = list(analysis['class_counts'].values())
            analysis['imbalance_ratio'] = max(counts) / max(min(counts), 1)
        else:
            analysis['imbalance_ratio'] = 1.0

        return analysis


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset
    Uses inverse frequency weighting
    """
    label_counts = Counter(labels)
    total_samples = len(labels)

    weights = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 1)  # Avoid division by zero
        weights[class_idx] = total_samples / (num_classes * count)

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return weights


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create weighted sampler for handling class imbalance
    """
    label_counts = Counter(labels)
    weights = [1.0 / label_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def prepare_data(config: Dict, mode: str = 'single') -> Tuple:
    """
    Prepare train, validation, and test datasets

    Args:
        config: Configuration dictionary
        mode: 'single' (buffalo only) or 'hybrid' (cattle + buffalo)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, analysis)
    """
    root_dir = config['dataset']['root_dir']

    # Analyze dataset
    logger.info(f"Analyzing dataset in {mode} mode...")
    analysis = analyze_dataset(root_dir, mode)

    if mode == 'hybrid':
        logger.info(f"Dataset Analysis (Hybrid Mode):")
        logger.info(f"  Total Images: {analysis['total_images']}")
        for category, cat_analysis in analysis['categories'].items():
            logger.info(f"  {category.upper()}:")
            logger.info(f"    Classes: {cat_analysis['num_classes']}")
            logger.info(f"    Images: {cat_analysis['total_images']}")
            logger.info(
                f"    Breeds: {', '.join(cat_analysis['class_names'][:5])}...")
    else:
        logger.info(f"Dataset Analysis (Single Mode):")
        logger.info(f"  Total Classes: {analysis['num_classes']}")
        logger.info(f"  Total Images: {analysis['total_images']}")
        logger.info(f"  Classes: {', '.join(analysis['class_names'])}")

    if analysis['imbalanced_classes']:
        logger.warning(
            f"⚠ Classes with <50 images: {len(analysis['imbalanced_classes'])}")
        for class_name, count in analysis['imbalanced_classes'][:5]:
            logger.warning(f"    {class_name}: {count} images")

    # Collect all image paths and labels
    image_paths = []
    breed_labels = []
    animal_type_labels = []

    if mode == 'hybrid':
        # Build global class mapping
        all_breed_names = []
        for category in ['buffalo', 'cattle']:
            if category in analysis['categories']:
                all_breed_names.extend(
                    analysis['categories'][category]['class_names'])

        breed_to_idx = {name: idx for idx, name in enumerate(all_breed_names)}
        animal_type_map = {'buffalo': 0, 'cattle': 1}

        for category in ['buffalo', 'cattle']:
            if category not in analysis['categories']:
                continue

            category_path = Path(root_dir) / category
            for class_name in analysis['categories'][category]['class_names']:
                class_dir = category_path / class_name
                class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                    list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + \
                    list(class_dir.glob('*.JPEG'))

                for img_path in class_images:
                    image_paths.append(str(img_path))
                    breed_labels.append(breed_to_idx[class_name])
                    animal_type_labels.append(animal_type_map[category])
    else:
        # Single mode (buffalo only)
        class_to_idx = {name: idx for idx,
                        name in enumerate(analysis['class_names'])}

        for class_name in analysis['class_names']:
            class_dir = Path(root_dir) / class_name
            class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + \
                list(class_dir.glob('*.JPEG'))

            for img_path in class_images:
                image_paths.append(str(img_path))
                breed_labels.append(class_to_idx[class_name])
                animal_type_labels.append(0)  # Default to buffalo

    # Split dataset with stratification
    train_ratio = config['dataset']['train_split']
    val_ratio = config['dataset']['val_split']
    test_ratio = config['dataset']['test_split']
    seed = config['dataset']['random_seed']

    # Use breed labels for stratification
    X_temp, X_test, y_breed_temp, y_breed_test, y_type_temp, y_type_test = train_test_split(
        image_paths, breed_labels, animal_type_labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=breed_labels
    )

    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_breed_train, y_breed_val, y_type_train, y_type_val = train_test_split(
        X_temp, y_breed_temp, y_type_temp,
        test_size=val_size,
        random_state=seed,
        stratify=y_breed_temp
    )

    logger.info(f"Data Split:")
    logger.info(f"  Train: {len(X_train)} images")
    logger.info(f"  Val: {len(X_val)} images")
    logger.info(f"  Test: {len(X_test)} images")

    # Compute class weights for imbalanced data
    if mode == 'hybrid':
        num_classes = len(breed_to_idx)
        class_names = all_breed_names
    else:
        num_classes = analysis['num_classes']
        class_names = analysis['class_names']

    class_weights = compute_class_weights(y_breed_train, num_classes)
    logger.info(f"✓ Computed class weights for {num_classes} classes")

    # Create datasets
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    test_transform = get_transforms(config, 'test')

    if mode == 'hybrid':
        train_dataset = HybridAnimalDataset(
            X_train, y_breed_train, y_type_train, train_transform, class_names, mode='hybrid')
        val_dataset = HybridAnimalDataset(
            X_val, y_breed_val, y_type_val, val_transform, class_names, mode='hybrid')
        test_dataset = HybridAnimalDataset(
            X_test, y_breed_test, y_type_test, test_transform, class_names, mode='hybrid')
    else:
        train_dataset = BuffaloBreedDataset(
            X_train, y_breed_train, train_transform, class_names)
        val_dataset = BuffaloBreedDataset(
            X_val, y_breed_val, val_transform, class_names)
        test_dataset = BuffaloBreedDataset(
            X_test, y_breed_test, test_transform, class_names)

    # Add class weights to analysis
    analysis['class_weights'] = class_weights
    analysis['num_classes'] = num_classes
    if mode == 'hybrid':
        analysis['all_breed_names'] = all_breed_names

    return train_dataset, val_dataset, test_dataset, analysis


def get_dataloaders(train_dataset, val_dataset, test_dataset, config: Dict,
                    use_weighted_sampler: bool = False):
    """
    Create DataLoaders for train, val, and test sets

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        config: Configuration dictionary
        use_weighted_sampler: Whether to use weighted sampling for class imbalance
    """
    batch_size = config['training']['batch_size']

    # Create weighted sampler if requested
    sampler = None
    shuffle = True
    if use_weighted_sampler and hasattr(train_dataset, 'breed_labels'):
        sampler = create_weighted_sampler(train_dataset.breed_labels)
        shuffle = False  # Sampler handles shuffling
        logger.info("✓ Using weighted sampler for class imbalance")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
