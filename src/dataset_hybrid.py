"""
Hybrid Dataset Pipeline for Two-Stage Classification
Stage 1: Binary (Cattle vs Buffalo)
Stage 2: Breed-specific classification
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

from src.dataset import BuffaloBreedDataset, get_transforms, analyze_dataset

logger = logging.getLogger(__name__)


class HybridDatasetManager:
    """Manages datasets for two-stage hybrid classification"""

    def __init__(self, config: Dict):
        self.config = config
        self.root_dir = Path(config['dataset']['root_dir'])

        # Detect dataset structure
        self.has_cattle = (self.root_dir / 'cattle').exists()
        self.has_buffalo = (self.root_dir / 'buffalo').exists()

        if not self.has_buffalo:
            # Legacy structure: root contains buffalo breeds directly
            self.buffalo_dir = self.root_dir
            self.cattle_dir = None
        else:
            # New structure: root/buffalo and root/cattle
            self.buffalo_dir = self.root_dir / 'buffalo'
            self.cattle_dir = self.root_dir / 'cattle' if self.has_cattle else None

        logger.info(f"Dataset structure detected:")
        logger.info(f"  Buffalo: {self.buffalo_dir}")
        logger.info(
            f"  Cattle: {self.cattle_dir if self.cattle_dir else 'Not found'}")

    def prepare_binary_dataset(self) -> Tuple:
        """Prepare binary classification dataset (cattle vs buffalo)"""
        if not self.cattle_dir:
            logger.warning(
                "No cattle directory found. Skipping binary classifier.")
            return None, None, None, None

        logger.info("\n" + "="*60)
        logger.info(
            "Preparing Binary Classification Dataset (Cattle vs Buffalo)")
        logger.info("="*60)

        # Collect all images with binary labels
        image_paths = []
        labels = []

        # Buffalo = 0
        buffalo_images = self._collect_images(self.buffalo_dir)
        image_paths.extend(buffalo_images)
        labels.extend([0] * len(buffalo_images))

        # Cattle = 1
        cattle_images = self._collect_images(self.cattle_dir)
        image_paths.extend(cattle_images)
        labels.extend([1] * len(cattle_images))

        logger.info(f"Binary dataset:")
        logger.info(f"  Buffalo: {len(buffalo_images)} images")
        logger.info(f"  Cattle: {len(cattle_images)} images")
        logger.info(f"  Total: {len(image_paths)} images")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels,
            test_size=0.2,
            random_state=self.config['dataset']['random_seed'],
            stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.config['dataset']['random_seed'],
            stratify=y_train
        )

        # Create datasets
        train_transform = get_transforms(self.config, 'train', enhanced=True)
        val_transform = get_transforms(self.config, 'val')

        train_dataset = BuffaloBreedDataset(
            X_train, y_train, train_transform, ['buffalo', 'cattle'])
        val_dataset = BuffaloBreedDataset(
            X_val, y_val, val_transform, ['buffalo', 'cattle'])
        test_dataset = BuffaloBreedDataset(
            X_test, y_test, val_transform, ['buffalo', 'cattle'])

        analysis = {
            'num_classes': 2,
            'class_names': ['buffalo', 'cattle'],
            'class_counts': {'buffalo': len(buffalo_images), 'cattle': len(cattle_images)},
            'total_images': len(image_paths)
        }

        return train_dataset, val_dataset, test_dataset, analysis

    def prepare_breed_dataset(self, animal_type: str) -> Tuple:
        """
        Prepare breed-specific classification dataset

        Args:
            animal_type: 'buffalo' or 'cattle'
        """
        logger.info("\n" + "="*60)
        logger.info(
            f"Preparing {animal_type.capitalize()} Breed Classification Dataset")
        logger.info("="*60)

        if animal_type == 'buffalo':
            breed_dir = self.buffalo_dir
        elif animal_type == 'cattle':
            if not self.cattle_dir:
                raise ValueError("Cattle directory not found")
            breed_dir = self.cattle_dir
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

        # Analyze dataset
        analysis = analyze_dataset(str(breed_dir))

        logger.info(f"Dataset Analysis:")
        logger.info(f"  Total Classes: {analysis['num_classes']}")
        logger.info(f"  Total Images: {analysis['total_images']}")
        logger.info(f"  Imbalance Ratio: {analysis['imbalance_ratio']:.2f}")

        if analysis['imbalanced_classes']:
            logger.warning(
                f"⚠ Classes with <50 images: {analysis['imbalanced_classes']}")

        # Collect all image paths and labels
        image_paths = []
        labels = []
        class_to_idx = {name: idx for idx,
                        name in enumerate(analysis['class_names'])}

        for class_name in analysis['class_names']:
            class_dir = breed_dir / class_name
            class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG'))

            for img_path in class_images:
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])

        # Split dataset with stratification
        train_ratio = self.config['dataset']['train_split']
        val_ratio = self.config['dataset']['val_split']
        test_ratio = self.config['dataset']['test_split']
        seed = self.config['dataset']['random_seed']

        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels
        )

        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=seed,
            stratify=y_temp
        )

        logger.info(f"Data Split:")
        logger.info(f"  Train: {len(X_train)} images")
        logger.info(f"  Val: {len(X_val)} images")
        logger.info(f"  Test: {len(X_test)} images")

        # Create datasets with enhanced augmentation
        train_transform = get_transforms(self.config, 'train', enhanced=True)
        val_transform = get_transforms(self.config, 'val')

        train_dataset = BuffaloBreedDataset(
            X_train, y_train, train_transform, analysis['class_names'])
        val_dataset = BuffaloBreedDataset(
            X_val, y_val, val_transform, analysis['class_names'])
        test_dataset = BuffaloBreedDataset(
            X_test, y_test, val_transform, analysis['class_names'])

        return train_dataset, val_dataset, test_dataset, analysis

    def _collect_images(self, directory: Path) -> List[str]:
        """Recursively collect all images from directory and subdirectories"""
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']:
            images.extend([str(p) for p in directory.rglob(ext)])
        return images

    def get_weighted_sampler(self, dataset, labels: List[int]):
        """Create weighted sampler for handling class imbalance"""
        class_counts = Counter(labels)
        class_weights = {cls: 1.0 / count for cls,
                         count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        logger.info("✓ Weighted sampler created for class imbalance")
        return sampler


def get_hybrid_dataloaders(config: Dict) -> Dict:
    """
    Get all dataloaders for hybrid classification system

    Returns:
        Dict containing:
        - binary: (train, val, test) loaders for cattle/buffalo classification
        - buffalo_breed: (train, val, test) loaders for buffalo breeds
        - cattle_breed: (train, val, test) loaders for cattle breeds (if available)
    """
    manager = HybridDatasetManager(config)
    result = {}

    # Binary classification loaders
    binary_data = manager.prepare_binary_dataset()
    if binary_data[0] is not None:
        train_ds, val_ds, test_ds, analysis = binary_data

        batch_size = config['training']['batch_size']
        result['binary'] = {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            'analysis': analysis
        }

    # Buffalo breed loaders
    train_ds, val_ds, test_ds, analysis = manager.prepare_breed_dataset(
        'buffalo')
    batch_size = config['training']['batch_size']
    result['buffalo_breed'] = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        'analysis': analysis
    }

    # Cattle breed loaders (if available)
    if manager.cattle_dir:
        try:
            train_ds, val_ds, test_ds, analysis = manager.prepare_breed_dataset(
                'cattle')
            result['cattle_breed'] = {
                'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'analysis': analysis
            }
        except Exception as e:
            logger.warning(f"Could not prepare cattle breed dataset: {e}")

    return result
