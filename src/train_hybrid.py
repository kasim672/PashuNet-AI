"""
Training script for Hybrid Two-Stage Classification System
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch_single_model(model, dataloader, criterion, optimizer, scheduler, device, config, model_name="Model"):
    """Train single model for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Training {model_name}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        if config['training'].get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['training']['gradient_clip'])

        optimizer.step()

        if config['training']['scheduler'] == 'cosine':
            scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate_single_model(model, dataloader, criterion, device, model_name="Model"):
    """Validate single model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validating {model_name}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return epoch_loss, epoch_acc, precision, recall, f1


def train_binary_classifier(binary_model, train_loader, val_loader, config, device, save_path):
    """Train binary classifier (cattle vs buffalo)"""

    logger.info("\n" + "="*60)
    logger.info("STAGE 1: Training Binary Classifier (Cattle vs Buffalo)")
    logger.info("="*60)

    optimizer = torch.optim.Adam(
        binary_model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'] * len(train_loader)
    )

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'])

    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        logger.info("-" * 60)

        train_loss, train_acc = train_epoch_single_model(
            binary_model, train_loader, criterion, optimizer, scheduler, device, config, "Binary"
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_single_model(
            binary_model, val_loader, criterion, device, "Binary"
        )

        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(
            f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': binary_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            logger.info(f"✓ Best binary model saved! Val Acc: {val_acc:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(
        f"\nBinary Classifier Training Complete! Best Val Acc: {best_val_acc:.4f}")
    return history, best_val_acc


def train_breed_classifier(breed_model, train_loader, val_loader, config, device, save_path, animal_type):
    """Train breed-specific classifier"""

    logger.info("\n" + "="*60)
    logger.info(
        f"STAGE 2: Training {animal_type.capitalize()} Breed Classifier")
    logger.info("="*60)

    optimizer = torch.optim.Adam(
        breed_model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'] * len(train_loader)
    )

    # Calculate class weights if needed
    class_weights = None
    if config['training']['use_weighted_loss']:
        # Compute class weights from dataloader
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())

        from collections import Counter
        class_counts = Counter(all_labels)
        num_classes = len(class_counts)
        class_weights = torch.FloatTensor(
            [1.0 / class_counts[i] for i in range(num_classes)]).to(device)
        class_weights = class_weights / class_weights.sum() * num_classes

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=config['training'].get('label_smoothing', 0.0))
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'])

    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        logger.info("-" * 60)

        train_loss, train_acc = train_epoch_single_model(
            breed_model, train_loader, criterion, optimizer, scheduler, device, config, f"{animal_type.capitalize()} Breed"
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_single_model(
            breed_model, val_loader, criterion, device, f"{animal_type.capitalize()} Breed"
        )

        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(
            f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': breed_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            logger.info(
                f"✓ Best {animal_type} breed model saved! Val Acc: {val_acc:.4f}")

        # Fine-tuning
        if config['fine_tuning']['enabled'] and epoch == config['fine_tuning']['start_epoch']:
            logger.info("\n" + "=" * 60)
            logger.info(
                f"Starting Fine-Tuning Phase for {animal_type.capitalize()}")
            logger.info("=" * 60)
            breed_model.unfreeze_backbone(
                config['fine_tuning']['unfreeze_layers'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['fine_tuning']['learning_rate']
            logger.info(
                f"✓ Learning rate updated to {config['fine_tuning']['learning_rate']}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(
        f"\n{animal_type.capitalize()} Breed Classifier Training Complete! Best Val Acc: {best_val_acc:.4f}")
    return history, best_val_acc


def train_hybrid_system(hybrid_system, dataloaders, config, save_dir='models/hybrid'):
    """
    Train complete hybrid system

    Args:
        hybrid_system: HybridClassificationSystem instance
        dataloaders: Dict with 'binary', 'buffalo_breed', 'cattle_breed' loaders
        config: Configuration dict
        save_dir: Directory to save models
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    device = hybrid_system.device
    results = {}

    # Stage 1: Train binary classifier
    if 'binary' in dataloaders:
        binary_save_path = os.path.join(save_dir, 'binary_classifier.pth')
        binary_history, binary_acc = train_binary_classifier(
            hybrid_system.binary_classifier,
            dataloaders['binary']['train'],
            dataloaders['binary']['val'],
            config,
            device,
            binary_save_path
        )
        results['binary'] = {
            'history': binary_history,
            'best_val_acc': binary_acc
        }

    # Stage 2a: Train buffalo breed classifier
    if 'buffalo_breed' in dataloaders:
        buffalo_save_path = os.path.join(save_dir, 'buffalo_classifier.pth')
        buffalo_history, buffalo_acc = train_breed_classifier(
            hybrid_system.buffalo_classifier,
            dataloaders['buffalo_breed']['train'],
            dataloaders['buffalo_breed']['val'],
            config,
            device,
            buffalo_save_path,
            'buffalo'
        )
        results['buffalo_breed'] = {
            'history': buffalo_history,
            'best_val_acc': buffalo_acc
        }

    # Stage 2b: Train cattle breed classifier (if available)
    if 'cattle_breed' in dataloaders and hybrid_system.cattle_classifier:
        cattle_save_path = os.path.join(save_dir, 'cattle_classifier.pth')
        cattle_history, cattle_acc = train_breed_classifier(
            hybrid_system.cattle_classifier,
            dataloaders['cattle_breed']['train'],
            dataloaders['cattle_breed']['val'],
            config,
            device,
            cattle_save_path,
            'cattle'
        )
        results['cattle_breed'] = {
            'history': cattle_history,
            'best_val_acc': cattle_acc
        }

    logger.info("\n" + "="*60)
    logger.info("HYBRID SYSTEM TRAINING COMPLETE")
    logger.info("="*60)
    for key, value in results.items():
        logger.info(f"{key}: Best Val Acc = {value['best_val_acc']:.4f}")
    logger.info("="*60 + "\n")

    return results
