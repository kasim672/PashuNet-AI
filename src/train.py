"""
Training script for Buffalo Breed Recognition
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


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['training'].get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )

        optimizer.step()

        if config['training']['scheduler'] == 'cosine':
            scheduler.step()

        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
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

    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return epoch_loss, epoch_acc, precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, config, save_path='models/best_model.pth'):
    """Complete training loop"""

    num_epochs = config['training']['num_epochs']
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience']
    )

    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config
        )

        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler (for non-cosine schedulers)
        if config['training']['scheduler'] == 'plateau':
            scheduler.step(val_loss)
        elif config['training']['scheduler'] == 'step':
            scheduler.step()

        # Log metrics
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(
            f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            logger.info(f"✓ Best model saved! Val Acc: {val_acc:.4f}")

        # Fine-tuning: Unfreeze backbone
        if config['fine_tuning']['enabled'] and epoch == config['fine_tuning']['start_epoch']:
            logger.info("\n" + "=" * 60)
            logger.info("Starting Fine-Tuning Phase")
            logger.info("=" * 60)
            model.unfreeze_backbone(config['fine_tuning']['unfreeze_layers'])

            # Update optimizer with lower learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['fine_tuning']['learning_rate']
            logger.info(
                f"✓ Learning rate updated to {config['fine_tuning']['learning_rate']}")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    logger.info("\n" + "=" * 60)
    logger.info(f"Training Complete! Best Val Acc: {best_val_acc:.4f}")
    logger.info("=" * 60)

    return history, best_val_acc
