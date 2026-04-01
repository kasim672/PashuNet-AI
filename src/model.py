"""
⚠️ DEPRECATED - Use src/model_hybrid.py instead

Model architecture for Buffalo Breed Recognition (Legacy Single-Stage)
This file is maintained for backward compatibility only.

For new projects, use:
- src/model_hybrid.py for two-stage hybrid classification (cattle + buffalo)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class BuffaloBreedClassifier(nn.Module):
    """Buffalo Breed Classification Model with Transfer Learning"""

    def __init__(self, num_classes: int, config: Dict):
        super(BuffaloBreedClassifier, self).__init__()

        self.architecture = config['model']['architecture']
        self.num_classes = num_classes
        self.dropout = config['model']['dropout']

        # Load pretrained backbone
        if self.architecture == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(
                pretrained=config['model']['pretrained'])
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif self.architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                pretrained=config['model']['pretrained'])
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif self.architecture == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(
                pretrained=config['model']['pretrained'])
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.dropout / 2),
            nn.Linear(256, num_classes)
        )

        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            self.freeze_backbone()

        logger.info(f"✓ Model created: {self.architecture}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Dropout: {self.dropout}")

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("✓ Backbone frozen")

    def unfreeze_backbone(self, num_layers: int = None):
        """Unfreeze backbone parameters"""
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("✓ Backbone unfrozen (all layers)")
        else:
            # Unfreeze last N layers
            params = list(self.backbone.parameters())
            for param in params[-num_layers:]:
                param.requires_grad = True
            logger.info(f"✓ Backbone unfrozen (last {num_layers} layers)")


def create_model(num_classes: int, config: Dict, device: torch.device):
    """Create and initialize model"""
    model = BuffaloBreedClassifier(num_classes, config)
    model = model.to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logger.info(f"Model Parameters:")
    logger.info(f"  Trainable: {trainable:,}")
    logger.info(f"  Total: {total:,}")
    logger.info(f"  Frozen: {total - trainable:,}")

    return model


def get_optimizer(model, config: Dict):
    """Get optimizer based on configuration"""
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer_name = config['training']['optimizer'].lower()

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    logger.info(f"✓ Optimizer: {optimizer_name.upper()}, LR: {lr}")
    return optimizer


def get_scheduler(optimizer, config: Dict, steps_per_epoch: int):
    """Get learning rate scheduler"""
    scheduler_name = config['training']['scheduler'].lower()
    num_epochs = config['training']['num_epochs']

    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * steps_per_epoch
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logger.info(f"✓ Scheduler: {scheduler_name.upper()}")
    return scheduler


def get_loss_function(config: Dict, class_weights=None):
    """Get loss function with optional class weighting"""
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    if config['training']['use_weighted_loss'] and class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        logger.info("✓ Loss: Weighted CrossEntropyLoss with label smoothing")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info("✓ Loss: CrossEntropyLoss with label smoothing")

    return criterion
