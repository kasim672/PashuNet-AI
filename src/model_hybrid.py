"""
Hybrid Model Architecture for Two-Stage Classification
Stage 1: Binary classifier (Cattle vs Buffalo)
Stage 2: Breed-specific classifiers
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class BinaryAnimalClassifier(nn.Module):
    """Binary classifier for Cattle vs Buffalo"""

    def __init__(self, config: Dict):
        super(BinaryAnimalClassifier, self).__init__()

        self.architecture = config['model']['architecture']
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
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.dropout / 2),
            nn.Linear(256, 2)  # 2 classes: buffalo, cattle
        )

        if config['model']['freeze_backbone']:
            self.freeze_backbone()

        logger.info(f"✓ Binary classifier created: {self.architecture}")

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("✓ Binary classifier backbone frozen")

    def unfreeze_backbone(self, num_layers: int = None):
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("✓ Binary classifier backbone unfrozen (all layers)")
        else:
            params = list(self.backbone.parameters())
            for param in params[-num_layers:]:
                param.requires_grad = True
            logger.info(
                f"✓ Binary classifier backbone unfrozen (last {num_layers} layers)")


class BreedClassifier(nn.Module):
    """Breed-specific classifier (for buffalo or cattle breeds)"""

    def __init__(self, num_classes: int, config: Dict, animal_type: str):
        super(BreedClassifier, self).__init__()

        self.architecture = config['model']['architecture']
        self.num_classes = num_classes
        self.dropout = config['model']['dropout']
        self.animal_type = animal_type

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
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Breed classification head
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

        if config['model']['freeze_backbone']:
            self.freeze_backbone()

        logger.info(
            f"✓ {animal_type.capitalize()} breed classifier created: {self.architecture}")
        logger.info(f"  Num classes: {num_classes}")

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info(
            f"✓ {self.animal_type.capitalize()} breed classifier backbone frozen")

    def unfreeze_backbone(self, num_layers: int = None):
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info(
                f"✓ {self.animal_type.capitalize()} breed classifier backbone unfrozen (all layers)")
        else:
            params = list(self.backbone.parameters())
            for param in params[-num_layers:]:
                param.requires_grad = True
            logger.info(
                f"✓ {self.animal_type.capitalize()} breed classifier backbone unfrozen (last {num_layers} layers)")


class HybridClassificationSystem:
    """Complete two-stage hybrid classification system"""

    def __init__(self, config: Dict, buffalo_classes: int, cattle_classes: int = None):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Stage 1: Binary classifier
        self.binary_classifier = BinaryAnimalClassifier(config).to(self.device)

        # Stage 2: Breed classifiers
        self.buffalo_classifier = BreedClassifier(
            buffalo_classes, config, 'buffalo').to(self.device)
        self.cattle_classifier = None
        if cattle_classes:
            self.cattle_classifier = BreedClassifier(
                cattle_classes, config, 'cattle').to(self.device)

        logger.info("\n" + "="*60)
        logger.info("Hybrid Classification System Initialized")
        logger.info("="*60)
        logger.info(f"Binary Classifier: 2 classes (buffalo, cattle)")
        logger.info(f"Buffalo Breed Classifier: {buffalo_classes} classes")
        if cattle_classes:
            logger.info(f"Cattle Breed Classifier: {cattle_classes} classes")
        logger.info("="*60 + "\n")

    def predict(self, image: torch.Tensor):
        """
        Two-stage prediction

        Args:
            image: Input image tensor (B, C, H, W)

        Returns:
            animal_type: 0 (buffalo) or 1 (cattle)
            breed_logits: Breed classification logits
            breed_probs: Breed probabilities
        """
        # Stage 1: Determine animal type
        with torch.no_grad():
            binary_logits = self.binary_classifier(image)
            binary_probs = torch.softmax(binary_logits, dim=1)
            animal_type = torch.argmax(binary_probs, dim=1)

        # Stage 2: Breed classification
        breed_logits_list = []
        breed_probs_list = []

        for i, atype in enumerate(animal_type):
            img = image[i:i+1]
            if atype == 0:  # Buffalo
                with torch.no_grad():
                    logits = self.buffalo_classifier(img)
                    probs = torch.softmax(logits, dim=1)
            else:  # Cattle
                if self.cattle_classifier:
                    with torch.no_grad():
                        logits = self.cattle_classifier(img)
                        probs = torch.softmax(logits, dim=1)
                else:
                    # No cattle classifier available
                    logits = torch.zeros(1, 1).to(self.device)
                    probs = torch.zeros(1, 1).to(self.device)

            breed_logits_list.append(logits)
            breed_probs_list.append(probs)

        return animal_type, breed_logits_list, breed_probs_list

    def save_models(self, save_dir: str):
        """Save all models"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.binary_classifier.state_dict(),
                   os.path.join(save_dir, 'binary_classifier.pth'))
        torch.save(self.buffalo_classifier.state_dict(),
                   os.path.join(save_dir, 'buffalo_classifier.pth'))
        if self.cattle_classifier:
            torch.save(self.cattle_classifier.state_dict(),
                       os.path.join(save_dir, 'cattle_classifier.pth'))

        logger.info(f"✓ Models saved to {save_dir}")

    def load_models(self, save_dir: str):
        """Load all models"""
        import os

        self.binary_classifier.load_state_dict(
            torch.load(os.path.join(save_dir, 'binary_classifier.pth')))
        self.buffalo_classifier.load_state_dict(
            torch.load(os.path.join(save_dir, 'buffalo_classifier.pth')))
        if self.cattle_classifier and os.path.exists(os.path.join(save_dir, 'cattle_classifier.pth')):
            self.cattle_classifier.load_state_dict(
                torch.load(os.path.join(save_dir, 'cattle_classifier.pth')))

        logger.info(f"✓ Models loaded from {save_dir}")


def create_hybrid_system(config: Dict, buffalo_classes: int, cattle_classes: int = None):
    """Create hybrid classification system"""
    system = HybridClassificationSystem(
        config, buffalo_classes, cattle_classes)

    # Count parameters
    binary_params = sum(
        p.numel() for p in system.binary_classifier.parameters() if p.requires_grad)
    buffalo_params = sum(
        p.numel() for p in system.buffalo_classifier.parameters() if p.requires_grad)

    logger.info(f"Model Parameters:")
    logger.info(f"  Binary Classifier: {binary_params:,} trainable")
    logger.info(f"  Buffalo Classifier: {buffalo_params:,} trainable")

    if system.cattle_classifier:
        cattle_params = sum(
            p.numel() for p in system.cattle_classifier.parameters() if p.requires_grad)
        logger.info(f"  Cattle Classifier: {cattle_params:,} trainable")

    return system
