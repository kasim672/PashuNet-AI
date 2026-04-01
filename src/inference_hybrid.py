"""
Hybrid Two-Stage Inference System
Stage 1: Animal Type Classification (Cattle vs Buffalo)
Stage 2: Breed Classification
Features: Multi-image prediction, Decision support, Domain intelligence
"""

from src.inference import BREED_FEATURES, DecisionSupportEngine
import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
from collections import Counter
import json
import os

logger = logging.getLogger(__name__)

# Import domain intelligence from legacy inference


class HybridBreedPredictor:
    """
    Production-ready Hybrid Two-Stage Inference System
    Stage 1: Determines animal type (cattle vs buffalo)
    Stage 2: Predicts specific breed based on animal type
    """

    def __init__(self, model_dir: str, config: Dict, device: torch.device):
        """
        Initialize hybrid predictor

        Args:
            model_dir: Directory containing trained models and metadata
            config: Configuration dictionary
            device: torch device
        """
        self.device = device
        self.config = config
        self.model_dir = Path(model_dir)
        self.decision_engine = DecisionSupportEngine()

        # Load metadata
        metadata_path = self.model_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load models
        from src.model_hybrid import BinaryAnimalClassifier, BreedClassifier

        # Binary classifier
        self.binary_classifier = BinaryAnimalClassifier(config)
        binary_path = self.model_dir / 'binary_classifier.pth'
        if binary_path.exists():
            self.binary_classifier.load_state_dict(
                torch.load(binary_path, map_location=device))
            self.binary_classifier.to(device)
            self.binary_classifier.eval()
            self.has_binary = True
            logger.info("✓ Binary classifier loaded")
        else:
            self.has_binary = False
            logger.warning(
                "⚠ Binary classifier not found - assuming buffalo only")

        # Buffalo breed classifier
        buffalo_classes = self.metadata['num_buffalo_breeds']
        self.buffalo_classifier = BreedClassifier(
            buffalo_classes, config, 'buffalo')
        buffalo_path = self.model_dir / 'buffalo_classifier.pth'
        self.buffalo_classifier.load_state_dict(
            torch.load(buffalo_path, map_location=device))
        self.buffalo_classifier.to(device)
        self.buffalo_classifier.eval()
        self.buffalo_breeds = self.metadata['buffalo_breeds']
        logger.info(
            f"✓ Buffalo breed classifier loaded ({buffalo_classes} breeds)")

        # Cattle breed classifier (optional)
        self.cattle_classifier = None
        self.cattle_breeds = []
        if 'num_cattle_breeds' in self.metadata:
            cattle_classes = self.metadata['num_cattle_breeds']
            self.cattle_classifier = BreedClassifier(
                cattle_classes, config, 'cattle')
            cattle_path = self.model_dir / 'cattle_classifier.pth'
            if cattle_path.exists():
                self.cattle_classifier.load_state_dict(
                    torch.load(cattle_path, map_location=device))
                self.cattle_classifier.to(device)
                self.cattle_classifier.eval()
                self.cattle_breeds = self.metadata['cattle_breeds']
                logger.info(
                    f"✓ Cattle breed classifier loaded ({cattle_classes} breeds)")

        # Setup transforms
        img_size = config['image']['input_size']
        mean = config['image']['mean']
        std = config['image']['std']

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        logger.info(f"✓ Hybrid predictor initialized")
        logger.info(f"  Device: {device}")
        logger.info(
            f"  Binary classifier: {'Yes' if self.has_binary else 'No'}")
        logger.info(f"  Buffalo breeds: {len(self.buffalo_breeds)}")
        logger.info(f"  Cattle breeds: {len(self.cattle_breeds)}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0)

        return image_tensor

    def predict_animal_type(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Stage 1: Predict animal type (cattle vs buffalo)

        Returns:
            (animal_type, confidence)
        """
        if not self.has_binary:
            return 'buffalo', 1.0

        with torch.no_grad():
            outputs = self.binary_classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            animal_idx = torch.argmax(probabilities).item()
            confidence = probabilities[animal_idx].item()

        animal_type = 'buffalo' if animal_idx == 0 else 'cattle'
        return animal_type, confidence

    def predict_breed(self, image_tensor: torch.Tensor, animal_type: str, top_k: int = 3) -> List[Dict]:
        """
        Stage 2: Predict breed based on animal type

        Returns:
            List of predictions with breed name, confidence, and rank
        """
        if animal_type == 'buffalo':
            classifier = self.buffalo_classifier
            breed_names = self.buffalo_breeds
        elif animal_type == 'cattle':
            if self.cattle_classifier is None:
                raise ValueError("Cattle classifier not available")
            classifier = self.cattle_classifier
            breed_names = self.cattle_breeds
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

        with torch.no_grad():
            outputs = classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        top_probs, top_indices = torch.topk(
            probabilities, min(top_k, len(breed_names)))

        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            breed_name = breed_names[idx.item()]
            predictions.append({
                'rank': i + 1,
                'breed': breed_name,
                'confidence': prob.item(),
                'confidence_percent': f"{prob.item() * 100:.2f}%",
                'animal_type': animal_type
            })

        return predictions

    def predict_single(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Two-stage prediction for single image

        Returns:
            Complete prediction with animal type and breed
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Stage 1: Animal type
        animal_type, animal_confidence = self.predict_animal_type(image_tensor)

        # Stage 2: Breed
        breed_predictions = self.predict_breed(
            image_tensor, animal_type, top_k)

        return {
            'animal_type': animal_type,
            'animal_confidence': animal_confidence,
            'breed_predictions': breed_predictions
        }

    def predict_with_decision_support(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Single image prediction with full decision support

        Returns:
            Complete prediction with decision support and domain intelligence
        """
        result = self.predict_single(image_path, top_k)
        predictions = result['breed_predictions']

        # Decision support
        decision_info = self.decision_engine.make_decision(
            predictions[0]['confidence'],
            predictions,
            image_count=1
        )

        # Domain intelligence
        reasoning = self.decision_engine.get_breed_reasoning(
            predictions[0]['breed'])

        # Breed information
        breed_info = BREED_FEATURES.get(predictions[0]['breed'], {})

        return {
            'top_predictions': predictions,
            'final_prediction': predictions[0]['breed'],
            'confidence': predictions[0]['confidence'],
            'confidence_percent': predictions[0]['confidence_percent'],
            'animal_type': result['animal_type'],
            'animal_confidence': result['animal_confidence'],
            'decision': decision_info['decision'],
            'decision_message': decision_info['message'],
            'recommendation': decision_info['recommendation'],
            'confidence_level': decision_info['confidence_level'],
            'reasoning': reasoning,
            'breed_info': breed_info
        }

    def predict_multi(self, image_paths: List[str], top_k: int = 3, aggregation: str = 'average') -> Dict:
        """
        Predict breed from multiple images with aggregation

        Args:
            image_paths: List of image paths
            top_k: Number of top predictions to return
            aggregation: 'average' or 'voting'

        Returns:
            Aggregated prediction with decision support
        """
        if not image_paths:
            raise ValueError("No images provided")

        logger.info(
            f"Processing {len(image_paths)} images with {aggregation} aggregation...")

        # Collect predictions from all images
        all_animal_types = []
        all_probabilities = []
        all_breed_predictions = []

        for img_path in image_paths:
            try:
                image_tensor = self.preprocess_image(img_path)
                image_tensor = image_tensor.to(self.device)

                # Stage 1: Animal type
                animal_type, _ = self.predict_animal_type(image_tensor)
                all_animal_types.append(animal_type)

                # Stage 2: Breed probabilities
                if animal_type == 'buffalo':
                    classifier = self.buffalo_classifier
                    breed_names = self.buffalo_breeds
                elif animal_type == 'cattle' and self.cattle_classifier:
                    classifier = self.cattle_classifier
                    breed_names = self.cattle_breeds
                else:
                    continue

                with torch.no_grad():
                    outputs = classifier(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                all_probabilities.append(probabilities.cpu().numpy())

                # Get top prediction for voting
                top_idx = torch.argmax(probabilities).item()
                all_breed_predictions.append(breed_names[top_idx])

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        if not all_probabilities:
            raise ValueError("Failed to process any images")

        # Determine dominant animal type
        animal_type_counts = Counter(all_animal_types)
        dominant_animal_type = animal_type_counts.most_common(1)[0][0]

        # Get breed names for dominant type
        if dominant_animal_type == 'buffalo':
            breed_names = self.buffalo_breeds
        else:
            breed_names = self.cattle_breeds

        # Aggregate predictions
        if aggregation == 'average':
            avg_probs = np.mean(all_probabilities, axis=0)
            top_indices = np.argsort(avg_probs)[::-1][:top_k]

            aggregated_predictions = []
            for i, idx in enumerate(top_indices):
                breed_name = breed_names[idx]
                aggregated_predictions.append({
                    'rank': i + 1,
                    'breed': breed_name,
                    'confidence': float(avg_probs[idx]),
                    'confidence_percent': f"{avg_probs[idx] * 100:.2f}%",
                    'animal_type': dominant_animal_type
                })

        elif aggregation == 'voting':
            vote_counts = Counter(all_breed_predictions)
            total_votes = len(all_breed_predictions)

            aggregated_predictions = []
            for i, (breed, count) in enumerate(vote_counts.most_common(top_k)):
                confidence = count / total_votes
                aggregated_predictions.append({
                    'rank': i + 1,
                    'breed': breed,
                    'confidence': confidence,
                    'confidence_percent': f"{confidence * 100:.2f}%",
                    'votes': count,
                    'total_votes': total_votes,
                    'animal_type': dominant_animal_type
                })
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Get final prediction
        final_prediction = aggregated_predictions[0]

        # Decision support
        decision_info = self.decision_engine.make_decision(
            final_prediction['confidence'],
            aggregated_predictions,
            len(image_paths)
        )

        # Domain intelligence
        reasoning = self.decision_engine.get_breed_reasoning(
            final_prediction['breed'])

        # Breed information
        breed_info = BREED_FEATURES.get(final_prediction['breed'], {})

        return {
            'top_predictions': aggregated_predictions,
            'final_prediction': final_prediction['breed'],
            'confidence': final_prediction['confidence'],
            'confidence_percent': final_prediction['confidence_percent'],
            'animal_type': dominant_animal_type,
            'decision': decision_info['decision'],
            'decision_message': decision_info['message'],
            'recommendation': decision_info['recommendation'],
            'confidence_level': decision_info['confidence_level'],
            'reasoning': reasoning,
            'breed_info': breed_info,
            'aggregation_method': aggregation,
            'images_processed': len(image_paths),
            'images_successful': len(all_probabilities)
        }

    def predict_batch(self, image_paths: List[str], top_k: int = 3) -> List[Dict]:
        """Process multiple images independently (not aggregated)"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_with_decision_support(image_path, top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({'error': str(e), 'image_path': image_path})
        return results

    @property
    def class_names(self):
        """Get all breed names (for compatibility)"""
        return self.buffalo_breeds + self.cattle_breeds
