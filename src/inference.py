"""
⚠️ PARTIALLY DEPRECATED - Core logic moved to src/inference_hybrid.py

Enhanced Inference System for Hybrid Cattle+Buffalo Breed Recognition
This file contains:
- BREED_FEATURES database (still used)
- DecisionSupportEngine (still used)
- Legacy predictor classes (deprecated - use src/inference_hybrid.py)

For new projects, use:
- src/inference_hybrid.py for two-stage hybrid predictions
- Import BREED_FEATURES and DecisionSupportEngine from this file
"""

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

logger = logging.getLogger(__name__)


# Domain Intelligence: Breed-specific features and characteristics
BREED_FEATURES = {
    # Buffalo Breeds
    "Murrah": {
        "description": "Most popular dairy buffalo breed in India",
        "key_features": ["Curved horns sweeping back", "Jet black coat", "Heavy body", "Large udder"],
        "origin": "Haryana, Punjab",
        "milk_yield": "High (15-20 liters/day)"
    },
    "Mehsana": {
        "description": "Dual-purpose breed from Gujarat",
        "key_features": ["Medium-sized curved horns", "Black coat with white markings", "Compact body"],
        "origin": "Gujarat",
        "milk_yield": "Medium (8-12 liters/day)"
    },
    "Jaffarabadi": {
        "description": "Largest buffalo breed",
        "key_features": ["Massive body", "Drooping horns", "Black coat", "Bulging forehead"],
        "origin": "Gujarat",
        "milk_yield": "High (12-15 liters/day)"
    },
    "Surti": {
        "description": "Compact dairy breed",
        "key_features": ["Small curved horns", "Black/brown coat", "Compact body", "Good temperament"],
        "origin": "Gujarat",
        "milk_yield": "Medium (8-10 liters/day)"
    },
    "Banni": {
        "description": "Hardy breed from Kutch region",
        "key_features": ["Long curved horns", "Black coat", "Adapted to harsh climate"],
        "origin": "Gujarat (Kutch)",
        "milk_yield": "Medium (6-8 liters/day)"
    },
    "Nili-Ravi": {
        "description": "High-yielding Pakistani breed",
        "key_features": ["Wall-eyed appearance", "Tightly curled horns", "Black coat with white markings"],
        "origin": "Pakistan (also in Punjab, India)",
        "milk_yield": "Very High (18-25 liters/day)"
    },
    "Pandharpuri": {
        "description": "Draught breed from Maharashtra",
        "key_features": ["Long horns", "Black coat", "Strong build", "Good for work"],
        "origin": "Maharashtra",
        "milk_yield": "Low (4-6 liters/day)"
    },
    "Toda": {
        "description": "Small hill breed",
        "key_features": ["Small size", "Curved horns", "Adapted to hills"],
        "origin": "Tamil Nadu (Nilgiris)",
        "milk_yield": "Low (2-4 liters/day)"
    },

    # Cattle Breeds (Add as needed)
    "Gir": {
        "description": "Indigenous dairy breed",
        "key_features": ["Large pendulous ears", "Convex forehead", "Reddish-brown coat", "Lyre-shaped horns"],
        "origin": "Gujarat",
        "milk_yield": "High (10-15 liters/day)"
    },
    "Sahiwal": {
        "description": "High-yielding dairy breed",
        "key_features": ["Reddish-brown coat", "Loose skin", "Short horns", "Drooping ears"],
        "origin": "Punjab, Pakistan",
        "milk_yield": "Very High (12-18 liters/day)"
    },
    "Red Sindhi": {
        "description": "Heat-tolerant dairy breed",
        "key_features": ["Red coat", "Compact body", "Small horns", "Good heat tolerance"],
        "origin": "Sindh, Pakistan",
        "milk_yield": "Medium (8-12 liters/day)"
    },
    "Tharparkar": {
        "description": "Dual-purpose breed",
        "key_features": ["White/grey coat", "Medium-sized horns", "Good draught ability"],
        "origin": "Rajasthan",
        "milk_yield": "Medium (6-10 liters/day)"
    }
}


class DecisionSupportEngine:
    """
    Decision Support Engine for Field Workers
    Provides structured recommendations based on confidence levels
    """

    # Decision thresholds
    ACCEPT_THRESHOLD = 0.70
    REVIEW_THRESHOLD = 0.50

    @staticmethod
    def make_decision(confidence: float, top_predictions: List[Dict],
                      image_count: int = 1) -> Dict:
        """
        Make decision based on confidence and predictions

        Returns:
            {
                'decision': 'ACCEPTED' | 'REVIEW' | 'REJECTED',
                'message': str,
                'recommendation': str,
                'confidence_level': str
            }
        """
        if confidence >= DecisionSupportEngine.ACCEPT_THRESHOLD:
            decision = "ACCEPTED"
            confidence_level = "HIGH"
            message = f"High confidence prediction ({confidence*100:.1f}%). Breed identification is reliable."
            recommendation = "Proceed with registration. No manual verification needed."

        elif confidence >= DecisionSupportEngine.REVIEW_THRESHOLD:
            decision = "REVIEW"
            confidence_level = "MEDIUM"
            message = f"Medium confidence ({confidence*100:.1f}%). Manual review recommended."

            # Check if top 2 predictions are close
            if len(top_predictions) >= 2:
                diff = top_predictions[0]['confidence'] - \
                    top_predictions[1]['confidence']
                if diff < 0.15:
                    recommendation = f"Top predictions are close. Consider: {top_predictions[0]['breed']} or {top_predictions[1]['breed']}. Verify physical characteristics."
                else:
                    recommendation = f"Most likely {top_predictions[0]['breed']}, but verify key features before confirming."
            else:
                recommendation = "Verify breed characteristics with expert or take additional photos."

        else:
            decision = "REJECTED"
            confidence_level = "LOW"
            message = f"Low confidence ({confidence*100:.1f}%). Cannot reliably identify breed."
            recommendation = "Take clearer photos from multiple angles. Ensure good lighting and full animal visibility. Consult veterinary expert."

        # Adjust for multi-image predictions
        if image_count > 1:
            message += f" (Based on {image_count} images)"
            if decision == "REVIEW" and image_count >= 3:
                recommendation += " Multiple images analyzed - confidence improved."

        return {
            'decision': decision,
            'message': message,
            'recommendation': recommendation,
            'confidence_level': confidence_level,
            'thresholds': {
                'accept': DecisionSupportEngine.ACCEPT_THRESHOLD,
                'review': DecisionSupportEngine.REVIEW_THRESHOLD
            }
        }

    @staticmethod
    def get_breed_reasoning(breed_name: str) -> str:
        """Get reasoning for breed prediction based on domain knowledge"""
        if breed_name in BREED_FEATURES:
            features = BREED_FEATURES[breed_name]
            reasoning = f"Identified as {breed_name} based on: "
            reasoning += ", ".join(features['key_features'][:3])
            reasoning += f". Origin: {features['origin']}. "
            reasoning += f"Expected milk yield: {features['milk_yield']}."
            return reasoning
        else:
            return f"Identified as {breed_name}. Verify physical characteristics for confirmation."


class HybridBreedPredictor:
    """
    Production-ready Hybrid Inference System
    Supports: Single/Multi-image prediction, Decision support, Domain intelligence
    """

    def __init__(self, model_path: str, class_names: List[str],
                 config: Dict, device: torch.device,
                 animal_type: str = 'buffalo'):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model
            class_names: List of breed names
            config: Configuration dictionary
            device: torch device
            animal_type: 'buffalo' or 'cattle' (for single-type models)
        """
        self.device = device
        self.class_names = class_names
        self.config = config
        self.animal_type = animal_type
        self.decision_engine = DecisionSupportEngine()

        # Load model
        from src.model import BuffaloBreedClassifier
        self.model = BuffaloBreedClassifier(len(class_names), config)

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Setup transforms
        img_size = config['image']['input_size']
        mean = config['image']['mean']
        std = config['image']['std']

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        logger.info(f"✓ Model loaded from {model_path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Animal Type: {animal_type}")
        logger.info(f"  Classes: {len(class_names)}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0)

        return image_tensor

    def predict_single(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """
        Predict breed for a single image

        Returns:
            List of predictions with breed name, confidence, and rank
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        top_probs, top_indices = torch.topk(
            probabilities, min(top_k, len(self.class_names)))

        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            breed_name = self.class_names[idx.item()]
            predictions.append({
                'rank': i + 1,
                'breed': breed_name,
                'confidence': prob.item(),
                'confidence_percent': f"{prob.item() * 100:.2f}%",
                'animal_type': self.animal_type
            })

        return predictions

    def predict_multi(self, image_paths: List[str], top_k: int = 3,
                      aggregation: str = 'average') -> Dict:
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
        all_predictions = []
        all_probabilities = []

        for img_path in image_paths:
            try:
                # Get probabilities for this image
                image_tensor = self.preprocess_image(img_path)
                image_tensor = image_tensor.to(self.device)

                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                all_probabilities.append(probabilities.cpu().numpy())

                # Get top prediction for voting
                top_idx = torch.argmax(probabilities).item()
                all_predictions.append(self.class_names[top_idx])

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        if not all_probabilities:
            raise ValueError("Failed to process any images")

        # Aggregate predictions
        if aggregation == 'average':
            # Average probabilities across all images
            avg_probs = np.mean(all_probabilities, axis=0)
            top_indices = np.argsort(avg_probs)[::-1][:top_k]

            aggregated_predictions = []
            for i, idx in enumerate(top_indices):
                breed_name = self.class_names[idx]
                aggregated_predictions.append({
                    'rank': i + 1,
                    'breed': breed_name,
                    'confidence': float(avg_probs[idx]),
                    'confidence_percent': f"{avg_probs[idx] * 100:.2f}%",
                    'animal_type': self.animal_type
                })

        elif aggregation == 'voting':
            # Majority voting
            vote_counts = Counter(all_predictions)
            total_votes = len(all_predictions)

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
                    'animal_type': self.animal_type
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
            'animal_type': self.animal_type,
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

    def predict_with_decision_support(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Single image prediction with full decision support

        Returns:
            Complete prediction with decision support and domain intelligence
        """
        predictions = self.predict_single(image_path, top_k)

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
            'animal_type': self.animal_type,
            'decision': decision_info['decision'],
            'decision_message': decision_info['message'],
            'recommendation': decision_info['recommendation'],
            'confidence_level': decision_info['confidence_level'],
            'reasoning': reasoning,
            'breed_info': breed_info
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


# Legacy class for backward compatibility
class BuffaloBreedPredictor(HybridBreedPredictor):
    """Legacy predictor class - redirects to HybridBreedPredictor"""

    def __init__(self, model_path: str, class_names: List[str],
                 config: Dict, device: torch.device):
        super().__init__(model_path, class_names, config, device, animal_type='buffalo')

    def predict(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """Legacy predict method"""
        return self.predict_single(image_path, top_k)

    def predict_with_threshold(self, image_path: str,
                               confidence_threshold: float = 0.5,
                               top_k: int = 3) -> Dict:
        """Legacy method with threshold"""
        result = self.predict_with_decision_support(image_path, top_k)
        result['high_confidence'] = result['confidence'] >= confidence_threshold
        result['top_prediction'] = result['final_prediction']
        result['top_confidence'] = result['confidence']
        if not result['high_confidence']:
            result['warning'] = f"Low confidence ({result['confidence_percent']}). Manual verification recommended."
        return result


def format_prediction_output(result: Dict, detailed: bool = True) -> str:
    """Format prediction results for display"""
    output = "\n" + "="*70 + "\n"
    output += "🐃 BREED IDENTIFICATION RESULTS\n"
    output += "="*70 + "\n\n"

    # Final prediction
    output += f"📌 FINAL PREDICTION: {result['final_prediction']}\n"
    output += f"   Confidence: {result['confidence_percent']} ({result['confidence_level']})\n"
    output += f"   Animal Type: {result['animal_type'].upper()}\n\n"

    # Decision
    decision_emoji = {"ACCEPTED": "✅", "REVIEW": "⚠️", "REJECTED": "❌"}
    output += f"{decision_emoji.get(result['decision'], '❓')} DECISION: {result['decision']}\n"
    output += f"   {result['decision_message']}\n\n"

    # Recommendation
    output += f"💡 RECOMMENDATION:\n"
    output += f"   {result['recommendation']}\n\n"

    if detailed:
        # Top predictions
        output += "📊 TOP PREDICTIONS:\n"
        for pred in result['top_predictions']:
            bar = "█" * int(pred['confidence'] * 40)
            output += f"   {pred['rank']}. {pred['breed']}: {pred['confidence_percent']} {bar}\n"
        output += "\n"

        # Reasoning
        output += f"🔍 REASONING:\n"
        output += f"   {result['reasoning']}\n\n"

        # Breed info
        if result.get('breed_info'):
            info = result['breed_info']
            output += f"📖 BREED INFORMATION:\n"
            output += f"   Description: {info.get('description', 'N/A')}\n"
            output += f"   Key Features: {', '.join(info.get('key_features', []))}\n"
            output += f"   Origin: {info.get('origin', 'N/A')}\n"
            output += f"   Milk Yield: {info.get('milk_yield', 'N/A')}\n\n"

    output += "="*70 + "\n"
    return output


def run_inference_demo(model_path: str, image_path: str,
                       class_names: List[str], config: Dict, device: torch.device):
    """Run inference demo on a single image"""
    predictor = HybridBreedPredictor(model_path, class_names, config, device)

    logger.info(f"\nRunning inference on: {image_path}")

    # Get predictions with decision support
    result = predictor.predict_with_decision_support(
        image_path,
        top_k=config['evaluation']['top_k']
    )

    # Display results
    print(format_prediction_output(result, detailed=True))

    return result
