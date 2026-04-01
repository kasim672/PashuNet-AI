"""
Test script for Hybrid Two-Stage Classification System
"""

from src.model_hybrid import HybridClassificationSystem
from src.utils import load_config, get_device
import os
import sys
import torch
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))


def load_hybrid_system(config, model_dir='models/hybrid'):
    """Load trained hybrid system"""

    # Load metadata
    with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    buffalo_classes = metadata['num_buffalo_breeds']
    cattle_classes = metadata.get('num_cattle_breeds', None)

    # Create system
    system = HybridClassificationSystem(
        config, buffalo_classes, cattle_classes)

    # Load models
    system.load_models(model_dir)

    return system, metadata


def preprocess_image(image_path, config):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize(
            (config['image']['input_size'], config['image']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['image']['mean'], std=config['image']['std'])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor


def predict_hybrid(system, image_tensor, metadata, device):
    """Make prediction using hybrid system"""
    image_tensor = image_tensor.to(device)

    # Two-stage prediction
    animal_types, breed_logits_list, breed_probs_list = system.predict(
        image_tensor)

    # Extract results
    animal_type = animal_types[0].item()
    animal_name = 'buffalo' if animal_type == 0 else 'cattle'

    breed_probs = breed_probs_list[0].squeeze()
    top_k = min(3, len(breed_probs))
    top_probs, top_indices = torch.topk(breed_probs, k=top_k)

    # Get breed names
    if animal_type == 0:
        breed_names = metadata['buffalo_breeds']
    else:
        breed_names = metadata.get('cattle_breeds', [])

    results = {
        'animal_type': animal_name,
        'top_predictions': []
    }

    for i in range(top_k):
        breed_idx = top_indices[i].item()
        confidence = top_probs[i].item()

        if breed_idx < len(breed_names):
            results['top_predictions'].append({
                'rank': i + 1,
                'breed': breed_names[breed_idx],
                'confidence': confidence,
                'confidence_percent': f"{confidence*100:.2f}%"
            })

    return results


def main():
    """Test hybrid inference"""

    print("\n" + "="*60)
    print("Hybrid Two-Stage Classification System - Testing")
    print("="*60 + "\n")

    # Load configuration
    config = load_config('config.yaml')
    device = get_device()

    # Load hybrid system
    print("Loading hybrid system...")
    model_dir = 'models/hybrid'

    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory not found: {model_dir}")
        print("Please train the hybrid system first using: python main_hybrid.py")
        return

    system, metadata = load_hybrid_system(config, model_dir)
    print("✓ Hybrid system loaded successfully")
    print(f"  Buffalo breeds: {metadata['num_buffalo_breeds']}")
    if 'num_cattle_breeds' in metadata:
        print(f"  Cattle breeds: {metadata['num_cattle_breeds']}")

    # Test on sample images
    print("\n" + "="*60)
    print("Testing on sample images...")
    print("="*60 + "\n")

    # Find sample images
    dataset_root = Path(config['dataset']['root_dir'])

    # Try to find buffalo sample
    buffalo_dir = dataset_root / 'buffalo'
    if buffalo_dir.exists():
        buffalo_breeds = [d for d in buffalo_dir.iterdir() if d.is_dir()]
        if buffalo_breeds:
            sample_breed = buffalo_breeds[0]
            sample_images = list(sample_breed.glob('*.jpg')) + \
                list(sample_breed.glob('*.jpeg'))

            if sample_images:
                print(f"Testing buffalo image from: {sample_breed.name}")
                image_path = sample_images[0]

                # Preprocess and predict
                image_tensor = preprocess_image(str(image_path), config)
                results = predict_hybrid(
                    system, image_tensor, metadata, device)

                print(f"\nImage: {image_path.name}")
                print(f"Animal Type: {results['animal_type'].upper()}")
                print("\nTop Predictions:")
                for pred in results['top_predictions']:
                    print(
                        f"  #{pred['rank']}: {pred['breed']} - {pred['confidence_percent']}")

    # Try to find cattle sample
    cattle_dir = dataset_root / 'cattle'
    if cattle_dir.exists():
        cattle_breeds = [d for d in cattle_dir.iterdir() if d.is_dir()]
        if cattle_breeds:
            sample_breed = cattle_breeds[0]
            sample_images = list(sample_breed.glob('*.jpg')) + \
                list(sample_breed.glob('*.jpeg'))

            if sample_images:
                print(f"\n\nTesting cattle image from: {sample_breed.name}")
                image_path = sample_images[0]

                # Preprocess and predict
                image_tensor = preprocess_image(str(image_path), config)
                results = predict_hybrid(
                    system, image_tensor, metadata, device)

                print(f"\nImage: {image_path.name}")
                print(f"Animal Type: {results['animal_type'].upper()}")
                print("\nTop Predictions:")
                for pred in results['top_predictions']:
                    print(
                        f"  #{pred['rank']}: {pred['breed']} - {pred['confidence_percent']}")

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
