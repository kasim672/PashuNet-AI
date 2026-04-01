"""
Unified Testing Script for Buffalo Breed Recognition System
"""

import argparse
from src.inference import BreedPredictor
from src.utils import load_config, get_device
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_single_image(predictor, image_path):
    """Test single image prediction"""
    print(f"\n{'='*60}")
    print("Single Image Prediction Test")
    print('='*60)

    result = predictor.predict_with_decision_support(image_path)

    print(f"\nImage: {Path(image_path).name}")
    print(f"Animal Type: {result['animal_type'].upper()}")
    print(f"Prediction: {result['final_prediction']}")
    print(f"Confidence: {result['confidence_percent']}")
    print(f"Decision: {result['decision']}")
    print(f"Recommendation: {result['recommendation']}")

    print(f"\nTop Predictions:")
    for pred in result['top_predictions']:
        print(
            f"  #{pred['rank']}: {pred['breed']} - {pred['confidence_percent']}")


def test_multi_image(predictor, image_paths):
    """Test multi-image prediction"""
    print(f"\n{'='*60}")
    print("Multi-Image Prediction Test")
    print('='*60)

    result = predictor.predict_multi(image_paths, aggregation='average')

    print(f"\nImages Processed: {result['images_processed']}")
    print(f"Aggregation: {result['aggregation_method']}")
    print(f"Animal Type: {result['animal_type'].upper()}")
    print(f"Final Prediction: {result['final_prediction']}")
    print(f"Confidence: {result['confidence_percent']}")
    print(f"Decision: {result['decision']}")

    print(f"\nTop Predictions:")
    for pred in result['top_predictions']:
        print(
            f"  #{pred['rank']}: {pred['breed']} - {pred['confidence_percent']}")


def test_dataset_samples(predictor, config):
    """Test on sample images from dataset"""
    print(f"\n{'='*60}")
    print("Dataset Sample Testing")
    print('='*60)

    dataset_root = Path(config['dataset']['root_dir'])
    buffalo_dir = dataset_root / 'buffalo'

    if not buffalo_dir.exists():
        print("❌ Buffalo dataset not found")
        return

    breeds = [d for d in buffalo_dir.iterdir() if d.is_dir()]
    if not breeds:
        print("❌ No breed folders found")
        return

    # Test first 3 breeds
    for breed_dir in breeds[:3]:
        images = list(breed_dir.glob('*.jpg')) + list(breed_dir.glob('*.jpeg'))
        if images:
            print(f"\n--- Testing: {breed_dir.name} ---")
            result = predictor.predict_with_decision_support(str(images[0]))
            print(f"Prediction: {result['final_prediction']}")
            print(f"Confidence: {result['confidence_percent']}")
            print(f"Decision: {result['decision']}")

            # Check if prediction matches actual breed
            if result['final_prediction'].lower() == breed_dir.name.lower():
                print("✓ Correct prediction!")
            else:
                print(f"✗ Incorrect (Expected: {breed_dir.name})")


def main():
    parser = argparse.ArgumentParser(
        description="Test Buffalo Breed Recognition System")
    parser.add_argument('--config', default='config.yaml',
                        help='Config file path')
    parser.add_argument('--image', help='Path to single image for testing')
    parser.add_argument('--images', nargs='+',
                        help='Paths to multiple images for testing')
    parser.add_argument('--dataset', action='store_true',
                        help='Test on dataset samples')

    args = parser.parse_args()

    # Load config and model
    config = load_config(args.config)
    device = get_device()

    print("="*60)
    print("Buffalo Breed Recognition - Testing")
    print("="*60)
    print(f"Device: {device}")

    # Load predictor
    model_dir = Path(config['output']['model_dir'])
    if not model_dir.exists():
        print(f"\n❌ Error: Model directory not found: {model_dir}")
        print("Please train the model first: python main.py train")
        return

    predictor = BreedPredictor(str(model_dir), config, device)
    print(f"✓ Model loaded")
    print(f"  Buffalo breeds: {len(predictor.buffalo_breeds)}")
    if predictor.cattle_breeds:
        print(f"  Cattle breeds: {len(predictor.cattle_breeds)}")

    # Run tests
    if args.image:
        test_single_image(predictor, args.image)
    elif args.images:
        test_multi_image(predictor, args.images)
    elif args.dataset:
        test_dataset_samples(predictor, config)
    else:
        # Default: test on dataset samples
        test_dataset_samples(predictor, config)

    print(f"\n{'='*60}")
    print("Testing Complete!")
    print('='*60 + "\n")


if __name__ == "__main__":
    main()
