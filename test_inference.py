"""
Test inference on sample images
"""

import sys
import json
import torch
from pathlib import Path

from src.utils import load_config, get_device, setup_logging
from src.inference import run_inference_demo


def main():
    """Test inference on sample images"""

    # Setup
    config = load_config('config.yaml')
    logger = setup_logging(config['logging']['log_dir'])
    device = get_device()

    # Load class names
    class_names_path = Path(config['output']['model_dir']) / 'class_names.json'
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # Model path
    model_path = Path(config['output']['model_dir']) / 'best_model.pth'

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first using: python main.py")
        return

    # Get sample images from each breed
    dataset_root = Path(config['dataset']['root_dir'])
    sample_images = []

    for breed_dir in sorted(dataset_root.iterdir()):
        if breed_dir.is_dir():
            images = list(breed_dir.glob('*.jpg')) + \
                list(breed_dir.glob('*.jpeg'))
            if images:
                sample_images.append(images[0])

    if not sample_images:
        logger.error("No sample images found")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Inference on {len(sample_images)} Sample Images")
    logger.info(f"{'='*60}\n")

    # Run inference on samples
    for img_path in sample_images[:5]:  # Test on first 5
        result = run_inference_demo(
            str(model_path),
            str(img_path),
            class_names,
            config,
            device
        )

        print(f"\nActual breed (from folder): {img_path.parent.name}")
        print(f"Predicted breed: {result['top_prediction']}")
        print(
            f"Match: {'✓' if img_path.parent.name == result['top_prediction'] else '✗'}")
        print("-" * 60)


if __name__ == "__main__":
    main()
