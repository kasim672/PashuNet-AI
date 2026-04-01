"""
Main training script for Hybrid Two-Stage Classification System
Trains: Binary Classifier + Buffalo Breed Classifier + Cattle Breed Classifier
"""

from src.evaluate import plot_training_history
from src.train_hybrid import train_hybrid_system
from src.model_hybrid import create_hybrid_system
from src.dataset_hybrid import get_hybrid_dataloaders
from src.utils import setup_logging, load_config, set_seed, get_device, create_directories
import os
import sys
import torch
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


def main():
    """Main hybrid training pipeline"""

    # Load configuration
    config = load_config('config.yaml')

    # Setup
    logger = setup_logging(config['logging']['log_dir'])
    set_seed(config['dataset']['random_seed'])
    device = get_device()
    create_directories(config)

    logger.info("\n" + "="*60)
    logger.info("Hybrid Two-Stage Classification System - Training")
    logger.info("="*60 + "\n")

    # Step 1: Prepare datasets
    logger.info("Step 1: Preparing hybrid datasets...")
    dataloaders = get_hybrid_dataloaders(config)

    # Extract class counts
    buffalo_classes = dataloaders['buffalo_breed']['analysis']['num_classes']
    cattle_classes = dataloaders.get('cattle_breed', {}).get(
        'analysis', {}).get('num_classes', None)

    logger.info(f"\nDataset Summary:")
    logger.info(f"  Buffalo Breeds: {buffalo_classes}")
    if cattle_classes:
        logger.info(f"  Cattle Breeds: {cattle_classes}")
    if 'binary' in dataloaders:
        logger.info(f"  Binary Classes: 2 (buffalo, cattle)")

    # Step 2: Create hybrid system
    logger.info("\nStep 2: Creating hybrid classification system...")
    hybrid_system = create_hybrid_system(
        config, buffalo_classes, cattle_classes)

    # Step 3: Train hybrid system
    logger.info("\nStep 3: Training hybrid system...")
    save_dir = os.path.join(config['output']['model_dir'], 'hybrid')
    results = train_hybrid_system(hybrid_system, dataloaders, config, save_dir)

    # Step 4: Plot training histories
    logger.info("\nStep 4: Plotting training histories...")
    plots_dir = os.path.join(config['output']['plots_dir'], 'hybrid')
    os.makedirs(plots_dir, exist_ok=True)

    for model_name, model_results in results.items():
        plot_training_history(
            model_results['history'], os.path.join(plots_dir, model_name))
        logger.info(f"✓ {model_name} training history saved")

    # Step 5: Save class names and metadata
    logger.info("\nStep 5: Saving metadata...")
    metadata = {
        'buffalo_breeds': dataloaders['buffalo_breed']['analysis']['class_names'],
        'num_buffalo_breeds': buffalo_classes
    }

    if 'cattle_breed' in dataloaders:
        metadata['cattle_breeds'] = dataloaders['cattle_breed']['analysis']['class_names']
        metadata['num_cattle_breeds'] = cattle_classes

    if 'binary' in dataloaders:
        metadata['binary_classes'] = ['buffalo', 'cattle']

    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"✓ Metadata saved to {save_dir}/metadata.json")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("HYBRID TRAINING PIPELINE COMPLETE!")
    logger.info("="*60)

    for model_name, model_results in results.items():
        logger.info(
            f"{model_name.upper()}: Best Val Acc = {model_results['best_val_acc']:.4f}")

    logger.info(f"\nModels saved to: {save_dir}")
    logger.info(f"Plots saved to: {plots_dir}")
    logger.info(f"Metadata saved to: {save_dir}/metadata.json")
    logger.info("="*60 + "\n")

    logger.info("\n🎉 System is ready for deployment!")
    logger.info("\nNext steps:")
    logger.info("  1. Test inference: python test_inference_hybrid.py")
    logger.info("  2. Start API: python api.py")
    logger.info("  3. Open frontend: http://localhost:8000/frontend")


if __name__ == "__main__":
    main()
