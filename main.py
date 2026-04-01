"""
Main training script for Buffalo Breed Recognition System
"""

from src.evaluate import evaluate_model, plot_training_history, analyze_misclassifications
from src.train import train_model
from src.model import create_model, get_optimizer, get_scheduler, get_loss_function
from src.dataset import prepare_data, get_dataloaders
from src.utils import (
    setup_logging, load_config, set_seed,
    get_device, create_directories
)
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


def main():
    """Main training pipeline"""

    # Load configuration
    config = load_config('config.yaml')

    # Setup
    logger = setup_logging(config['logging']['log_dir'])
    set_seed(config['dataset']['random_seed'])
    device = get_device()
    create_directories(config)

    logger.info("\n" + "="*60)
    logger.info("Buffalo Breed Recognition System - Training")
    logger.info("="*60 + "\n")

    # Prepare datasets
    logger.info("Step 1: Preparing datasets...")
    train_dataset, val_dataset, test_dataset, analysis = prepare_data(config)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )

    # Calculate class weights for imbalanced data
    class_counts = [analysis['class_counts'][name]
                    for name in analysis['class_names']]
    class_weights = None
    if config['training']['use_weighted_loss']:
        class_weights = torch.FloatTensor(
            [1.0 / count for count in class_counts]
        ).to(device)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        logger.info("✓ Class weights calculated for imbalanced data")

    # Create model
    logger.info("\nStep 2: Creating model...")
    num_classes = analysis['num_classes']
    model = create_model(num_classes, config, device)

    # Setup training components
    logger.info("\nStep 3: Setting up training components...")
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    criterion = get_loss_function(config, class_weights)

    # Train model
    logger.info("\nStep 4: Training model...")
    save_path = os.path.join(config['output']['model_dir'], 'best_model.pth')
    history, best_val_acc = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        device, config, save_path
    )

    # Plot training history
    logger.info("\nStep 5: Plotting training history...")
    plot_training_history(history, config['output']['plots_dir'])

    # Load best model for evaluation
    logger.info("\nStep 6: Evaluating on test set...")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    metrics, cm = evaluate_model(
        model, test_loader, device,
        analysis['class_names'],
        config['output']['results_dir']
    )

    # Analyze misclassifications
    logger.info("\nStep 7: Analyzing misclassifications...")
    misclassified = analyze_misclassifications(
        model, test_loader, device,
        analysis['class_names'],
        config['output']['results_dir']
    )

    # Save class names for inference
    import json
    with open(os.path.join(config['output']['model_dir'], 'class_names.json'), 'w') as f:
        json.dump(analysis['class_names'], f, indent=4)

    logger.info("\n" + "="*60)
    logger.info("Training Pipeline Complete!")
    logger.info("="*60)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"\nModel saved to: {save_path}")
    logger.info(f"Results saved to: {config['output']['results_dir']}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
