"""
Buffalo Breed Recognition System - Unified Entry Point
Two-stage hybrid classification with decision support
"""

import logging
from src.inference import BreedPredictor
from src.train import train_system
from src.utils import load_config, setup_logging, set_seed, get_device, create_directories
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


logger = logging.getLogger(__name__)


def train(args):
    """Train the breed recognition system"""
    config = load_config(args.config)

    # Override config with CLI args
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Setup
    setup_logging(config['logging']['log_dir'])
    set_seed(config['dataset']['random_seed'])
    device = get_device()
    create_directories(config)

    logger.info("="*60)
    logger.info("Buffalo Breed Recognition - Training")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")

    # Train
    train_system(config, device)

    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Models saved to: {config['output']['model_dir']}")
    logger.info(f"Next: python main.py test")


def test(args):
    """Test the trained model"""
    config = load_config(args.config)
    device = get_device()

    print("\n" + "="*60)
    print("Buffalo Breed Recognition - Testing")
    print("="*60 + "\n")

    # Load model
    model_dir = Path(config['output']['model_dir'])
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        print("Please train the model first: python main.py train")
        return

    predictor = BreedPredictor(str(model_dir), config, device)
    print(f"✓ Model loaded successfully")
    print(f"  Buffalo breeds: {len(predictor.buffalo_breeds)}")
    if predictor.cattle_breeds:
        print(f"  Cattle breeds: {len(predictor.cattle_breeds)}")

    # Test on sample images
    dataset_root = Path(config['dataset']['root_dir'])
    buffalo_dir = dataset_root / 'buffalo'

    if buffalo_dir.exists():
        breeds = [d for d in buffalo_dir.iterdir() if d.is_dir()]
        if breeds:
            sample_breed = breeds[0]
            images = list(sample_breed.glob('*.jpg')) + \
                list(sample_breed.glob('*.jpeg'))

            if images:
                print(f"\nTesting on sample from: {sample_breed.name}")
                result = predictor.predict_with_decision_support(
                    str(images[0]))

                print(f"\nImage: {images[0].name}")
                print(f"Animal Type: {result['animal_type'].upper()}")
                print(f"Prediction: {result['final_prediction']}")
                print(f"Confidence: {result['confidence_percent']}")
                print(f"Decision: {result['decision']}")
                print(f"\nTop 3 Predictions:")
                for pred in result['top_predictions'][:3]:
                    print(
                        f"  #{pred['rank']}: {pred['breed']} - {pred['confidence_percent']}")

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60 + "\n")


def serve(args):
    """Start the API server"""
    import uvicorn
    from api.app import app

    config = load_config(args.config)
    host = args.host or config['deployment']['api_host']
    port = args.port or config['deployment']['api_port']

    print("="*60)
    print("Buffalo Breed Recognition - API Server")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://localhost:{port}/docs")
    print(f"Frontend: http://localhost:{port}/frontend")
    print("="*60)

    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Buffalo Breed Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train model
  python main.py train --epochs 30        # Train with custom epochs
  python main.py test                     # Test model
  python main.py serve                    # Start API server
  python main.py serve --port 8080        # Custom port
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--config', default='config.yaml', help='Config file path')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument(
        '--config', default='config.yaml', help='Config file path')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument(
        '--config', default='config.yaml', help='Config file path')
    serve_parser.add_argument('--host', help='Server host')
    serve_parser.add_argument('--port', type=int, help='Server port')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'serve':
        serve(args)


if __name__ == "__main__":
    main()
