"""
Setup and verification script
"""

import subprocess
import sys
import torch
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")


def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ GPU not available - training will use CPU (slower)")
        return False


def check_dataset():
    """Check dataset structure"""
    dataset_path = Path("buffalo")

    if not dataset_path.exists():
        print("✗ Dataset folder 'buffalo' not found")
        return False

    breed_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if len(breed_dirs) == 0:
        print("✗ No breed folders found in dataset")
        return False

    print(f"✓ Dataset found: {len(breed_dirs)} breeds")

    total_images = 0
    for breed_dir in breed_dirs:
        images = list(breed_dir.glob('*.jpg')) + list(breed_dir.glob('*.jpeg')) + \
            list(breed_dir.glob('*.png'))
        total_images += len(images)

    print(f"  Total images: {total_images}")
    return True


def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'results', 'plots', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print(f"✓ Created directories: {', '.join(dirs)}")


def main():
    """Run setup verification"""
    print("="*60)
    print("Buffalo Breed Recognition - Setup Verification")
    print("="*60 + "\n")

    # Check Python
    print("1. Checking Python version...")
    check_python_version()
    print()

    # Check GPU
    print("2. Checking GPU availability...")
    check_gpu()
    print()

    # Check dataset
    print("3. Checking dataset...")
    dataset_ok = check_dataset()
    print()

    # Create directories
    print("4. Creating directories...")
    create_directories()
    print()

    # Summary
    print("="*60)
    print("Setup Verification Complete")
    print("="*60)

    if dataset_ok:
        print("\n✓ System ready for training!")
        print("\nNext steps:")
        print("  1. Review config.yaml for any customization")
        print("  2. Run: python main.py (to train the model)")
        print("  3. Run: python test_inference.py (to test predictions)")
        print("  4. Run: python api.py (to start API server)")
    else:
        print("\n⚠ Please fix dataset issues before training")


if __name__ == "__main__":
    main()
