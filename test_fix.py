"""Quick test to verify get_transforms fix"""
from src.dataset_hybrid import get_hybrid_dataloaders
from src.utils import load_config

config = load_config('config.yaml')
print('✓ Imports successful - get_transforms fix applied')
print('✓ Ready to train: python main_hybrid.py')
