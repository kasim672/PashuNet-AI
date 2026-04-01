# 📦 Installation Guide

Complete setup instructions for the Buffalo Breed Recognition System.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Dataset Setup](#dataset-setup)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [First Run](#first-run)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **GPU**: Optional (NVIDIA GPU with CUDA support recommended)

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16 GB
- **GPU**: NVIDIA GTX 1650 or better (4GB+ VRAM)
- **CUDA**: 11.8 or higher
- **Storage**: 15 GB free space

---

## Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/buffalo-breed-recognition.git
cd buffalo-breed-recognition
```

### Step 2: Create Virtual Environment

#### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install PyTorch

#### With CUDA (GPU Support)

**CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only
```bash
pip install torch torchvision torchaudio
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected output**:
```
PyTorch: 2.0.0+cu118
CUDA Available: True
```

---

## Dataset Setup

### Option 1: Use Existing Dataset

If you have the dataset:

```bash
# Ensure dataset structure:
dataset/
└── buffalo/
    ├── banni/
    ├── bargur/
    ├── bhadwari/
    └── ... (17 breeds total)
```

### Option 2: Download Dataset

```bash
# Download from your source
# Extract to dataset/ folder
# Verify structure matches above
```

### Verify Dataset

```bash
python -c "from pathlib import Path; breeds = [d.name for d in Path('dataset/buffalo').iterdir() if d.is_dir()]; print(f'Breeds found: {len(breeds)}'); print(breeds)"
```

**Expected output**:
```
Breeds found: 17
['banni', 'bargur', 'bhadwari', 'chhattisgarhi', 'chilika', 'gojri', 'jaffarabadi', 'kalahandi', 'luit', 'marathwada', 'mehsana', 'murrah', 'nagpuri', 'nili-ravi', 'pandharpuri', 'surti', 'toda']
```

---

## Configuration

### Review Configuration

Edit `config.yaml` if needed:

```yaml
# System mode
mode: "hybrid"  # Two-stage classification

# Dataset
dataset:
  root_dir: "dataset"  # Change if dataset is elsewhere
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Training
training:
  batch_size: 32       # Reduce to 16 or 8 if GPU memory issues
  num_epochs: 50       # Adjust based on time available
  learning_rate: 0.001

# Model
model:
  architecture: "mobilenet_v2"  # Options: mobilenet_v2, efficientnet_b0
  dropout: 0.3
```

### GPU Memory Settings

If you encounter CUDA out of memory errors:

```yaml
training:
  batch_size: 16  # or 8 for 4GB GPUs
```

---

## Verification

### Test Imports

```bash
python -c "from src.dataset import *; from src.model import *; from src.train import *; from src.inference import *; print('✓ All imports successful')"
```

### Test Configuration

```bash
python -c "from src.utils import load_config; config = load_config('config.yaml'); print('✓ Configuration loaded')"
```

### Test GPU

```bash
python -c "from src.utils import get_device; device = get_device(); print(f'Device: {device}')"
```

**Expected output**:
```
Device: cuda
```

Or if no GPU:
```
Device: cpu
```

---

## First Run

### 1. Train Model

```bash
python main.py train
```

**What it does**:
- Loads and prepares dataset
- Creates model architecture
- Trains for specified epochs
- Saves best model to `models/`
- Generates training plots in `plots/`

**Duration**: 20-35 minutes on GTX 1650

**Expected output**:
```
============================================================
Buffalo Breed Recognition - Training
============================================================
Device: cuda
Epochs: 50
Batch Size: 32

Step 1: Preparing hybrid datasets...
Dataset structure detected:
  Buffalo: dataset\buffalo
  Cattle: Not found

============================================================
Preparing Buffalo Breed Classification Dataset
============================================================

Dataset Analysis:
  Total Classes: 17
  Total Images: 5564
  Imbalance Ratio: 5.33

Data Split:
  Train: 3833 images
  Val: 822 images
  Test: 822 images

Step 2: Creating hybrid classification system...
✓ Buffalo breed classifier created: mobilenet_v2
  Num classes: 17

Step 3: Training hybrid system...
Training buffalo breed classifier...
Epoch 1/50: [Progress bar...]

[Training continues...]

HYBRID TRAINING PIPELINE COMPLETE!
Models saved to: models
```

### 2. Test Model

```bash
python main.py test
```

**What it does**:
- Loads trained model
- Tests on sample images
- Displays predictions

**Expected output**:
```
============================================================
Buffalo Breed Recognition - Testing
============================================================
Device: cuda
✓ Model loaded
  Buffalo breeds: 17

Testing on sample from: banni

Image: banni_001.jpg
Animal Type: BUFFALO
Prediction: Banni
Confidence: 87.45%
Decision: ACCEPTED

Top 3 Predictions:
  #1: Banni - 87.45%
  #2: Murrah - 8.32%
  #3: Mehsana - 2.15%

============================================================
Testing Complete!
============================================================
```

### 3. Start API

```bash
python main.py serve
```

**What it does**:
- Starts FastAPI server
- Loads trained model
- Serves REST API and web interface

**Expected output**:
```
============================================================
Buffalo Breed Recognition - API Server
============================================================
Host: 0.0.0.0
Port: 8000
Docs: http://localhost:8000/docs
Frontend: http://localhost:8000/frontend
============================================================
INFO:     Started server process
INFO:     Waiting for application startup.
✓ Model loaded successfully
  Device: cuda
  Mode: Two-stage (Animal Type → Breed)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Access**:
- API Docs: http://localhost:8000/docs
- Web Interface: http://localhost:8000/frontend

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```yaml
# In config.yaml
training:
  batch_size: 8  # Reduce batch size
```

Or use CPU:
```yaml
inference:
  device: "cpu"
```

### Issue: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'torch'
```

**Solution**:
```bash
# Ensure virtual environment is activated
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Dataset Not Found

**Symptoms**:
```
FileNotFoundError: dataset/buffalo not found
```

**Solution**:
```bash
# Verify dataset structure
ls dataset/buffalo/

# Should show 17 breed folders
# If not, check dataset path in config.yaml
```

### Issue: Slow Training

**Symptoms**:
- Training takes hours
- GPU not being used

**Solution**:
```bash
# Check if GPU is detected
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Or reduce epochs:
```yaml
# In config.yaml
training:
  num_epochs: 30  # Reduce from 50
```

### Issue: Module Not Found

**Symptoms**:
```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
```bash
# Ensure you're in project root directory
pwd  # Should show: .../buffalo-breed-recognition

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Issue: Permission Denied

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
```

**Solution**:
```bash
# Windows: Run as administrator
# Linux/Mac: Use sudo or fix permissions
chmod -R 755 dataset/
```

### Issue: API Won't Start

**Symptoms**:
```
Address already in use
```

**Solution**:
```bash
# Use different port
python main.py serve --port 8080

# Or kill process using port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

---

## Advanced Configuration

### Custom Dataset Location

```yaml
# config.yaml
dataset:
  root_dir: "/path/to/your/dataset"
```

### Custom Model Save Location

```yaml
# config.yaml
output:
  model_dir: "/path/to/save/models"
  plots_dir: "/path/to/save/plots"
  results_dir: "/path/to/save/results"
```

### Enable TensorBoard

```yaml
# config.yaml
logging:
  tensorboard: true
```

Then run:
```bash
tensorboard --logdir logs/
```

Access: http://localhost:6006

### Multi-GPU Training

```yaml
# config.yaml
training:
  use_multi_gpu: true
  gpu_ids: [0, 1]  # Use GPUs 0 and 1
```

### Custom Augmentation

```yaml
# config.yaml
augmentation:
  train:
    horizontal_flip: 0.5
    rotation_limit: 20  # Increase rotation
    brightness_contrast: 0.3  # Increase brightness variation
    blur_limit: 5  # Increase blur
```

---

## System Check Script

Create `check_system.py`:

```python
import torch
from pathlib import Path
from src.utils import load_config, get_device

print("="*60)
print("System Check")
print("="*60)

# Python version
import sys
print(f"Python: {sys.version.split()[0]}")

# PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Config
try:
    config = load_config('config.yaml')
    print(f"Config Loaded: ✓")
except Exception as e:
    print(f"Config Loaded: ✗ ({e})")

# Dataset
dataset_path = Path('dataset/buffalo')
if dataset_path.exists():
    breeds = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Dataset Found: ✓")
    print(f"Breeds: {len(breeds)}")
else:
    print(f"Dataset Found: ✗")

# Device
device = get_device()
print(f"Device: {device}")

print("="*60)
print("System check complete!")
print("="*60)
```

Run:
```bash
python check_system.py
```

---

## Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed (with CUDA if GPU available)
- [ ] Dependencies installed from requirements.txt
- [ ] Dataset downloaded and structured correctly
- [ ] Configuration reviewed and customized
- [ ] System check passed
- [ ] First training run successful
- [ ] API server starts without errors
- [ ] Web interface accessible

---

## Quick Commands Reference

```bash
# Environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Training
python main.py train
python main.py train --epochs 30
python main.py train --batch-size 16

# Testing
python main.py test
python test.py --dataset
python test.py --image buffalo.jpg

# Deployment
python main.py serve
python main.py serve --port 8080

# Verification
python -c "import torch; print(torch.cuda.is_available())"
python check_system.py
```

---

## Next Steps

After successful installation:

1. **Train your first model**: `python main.py train`
2. **Test predictions**: `python main.py test`
3. **Explore API**: `python main.py serve`
4. **Read documentation**: [README.md](README.md)
5. **Understand architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Getting Help

### Documentation
- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- [GitHub Issues](https://github.com/yourusername/buffalo-breed-recognition/issues) - Report bugs

### Common Issues
- Check logs in `logs/training.log`
- Review configuration in `config.yaml`
- Verify dataset structure
- Ensure GPU is detected (if using)

---

**Installation complete! You're ready to start recognizing buffalo breeds! 🐃**
