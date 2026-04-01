# Installation Guide - Bharat Pashudhan App (BPA) Breed Recognition System

## 🎯 System Overview

Production-grade AI system for cattle and buffalo breed identification with:
- **Two-stage hybrid classification** (Animal Type → Breed)
- **Multi-image prediction** with aggregation
- **Decision support engine** for field workers
- **Domain intelligence** with breed-specific features
- **REST API** for integration
- **Web frontend** for easy access

---

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
  - RTX 1650 or better
  - 4GB+ VRAM

### Software Requirements
- **Python**: 3.10, 3.11, or 3.13
- **CUDA**: 11.8 or 12.1 (if using GPU)
- **Git**: For cloning repository

---

## 🚀 Installation Steps

### Step 1: Check Python Version

```bash
# Windows
py -0

# Linux/Mac
python3 --version
```

**Expected Output:**
```
-V:3.13  Python 3.13 (64-bit)
-V:3.12  Python 3.12 (64-bit)
-V:3.11  Python 3.11 (64-bit)
-V:3.10  Python 3.10 (64-bit)
```

✅ **Recommended**: Python 3.10, 3.11, or 3.13

---

### Step 2: Clone Repository (if applicable)

```bash
git clone <repository-url>
cd breed_recognition_for_cattle_and_buffaloes
```

---

### Step 3: Create Virtual Environment

#### Windows:
```bash
# Using Python 3.13 (or your version)
py -3.13 -m venv venv

# Activate
venv\Scripts\activate
```

#### Linux/Mac:
```bash
python3 -m venv venv

# Activate
source venv/bin/activate
```

**Verify activation:**
```bash
# You should see (venv) in your prompt
(venv) C:\...\breed_recognition>
```

---

### Step 4: Install PyTorch with CUDA Support

#### For GPU (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For GPU (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU Only:
```bash
pip install torch torchvision torchaudio
```

**Verify PyTorch installation:**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.7.1+cu118
CUDA Available: True
```

---

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

**This installs:**
- OpenCV (image processing)
- Albumentations (augmentation)
- scikit-learn (ML utilities)
- FastAPI & Uvicorn (API server)
- Matplotlib & Seaborn (visualization)
- Pandas (data handling)
- Grad-CAM (explainability)
- And more...

---

### Step 6: Verify Installation

Run the setup verification script:

```bash
python setup.py
```

**Expected Output:**
```
============================================================
Buffalo Breed Recognition - Setup Verification
============================================================

1. Checking Python version...
Python Version: 3.13.7
✓ Python version compatible

2. Checking GPU availability...
✓ GPU Available: NVIDIA GeForce GTX 1650
  CUDA Version: 11.8
  Memory: 4.29 GB

3. Checking dataset...
✓ Dataset found: 17 breeds
  Total images: 2785

4. Creating directories...
✓ Created directories: models, results, plots, logs

============================================================
Setup Verification Complete
============================================================

✓ System ready for training!
```

---

## 📁 Dataset Structure

### For Hybrid Classification (Cattle + Buffalo):

```
dataset/
├── buffalo/
│   ├── Murrah/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Mehsana/
│   ├── Jaffarabadi/
│   ├── Surti/
│   ├── Banni/
│   └── ... (other buffalo breeds)
│
└── cattle/
    ├── Gir/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Sahiwal/
    ├── Red_Sindhi/
    ├── Tharparkar/
    └── ... (other cattle breeds)
```

### For Buffalo-Only (Legacy):

```
buffalo/
├── Murrah/
├── Mehsana/
├── Jaffarabadi/
└── ... (buffalo breeds)
```

**Supported Image Formats:**
- `.jpg`, `.jpeg`, `.png`, `.JPG`, `.JPEG`

**Image Requirements:**
- Minimum resolution: 224x224 pixels
- Clear visibility of animal
- Good lighting (augmentation handles variations)
- Multiple angles recommended

---

## 🎓 Training the Model

### Option 1: Single Mode (Buffalo Only)

```bash
python main.py
```

**Configuration** (`config.yaml`):
```yaml
dataset:
  root_dir: "buffalo"  # Buffalo-only dataset
```

### Option 2: Hybrid Mode (Cattle + Buffalo)

**Update `config.yaml`:**
```yaml
dataset:
  root_dir: "dataset"  # Hybrid dataset
  mode: "hybrid"       # Enable hybrid mode
```

**Run training:**
```bash
python main_hybrid.py  # Use hybrid training script
```

### Training Output:

```
============================================================
Starting Training
============================================================

Epoch 1/50
Training: 100%|████████████| 87/87 [02:15<00:00]
Validation: 100%|████████████| 19/19 [00:25<00:00]

Train Loss: 2.1234, Train Acc: 0.4521
Val Loss: 1.8765, Val Acc: 0.5234
✓ Best model saved! Val Acc: 0.5234

...

Epoch 50/50
Train Loss: 0.2134, Train Acc: 0.9421
Val Loss: 0.3456, Val Acc: 0.8934

============================================================
Training Complete! Best Val Acc: 0.9234
============================================================
```

**Training Time:**
- **With GPU**: 30-45 minutes
- **With CPU**: 3-4 hours

---

## 🔮 Testing Inference

### Test on Sample Images:

```bash
python test_inference.py
```

### Programmatic Usage:

```python
from src.utils import load_config, get_device
from src.inference import HybridBreedPredictor
import json

# Load configuration
config = load_config('config.yaml')
device = get_device()

# Load class names
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Initialize predictor
predictor = HybridBreedPredictor(
    model_path='models/best_model.pth',
    class_names=class_names,
    config=config,
    device=device,
    animal_type='buffalo'
)

# Single image prediction
result = predictor.predict_with_decision_support('path/to/image.jpg')
print(f"Prediction: {result['final_prediction']}")
print(f"Confidence: {result['confidence_percent']}")
print(f"Decision: {result['decision']}")

# Multi-image prediction (aggregated)
result = predictor.predict_multi(
    ['image1.jpg', 'image2.jpg', 'image3.jpg'],
    aggregation='average'
)
print(f"Final Prediction: {result['final_prediction']}")
print(f"Confidence: {result['confidence_percent']}")
print(f"Decision: {result['decision']}")
```

---

## 🌐 Starting the API Server

### Basic API:

```bash
python api.py
```

**Server runs at:** `http://localhost:8000`

### Hybrid API (Cattle + Buffalo):

```bash
python api_hybrid.py
```

### API Endpoints:

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **List Breeds**
   ```bash
   curl http://localhost:8000/breeds
   ```

3. **Single Image Prediction**
   ```bash
   curl -X POST "http://localhost:8000/predict_single" \
     -F "file=@path/to/image.jpg"
   ```

4. **Multi-Image Prediction**
   ```bash
   curl -X POST "http://localhost:8000/predict_multi" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
   ```

5. **Interactive Documentation**
   - Open browser: `http://localhost:8000/docs`

---

## 🖥️ Web Frontend

### Start Frontend Server:

```bash
python -m http.server 8080 --directory frontend
```

**Access at:** `http://localhost:8080`

### Features:
- Upload single or multiple images
- Real-time breed prediction
- Decision support visualization
- Breed information display
- Confidence meter
- Download results as JSON

---

## 🔧 Configuration

### Edit `config.yaml`:

```yaml
# Dataset Configuration
dataset:
  root_dir: "dataset"  # or "buffalo" for single mode
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42

# Model Configuration
model:
  architecture: "mobilenet_v2"  # or efficientnet_b0, efficientnet_b2
  pretrained: true
  dropout: 0.3

# Training Configuration
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  use_weighted_loss: true

# Deployment Configuration
deployment:
  api_host: "0.0.0.0"
  api_port: 8000
  max_image_size: 10485760  # 10MB
```

---

## 🐛 Troubleshooting

### Issue 1: GPU Not Detected

**Check CUDA:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution:**
- Install CUDA toolkit from NVIDIA
- Reinstall PyTorch with correct CUDA version
- Check GPU drivers

### Issue 2: Out of Memory

**Solution:**
- Reduce `batch_size` in `config.yaml` (try 16 or 8)
- Use smaller model: `mobilenet_v2` instead of `efficientnet_b2`
- Close other GPU applications

### Issue 3: Module Not Found

**Solution:**
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt

# Or install specific package
pip install <package-name>
```

### Issue 4: Low Accuracy

**Solutions:**
- Check data quality and labels
- Increase training epochs (50 → 100)
- Use data augmentation (already enabled)
- Collect more training data
- Try different model architecture

### Issue 5: API Not Starting

**Solution:**
```bash
# Check if port is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Use different port
uvicorn api:app --host 0.0.0.0 --port 8001
```

---

## 📊 Expected Performance

After training, you should achieve:

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 85-95% |
| Precision | 85-93% |
| Recall | 83-92% |
| F1-Score | 84-92% |
| Inference Time (GPU) | 50-100ms |
| Inference Time (CPU) | 200-500ms |
| Model Size | ~14MB |

---

## 🚀 Quick Start Commands

```bash
# 1. Setup
py -3.13 -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Verify
python setup.py

# 3. Train
python main.py

# 4. Test
python test_inference.py

# 5. Start API
python api.py

# 6. Start Frontend
python -m http.server 8080 --directory frontend
```

---

## 📞 Support

For issues:
1. Check logs: `logs/training.log`
2. Review configuration: `config.yaml`
3. Verify dataset structure
4. Check GPU availability
5. Review error messages

---

## 📚 Additional Resources

- **Documentation**: See `README.md`
- **Quick Start**: See `QUICKSTART.md`
- **Dataset Upgrade**: See `DATASET_UPGRADE_SUMMARY.md`
- **API Documentation**: `http://localhost:8000/docs` (when server is running)

---

## ✅ Installation Checklist

- [ ] Python 3.10/3.11/3.13 installed
- [ ] Virtual environment created and activated
- [ ] PyTorch with CUDA installed
- [ ] All dependencies installed
- [ ] Dataset structure verified
- [ ] Setup verification passed
- [ ] GPU detected (if available)
- [ ] Model training completed
- [ ] API server tested
- [ ] Frontend accessible

---

**System Status**: ✅ READY FOR PRODUCTION

**Last Updated**: 2026-04-01
