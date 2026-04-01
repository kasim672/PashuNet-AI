# 🚀 Quick Command Reference

## Training

### Train Hybrid System (Recommended)
```bash
python main_hybrid.py
```
Trains two-stage classification: Animal Type → Breed

### Train Legacy System (Deprecated)
```bash
python main.py
```
Single-stage buffalo-only classification (not recommended)

---

## Testing

### Test Hybrid Inference
```bash
python test_inference_hybrid.py
```
Tests trained hybrid models on sample images

---

## Deployment

### Start API Server
```bash
python api.py
```
Starts FastAPI server on http://localhost:8000

### Access API Documentation
```
http://localhost:8000/docs
```
Interactive Swagger UI documentation

### Access Web Frontend
```
http://localhost:8000/frontend
```
Web interface for breed prediction

---

## Development

### Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Check Python Files
```bash
python -m py_compile <filename>.py
```

### Run Diagnostics
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### List Breeds
```bash
curl http://localhost:8000/breeds
```

### Single Image Prediction
```bash
curl -X POST "http://localhost:8000/predict_single" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Multi-Image Prediction
```bash
curl -X POST "http://localhost:8000/predict_multi" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "aggregation=average"
```

---

## File Locations

### Models
```
models/hybrid/
├── binary_classifier.pth
├── buffalo_classifier.pth
├── cattle_classifier.pth
└── metadata.json
```

### Logs
```
logs/
└── training_YYYYMMDD_HHMMSS.log
```

### Plots
```
plots/hybrid/
├── binary_classifier_history.png
├── buffalo_classifier_history.png
└── cattle_classifier_history.png
```

### Dataset
```
dataset/
└── buffalo/
    ├── banni/
    ├── bargur/
    └── ...
```

---

## Configuration

### Edit Config
```bash
notepad config.yaml  # Windows
nano config.yaml     # Linux
```

### Key Settings
```yaml
mode: "hybrid"                    # System mode
dataset.root_dir: "dataset"       # Dataset location
training.batch_size: 32           # Batch size (reduce if OOM)
training.num_epochs: 50           # Training epochs
model.architecture: "mobilenet_v2" # Model architecture
```

---

## Troubleshooting

### Check GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Check Dataset
```bash
python -c "from pathlib import Path; print(list(Path('dataset/buffalo').iterdir()))"
```

### Clear Cache
```bash
# Windows
rmdir /s /q __pycache__
del /s *.pyc

# Linux/Mac
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## Quick Workflow

### Complete Training → Testing → Deployment
```bash
# 1. Train
python main_hybrid.py

# 2. Test
python test_inference_hybrid.py

# 3. Deploy
python api.py
```

### Development Cycle
```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Make changes
# ... edit files ...

# 3. Test syntax
python -m py_compile src/your_file.py

# 4. Run training
python main_hybrid.py
```

---

## Environment Setup

### Create Virtual Environment
```bash
python -m venv venv
```

### Install PyTorch (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install Other Dependencies
```bash
pip install fastapi uvicorn opencv-python albumentations scikit-learn pyyaml pillow
```

---

## Useful Python Snippets

### Check Model Size
```python
import torch
model = torch.load('models/hybrid/buffalo_classifier.pth')
print(f"Model size: {sum(p.numel() for p in model.values()):,} parameters")
```

### List Dataset Classes
```python
from pathlib import Path
classes = [d.name for d in Path('dataset/buffalo').iterdir() if d.is_dir()]
print(f"Classes: {len(classes)}")
print(classes)
```

### Test Single Prediction
```python
from src.inference_hybrid import HybridBreedPredictor
from src.utils import load_config, get_device

config = load_config('config.yaml')
device = get_device()
predictor = HybridBreedPredictor('models/hybrid', config, device)

result = predictor.predict_with_decision_support('path/to/image.jpg')
print(result)
```

---

**Last Updated**: 2026-04-01
