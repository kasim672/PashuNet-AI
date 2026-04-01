# 🐃 Buffalo Breed Recognition System

AI-powered buffalo breed identification using deep learning with two-stage classification, multi-image prediction, and intelligent decision support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

---

## 🎯 Overview

Production-grade AI system for identifying Indian buffalo breeds with advanced features:

- **Two-Stage Classification**: Animal type detection (cattle/buffalo) → Breed identification
- **17 Buffalo Breeds**: Comprehensive Indian buffalo breed recognition
- **Multi-Image Prediction**: Aggregate predictions from multiple images for higher accuracy
- **Decision Support System**: ACCEPTED/REVIEW/REJECTED recommendations with confidence analysis
- **Domain Intelligence**: Breed-specific features and characteristics database
- **REST API**: Production-ready FastAPI server with interactive documentation
- **Web Interface**: User-friendly frontend for image upload and prediction
- **GPU Acceleration**: CUDA support for fast inference

---

## 📊 Supported Breeds

**Buffalo (17 breeds)**:
Banni, Bargur, Bhadwari, Chhattisgarhi, Chilika, Gojri, Jaffarabadi, Kalahandi, Luit, Marathwada, Mehsana, Murrah, Nagpuri, Nili-Ravi, Pandharpuri, Surti, Toda

**Dataset**: 5,564 images across 17 breeds

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/buffalo-breed-recognition.git
cd buffalo-breed-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

### Training

```bash
# Train with default settings
python main.py train

# Train with custom parameters
python main.py train --epochs 30 --batch-size 16
```

### Testing

```bash
# Test on dataset samples
python main.py test

# Test on specific image
python test.py --image path/to/image.jpg

# Test multi-image prediction
python test.py --images img1.jpg img2.jpg img3.jpg
```

### Deployment

```bash
# Start API server
python main.py serve

# Custom port
python main.py serve --port 8080
```

Access:
- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/frontend

---

## 📖 Usage Examples

### Command Line Interface

```bash
# Training
python main.py train              # Train with default settings
python main.py train --epochs 30  # Custom epochs
python main.py train --batch-size 16  # Custom batch size

# Testing
python main.py test               # Test on dataset samples
python test.py --dataset          # Test on all dataset samples
python test.py --image buffalo.jpg  # Test single image
python test.py --images img1.jpg img2.jpg img3.jpg  # Multi-image

# Deployment
python main.py serve              # Start API server
python main.py serve --port 8080  # Custom port
```

### Python API

```python
from src.inference import BreedPredictor
from src.utils import load_config, get_device

# Load model
config = load_config('config.yaml')
device = get_device()
predictor = BreedPredictor('models', config, device)

# Single image prediction
result = predictor.predict_with_decision_support('image.jpg')
print(f"Breed: {result['final_prediction']}")
print(f"Confidence: {result['confidence_percent']}")
print(f"Decision: {result['decision']}")

# Multi-image prediction
result = predictor.predict_multi(['img1.jpg', 'img2.jpg', 'img3.jpg'])
print(f"Aggregated Prediction: {result['final_prediction']}")
```

### REST API

```bash
# Single image prediction
curl -X POST "http://localhost:8000/predict_single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@buffalo.jpg"

# Multi-image prediction
curl -X POST "http://localhost:8000/predict_multi" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "aggregation=average"

# Health check
curl http://localhost:8000/health

# List breeds
curl http://localhost:8000/breeds
```

---

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture.

### Two-Stage Classification Pipeline

```
Input Image(s)
    ↓
Stage 1: Animal Type Detection
    ├─→ Buffalo → Buffalo Breed Classifier (17 classes)
    └─→ Cattle → Cattle Breed Classifier (if available)
    ↓
Stage 2: Breed Classification
    ↓
Decision Support Engine
    ├─→ ACCEPTED (>70% confidence)
    ├─→ REVIEW (50-70% confidence)
    └─→ REJECTED (<50% confidence)
    ↓
Final Prediction + Recommendations
```

### Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Binary Classifier**: 2 classes (cattle vs buffalo)
- **Breed Classifiers**: Separate models for buffalo and cattle breeds
- **Input Size**: 224x224 RGB
- **Augmentation**: 15+ production-grade transforms

---

## 📁 Project Structure

```
breed_recognition/
├── src/                    # Source code
│   ├── dataset.py          # Dataset pipeline
│   ├── model.py            # Model architecture
│   ├── train.py            # Training logic
│   ├── inference.py        # Inference + decision support
│   └── utils.py            # Utilities
├── api/                    # API module
│   └── app.py              # FastAPI application
├── frontend/               # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
├── dataset/                # Training data
│   └── buffalo/            # 17 breeds, 5,564 images
├── models/                 # Saved models
├── logs/                   # Training logs
├── plots/                  # Training plots
├── results/                # Evaluation results
├── config.yaml             # Configuration
├── main.py                 # Unified entry point
├── test.py                 # Testing script
├── README.md               # This file
├── INSTALLATION.md         # Setup guide
└── ARCHITECTURE.md         # System architecture
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# System mode
mode: "hybrid"  # Two-stage classification

# Training
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001

# Model
model:
  architecture: "mobilenet_v2"
  dropout: 0.3

# Dataset
dataset:
  root_dir: "dataset"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

---

## 📊 Performance

### Expected Metrics
- **Binary Classification**: >95% accuracy
- **Breed Classification**: 75-90% accuracy (depends on data quality)
- **Multi-Image Boost**: +5-10% accuracy improvement

### Inference Speed (GTX 1650)
- Single image: ~50-100ms
- Multi-image (5 images): ~200-400ms
- Batch (10 images): ~500-800ms

### Training Time
- **GTX 1650 (4GB)**: 20-35 minutes (50 epochs)
- **CPU**: 2-3 hours

---

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/breeds` | GET | List supported breeds |
| `/predict_single` | POST | Single image prediction |
| `/predict_multi` | POST | Multi-image prediction |
| `/predict_batch` | POST | Batch prediction |
| `/breed_info/{name}` | GET | Breed information |
| `/docs` | GET | Interactive API docs |
| `/frontend` | GET | Web interface |

### Response Format

```json
{
  "success": true,
  "final_prediction": "Murrah",
  "confidence": 0.87,
  "confidence_percent": "87.00%",
  "animal_type": "buffalo",
  "decision": "ACCEPTED",
  "decision_message": "High confidence prediction",
  "recommendation": "Proceed with identification",
  "top_predictions": [
    {
      "rank": 1,
      "breed": "Murrah",
      "confidence": 0.87,
      "confidence_percent": "87.00%"
    },
    {
      "rank": 2,
      "breed": "Mehsana",
      "confidence": 0.08,
      "confidence_percent": "8.00%"
    },
    {
      "rank": 3,
      "breed": "Surti",
      "confidence": 0.03,
      "confidence_percent": "3.00%"
    }
  ]
}
```

---

## 🎯 Features

### Two-Stage Classification
1. **Stage 1**: Determines if animal is cattle or buffalo
2. **Stage 2**: Predicts specific breed based on animal type

### Multi-Image Prediction
- **Average Aggregation**: Averages probabilities across images
- **Voting Aggregation**: Majority voting across predictions
- Improved accuracy with 2-10 images

### Decision Support System
- **ACCEPTED** (>70% confidence): High confidence, proceed
- **REVIEW** (50-70% confidence): Manual verification recommended
- **REJECTED** (<50% confidence): Low confidence, retake images

### Domain Intelligence
- Breed-specific feature descriptions
- Regional origin information
- Physical characteristics database
- Milk production data (for dairy breeds)

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### CUDA Out of Memory
```yaml
# In config.yaml
training:
  batch_size: 16  # or 8
```

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Dataset Not Found
```bash
# Verify structure
ls dataset/buffalo/
# Should show 17 breed folders
```

See [INSTALLATION.md](INSTALLATION.md) for more troubleshooting.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: Indian buffalo breed images
- Model: MobileNetV2 (pretrained on ImageNet)
- Framework: PyTorch, FastAPI
- Augmentation: Albumentations

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🔗 Links

- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Architecture Details**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Docs**: http://localhost:8000/docs (when server running)

---

**Made with ❤️ for livestock management and breed preservation**
