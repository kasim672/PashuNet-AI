# Buffalo Breed Recognition System

Production-grade AI system for identifying Indian buffalo breeds, designed for integration with the Bharat Pashudhan App (BPA).

## 🎯 Overview

This system uses deep learning (PyTorch + Transfer Learning) to classify 17 Indian buffalo breeds from images, providing:
- Top-3 predictions with confidence scores
- Robust handling of real-world conditions (varying lighting, backgrounds, poses)
- Production-ready REST API
- Comprehensive evaluation metrics

## 📊 Supported Breeds

The system recognizes 17 buffalo breeds:
- Banni, Bargur, Bhadwari, Chhattisgarhi, Chilika
- Gojri, Jaffarabadi, Kalahandi, Luit
- Marathwada, Mehsana, Murrah, Nagpuri
- Nili-Ravi, Pandharpuri, Surti, Toda

## 🏗️ Architecture

- **Model**: MobileNetV2 (lightweight, mobile-friendly)
- **Framework**: PyTorch
- **Input**: 224x224 RGB images
- **Output**: Top-3 breed predictions with confidence scores
- **Deployment**: FastAPI REST API

## 📁 Project Structure

```
breed_recognition_for_cattle_and_buffaloes/
├── buffalo/                    # Dataset (17 breed folders)
├── src/
│   ├── __init__.py
│   ├── utils.py               # Utility functions
│   ├── dataset.py             # Dataset handling & augmentation
│   ├── model.py               # Model architecture
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation & metrics
│   └── inference.py           # Inference system
├── main.py                    # Main training script
├── api.py                     # FastAPI deployment
├── test_inference.py          # Test inference
├── config.yaml                # Configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Setup & Installation

### Step 1: Environment Setup

```bash
# Create virtual environment (Python 3.10 recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify GPU (Optional but Recommended)

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 🎓 Training

### Train the Model

```bash
python main.py
```

This will:
1. Analyze dataset and check for class imbalance
2. Split data (70% train, 15% val, 15% test)
3. Apply data augmentation
4. Train with transfer learning
5. Fine-tune the model
6. Evaluate on test set
7. Generate confusion matrix and metrics
8. Save best model to `models/best_model.pth`

### Training Features

- **Transfer Learning**: Pre-trained MobileNetV2 backbone
- **Data Augmentation**: Rotation, flip, brightness, blur, noise
- **Class Imbalance Handling**: Weighted loss function
- **Early Stopping**: Prevents overfitting
- **Fine-Tuning**: Unfreezes backbone layers after initial training
- **Learning Rate Scheduling**: Cosine annealing

### Expected Training Time

- **With GPU (RTX 4060)**: ~30-45 minutes
- **With CPU**: ~3-4 hours

## 📊 Evaluation

After training, check results in:
- `results/confusion_matrix.png` - Visual confusion matrix
- `results/classification_report.txt` - Per-class metrics
- `results/test_metrics.json` - Overall metrics
- `plots/training_history.png` - Training curves

## 🔮 Inference

### Test on Sample Images

```bash
python test_inference.py
```

### Programmatic Inference

```python
from src.utils import load_config, get_device
from src.inference import BuffaloBreedPredictor
import json

# Load config and model
config = load_config('config.yaml')
device = get_device()

with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

predictor = BuffaloBreedPredictor(
    'models/best_model.pth',
    class_names,
    config,
    device
)

# Predict
result = predictor.predict_with_threshold(
    'path/to/image.jpg',
    confidence_threshold=0.5,
    top_k=3
)

print(f"Top Prediction: {result['top_prediction']}")
print(f"Confidence: {result['top_confidence']:.2%}")
print(f"All Predictions: {result['predictions']}")
```

## 🌐 API Deployment

### Start API Server

```bash
python api.py
```

Server runs at: `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. List Breeds
```bash
curl http://localhost:8000/breeds
```

#### 3. Predict Breed
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/buffalo_image.jpg"
```

Response:
```json
{
  "success": true,
  "predictions": [
    {
      "rank": 1,
      "breed": "murrah",
      "confidence": 0.92,
      "confidence_percent": "92.00%"
    },
    {
      "rank": 2,
      "breed": "mehsana",
      "confidence": 0.05,
      "confidence_percent": "5.00%"
    },
    {
      "rank": 3,
      "breed": "surti",
      "confidence": 0.02,
      "confidence_percent": "2.00%"
    }
  ],
  "top_prediction": "murrah",
  "top_confidence": 0.92,
  "high_confidence": true
}
```

#### 4. Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### API Documentation

Interactive API docs: `http://localhost:8000/docs`

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Dataset**: Train/val/test split ratios
- **Image**: Input size, normalization
- **Augmentation**: Rotation, brightness, blur parameters
- **Model**: Architecture (mobilenet_v2, efficientnet_b0, efficientnet_b2)
- **Training**: Batch size, learning rate, epochs
- **Deployment**: API host, port, file size limits

## 🎯 Production Considerations

### 1. Data Quality
- **Issue**: Noisy images, mislabeled data
- **Solution**: Implement data validation, manual review of low-confidence predictions

### 2. Class Imbalance
- **Issue**: Some breeds have fewer images
- **Solution**: Weighted loss function, data augmentation, collect more data

### 3. Overfitting
- **Issue**: Model memorizes training data
- **Solution**: Dropout, early stopping, data augmentation, regularization

### 4. Real-World Deployment
- **Lighting**: Model trained on diverse lighting conditions
- **Background**: Augmentation handles various backgrounds
- **Pose**: Multiple angles in training data
- **Image Quality**: Handles blur and noise

### 5. Model Optimization
- **Size**: MobileNetV2 is lightweight (~14MB)
- **Speed**: Fast inference (~50ms on GPU, ~200ms on CPU)
- **Mobile**: Can be converted to TFLite or ONNX for mobile deployment

## 📈 Performance Metrics

Expected performance (after training):
- **Accuracy**: 85-95% (depends on data quality)
- **Precision**: 85-93%
- **Recall**: 83-92%
- **F1-Score**: 84-92%

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
- Reduce `batch_size` in `config.yaml`
- Use smaller model (mobilenet_v2 instead of efficientnet_b2)

### Low Accuracy
- Check data quality and labels
- Increase training epochs
- Adjust learning rate
- Collect more training data

## 📝 Future Enhancements

1. **Mobile App Integration**: Convert to TFLite/ONNX
2. **Grad-CAM Visualization**: Show which parts of image influenced prediction
3. **Multi-Animal Detection**: Detect and classify multiple animals in one image
4. **Breed Characteristics**: Provide breed information with predictions
5. **Continuous Learning**: Update model with new data from field workers

## 📄 License

This project is developed for the Bharat Pashudhan App initiative.

## 👥 Support

For issues or questions:
1. Check logs in `logs/training.log`
2. Review configuration in `config.yaml`
3. Verify dataset structure matches expected format

## 🙏 Acknowledgments

- Dataset: Indian buffalo breed images
- Framework: PyTorch, FastAPI
- Pre-trained models: ImageNet weights
