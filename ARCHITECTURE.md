# 🏗️ System Architecture

Complete technical architecture documentation for the Buffalo Breed Recognition System.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Two-Stage Classification Pipeline](#two-stage-classification-pipeline)
3. [Model Architecture](#model-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [API Architecture](#api-architecture)
8. [Decision Support System](#decision-support-system)
9. [File Structure](#file-structure)
10. [Technology Stack](#technology-stack)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER INTERFACE                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   CLI Tool   │  │   REST API   │  │ Web Frontend │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
└─────────┼──────────────────┼──────────────────┼─────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────┼────────────────────────────┐
│                    INFERENCE ENGINE                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Two-Stage Classification System          │   │
│  │                                                   │   │
│  │  Stage 1: Animal Type Detection                  │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │  Binary Classifier (Cattle vs Buffalo)  │    │   │
│  │  └─────────────────┬───────────────────────┘    │   │
│  │                    │                             │   │
│  │         ┌──────────┴──────────┐                 │   │
│  │         │                     │                 │   │
│  │  Stage 2: Breed Classification                  │   │
│  │  ┌──────▼──────┐      ┌──────▼──────┐          │   │
│  │  │   Buffalo   │      │   Cattle    │          │   │
│  │  │   Breed     │      │   Breed     │          │   │
│  │  │ Classifier  │      │ Classifier  │          │   │
│  │  │ (17 breeds) │      │ (N breeds)  │          │   │
│  │  └─────────────┘      └─────────────┘          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Decision Support Engine                  │   │
│  │  • Confidence Analysis                           │   │
│  │  • Multi-Image Aggregation                       │   │
│  │  • Domain Intelligence                           │   │
│  │  • Recommendation Generation                     │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────┐
│                    MODEL STORAGE                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Binary     │  │   Buffalo    │  │   Cattle     │  │
│  │  Classifier  │  │   Breed      │  │   Breed      │  │
│  │    Model     │  │   Model      │  │   Model      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Two-Stage Classification Pipeline

### Stage 1: Animal Type Detection

**Purpose**: Determine if the animal is cattle or buffalo

**Model**: Binary Classifier
- **Input**: 224x224 RGB image
- **Output**: 2 classes (buffalo=0, cattle=1)
- **Confidence**: Probability for each class

**Process**:
1. Image preprocessing (resize, normalize)
2. Feature extraction (MobileNetV2 backbone)
3. Binary classification head
4. Softmax activation
5. Output: Animal type + confidence

### Stage 2: Breed Classification

**Purpose**: Identify specific breed based on animal type

**Models**: 
- Buffalo Breed Classifier (17 classes)
- Cattle Breed Classifier (N classes, if available)

**Process**:
1. Route to appropriate classifier based on Stage 1 output
2. Feature extraction (MobileNetV2 backbone)
3. Breed classification head
4. Softmax activation
5. Output: Top-K breed predictions + confidences

### Pipeline Flow

```python
def predict(image):
    # Stage 1: Animal Type
    animal_type, animal_confidence = binary_classifier(image)
    
    # Stage 2: Breed
    if animal_type == 'buffalo':
        breed_predictions = buffalo_classifier(image)
    elif animal_type == 'cattle':
        breed_predictions = cattle_classifier(image)
    
    # Decision Support
    decision = decision_engine.analyze(breed_predictions)
    
    return {
        'animal_type': animal_type,
        'breed_predictions': breed_predictions,
        'decision': decision
    }
```

---

## Model Architecture

### Backbone: MobileNetV2

**Why MobileNetV2?**
- Lightweight (~14MB per model)
- Fast inference (~50-100ms)
- Mobile-friendly
- Good accuracy/speed tradeoff
- Pre-trained on ImageNet

**Architecture**:
```
Input (224x224x3)
    ↓
MobileNetV2 Backbone (frozen initially)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Dense Layer (512 units)
    ↓
ReLU + BatchNorm
    ↓
Dropout (0.15)
    ↓
Dense Layer (256 units)
    ↓
ReLU + BatchNorm
    ↓
Dropout (0.15)
    ↓
Output Layer (num_classes)
    ↓
Softmax
```

### Binary Classifier

```python
class BinaryAnimalClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.15),
            nn.Linear(256, 2)  # 2 classes
        )
```

### Breed Classifier

```python
class BreedClassifier(nn.Module):
    def __init__(self, num_classes, config, animal_type):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.15),
            nn.Linear(256, num_classes)
        )
```

---

## Data Pipeline

### Dataset Structure

```
dataset/
└── buffalo/
    ├── banni/          (127 images)
    ├── bargur/         (123 images)
    ├── bhadwari/       (148 images)
    ├── chhattisgarhi/  (131 images)
    └── ... (13 more breeds)

Total: 17 breeds, 5,564 images
```

### Data Split

- **Training**: 70% (3,833 images)
- **Validation**: 15% (822 images)
- **Test**: 15% (822 images)

**Stratified Split**: Maintains class distribution across splits

### Data Augmentation

**Training Augmentations** (15+ transforms):

1. **Geometric**:
   - Horizontal flip (p=0.5)
   - Rotation (±15°, p=0.5)

2. **Lighting**:
   - Random brightness/contrast (±20%, p=0.5)
   - Random gamma (80-120, p=0.3)
   - Random tone curve (p=0.3)

3. **Blur & Motion**:
   - Motion blur (p=0.3)
   - Gaussian blur (p=0.2)
   - Regular blur (p=0.2)

4. **Noise**:
   - Gaussian noise (p=0.3)
   - ISO noise (p=0.2)

5. **Weather**:
   - Random fog (p=0.15)
   - Random rain (p=0.1)
   - Random shadow (p=0.2)
   - Random sun flare (p=0.1)

6. **Occlusion**:
   - Coarse dropout (p=0.25)

7. **Color**:
   - Color jitter (p=0.3)
   - HSV adjustments (p=0.3)

**Validation/Test**: Only resize + normalize

### Data Loading

```python
class HybridDatasetManager:
    def prepare_breed_dataset(self, animal_type):
        # Analyze dataset
        analysis = analyze_dataset(breed_dir)
        
        # Collect images and labels
        image_paths, labels = self._collect_data()
        
        # Stratified split
        X_train, X_val, X_test = train_test_split(
            image_paths, labels, stratify=labels
        )
        
        # Create datasets with augmentation
        train_dataset = BuffaloBreedDataset(
            X_train, y_train, train_transform
        )
        
        return train_dataset, val_dataset, test_dataset
```

---

## Training Pipeline

### Training Process

```
1. Data Preparation
   ├─→ Load dataset
   ├─→ Analyze class distribution
   ├─→ Create train/val/test splits
   └─→ Apply augmentation

2. Model Initialization
   ├─→ Load pre-trained backbone
   ├─→ Freeze backbone layers
   └─→ Initialize classification head

3. Phase 1: Transfer Learning
   ├─→ Train classification head only
   ├─→ Epochs: 20
   ├─→ Learning rate: 0.001
   └─→ Early stopping: patience=10

4. Phase 2: Fine-Tuning
   ├─→ Unfreeze last 50 layers
   ├─→ Epochs: 30
   ├─→ Learning rate: 0.0001
   └─→ Early stopping: patience=10

5. Evaluation
   ├─→ Test set evaluation
   ├─→ Confusion matrix
   ├─→ Classification report
   └─→ Save best model
```

### Training Configuration

```yaml
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 10
  gradient_clip: 1.0
  use_weighted_loss: true
  label_smoothing: 0.1

fine_tuning:
  enabled: true
  start_epoch: 20
  learning_rate: 0.0001
  unfreeze_layers: 50
```

### Loss Function

**Weighted Cross-Entropy Loss**:
```python
# Compute class weights
class_weights = compute_class_weights(train_labels, num_classes)

# Weighted loss
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)
```

### Optimization

- **Optimizer**: Adam
- **Learning Rate**: 0.001 (initial), 0.0001 (fine-tuning)
- **Scheduler**: Cosine Annealing
- **Gradient Clipping**: 1.0
- **Weight Decay**: 0.0001

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False
```

---

## Inference Pipeline

### Single Image Inference

```python
def predict_single(image_path):
    # 1. Preprocess
    image = preprocess_image(image_path)
    
    # 2. Stage 1: Animal Type
    animal_type, animal_conf = predict_animal_type(image)
    
    # 3. Stage 2: Breed
    breed_predictions = predict_breed(image, animal_type)
    
    # 4. Decision Support
    decision = decision_engine.make_decision(
        breed_predictions[0]['confidence'],
        breed_predictions,
        image_count=1
    )
    
    # 5. Domain Intelligence
    reasoning = get_breed_reasoning(breed_predictions[0]['breed'])
    
    return {
        'animal_type': animal_type,
        'breed_predictions': breed_predictions,
        'decision': decision,
        'reasoning': reasoning
    }
```

### Multi-Image Inference

**Average Aggregation**:
```python
def predict_multi_average(image_paths):
    all_probabilities = []
    
    for image_path in image_paths:
        image = preprocess_image(image_path)
        animal_type = predict_animal_type(image)
        probabilities = get_breed_probabilities(image, animal_type)
        all_probabilities.append(probabilities)
    
    # Average probabilities
    avg_probs = np.mean(all_probabilities, axis=0)
    
    # Get top-K predictions
    top_indices = np.argsort(avg_probs)[::-1][:top_k]
    
    return format_predictions(top_indices, avg_probs)
```

**Voting Aggregation**:
```python
def predict_multi_voting(image_paths):
    all_predictions = []
    
    for image_path in image_paths:
        prediction = predict_single(image_path)
        all_predictions.append(prediction['final_prediction'])
    
    # Count votes
    vote_counts = Counter(all_predictions)
    
    # Get winner
    final_prediction = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[final_prediction] / len(all_predictions)
    
    return {
        'final_prediction': final_prediction,
        'confidence': confidence,
        'votes': vote_counts
    }
```

---

## API Architecture

### FastAPI Application

```python
app = FastAPI(
    title="Buffalo Breed Recognition API",
    description="AI-powered breed identification",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Static files
app.mount("/static", StaticFiles(directory="frontend"))

# Global predictor
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    config = load_config('config.yaml')
    device = get_device()
    predictor = BreedPredictor('models', config, device)
```

### Endpoints

#### 1. Health Check
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "gpu_available": torch.cuda.is_available()
    }
```

#### 2. Single Image Prediction
```python
@app.post("/predict_single")
async def predict_single(file: UploadFile = File(...)):
    # Validate file
    validate_file(file)
    
    # Save temporary file
    tmp_path = save_temp_file(file)
    
    # Run inference
    result = predictor.predict_with_decision_support(tmp_path)
    
    # Clean up
    os.unlink(tmp_path)
    
    return result
```

#### 3. Multi-Image Prediction
```python
@app.post("/predict_multi")
async def predict_multi(
    files: List[UploadFile] = File(...),
    aggregation: str = Form("average")
):
    # Validate files
    validate_files(files)
    
    # Save temporary files
    tmp_paths = save_temp_files(files)
    
    # Run multi-image inference
    result = predictor.predict_multi(tmp_paths, aggregation=aggregation)
    
    # Clean up
    cleanup_temp_files(tmp_paths)
    
    return result
```

---

## Decision Support System

### Decision Engine

```python
class DecisionSupportEngine:
    def make_decision(self, confidence, predictions, image_count):
        # Confidence thresholds
        if confidence > 0.7:
            decision = "ACCEPTED"
            message = "High confidence prediction"
            recommendation = "Proceed with identification"
        elif confidence > 0.5:
            decision = "REVIEW"
            message = "Medium confidence prediction"
            recommendation = "Manual verification recommended"
        else:
            decision = "REJECTED"
            message = "Low confidence prediction"
            recommendation = "Retake images or consult expert"
        
        # Adjust for multi-image
        if image_count > 1:
            confidence_boost = min(0.1, image_count * 0.02)
            confidence += confidence_boost
        
        return {
            'decision': decision,
            'message': message,
            'recommendation': recommendation,
            'confidence_level': self._get_confidence_level(confidence)
        }
```

### Domain Intelligence

```python
BREED_FEATURES = {
    'Murrah': {
        'origin': 'Haryana, Punjab',
        'characteristics': 'Black coat, curled horns',
        'milk_production': 'High (10-15 liters/day)',
        'weight': '500-800 kg'
    },
    'Mehsana': {
        'origin': 'Gujarat',
        'characteristics': 'Black coat, medium-sized',
        'milk_production': 'Medium (6-8 liters/day)',
        'weight': '400-600 kg'
    },
    # ... more breeds
}

def get_breed_reasoning(breed_name):
    features = BREED_FEATURES.get(breed_name, {})
    return f"Identified as {breed_name} based on physical characteristics"
```

---

## File Structure

```
breed_recognition/
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── dataset.py              # Dataset pipeline
│   ├── model.py                # Model architecture
│   ├── train.py                # Training logic
│   ├── inference.py            # Inference + decision support
│   └── utils.py                # Utilities
│
├── api/                        # API module
│   ├── __init__.py
│   └── app.py                  # FastAPI application
│
├── frontend/                   # Web interface
│   ├── index.html              # Main page
│   ├── style.css               # Styling
│   └── script.js               # Frontend logic
│
├── dataset/                    # Training data
│   └── buffalo/                # 17 breeds
│       ├── banni/
│       ├── bargur/
│       └── ...
│
├── models/                     # Saved models
│   ├── buffalo_classifier.pth
│   ├── binary_classifier.pth
│   └── metadata.json
│
├── logs/                       # Training logs
├── plots/                      # Training plots
├── results/                    # Evaluation results
│
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── main.py                     # Unified entry point
├── test.py                     # Testing script
├── README.md                   # Documentation
├── INSTALLATION.md             # Setup guide
└── ARCHITECTURE.md             # This file
```

---

## Technology Stack

### Core Technologies

- **Python**: 3.8+
- **PyTorch**: 2.0+ (Deep learning framework)
- **FastAPI**: 0.100+ (API framework)
- **Uvicorn**: ASGI server

### ML/AI Libraries

- **torchvision**: Pre-trained models
- **albumentations**: Data augmentation
- **scikit-learn**: Data splitting, metrics
- **numpy**: Numerical operations
- **opencv-python**: Image processing

### API & Web

- **FastAPI**: REST API
- **Pydantic**: Data validation
- **python-multipart**: File uploads
- **Jinja2**: Template rendering

### Utilities

- **PyYAML**: Configuration
- **Pillow**: Image handling
- **tqdm**: Progress bars
- **logging**: Logging

### Development

- **pytest**: Testing
- **black**: Code formatting
- **flake8**: Linting

---

## Performance Optimization

### Model Optimization

1. **Quantization**: Reduce model size
2. **Pruning**: Remove unnecessary weights
3. **Knowledge Distillation**: Train smaller model
4. **ONNX Export**: Cross-platform deployment

### Inference Optimization

1. **Batch Processing**: Process multiple images together
2. **GPU Acceleration**: Use CUDA when available
3. **Model Caching**: Keep model in memory
4. **Image Preprocessing**: Optimize resize/normalize

### API Optimization

1. **Async Processing**: Non-blocking operations
2. **Connection Pooling**: Reuse connections
3. **Caching**: Cache frequent requests
4. **Load Balancing**: Distribute requests

---

## Deployment Considerations

### Production Checklist

- [ ] Model trained and validated
- [ ] API tested with load testing
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Monitoring setup
- [ ] Security measures in place
- [ ] Documentation complete
- [ ] Backup strategy defined

### Scaling Strategy

1. **Horizontal Scaling**: Multiple API instances
2. **Load Balancer**: Distribute traffic
3. **Model Serving**: Dedicated inference servers
4. **Caching Layer**: Redis for frequent requests
5. **CDN**: Static file delivery

---

**Last Updated**: 2026-04-02  
**Version**: 2.0.0  
**Status**: Production Ready
