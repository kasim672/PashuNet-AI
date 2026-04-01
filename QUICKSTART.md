# 🚀 QUICKSTART GUIDE - Bharat Pashudhan App Breed Recognition

## ⚡ Quick Setup (5 Minutes)

### 1. Environment Setup

```bash
# Create virtual environment
py -3.13 -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup.py
```

### 2. Choose Your Mode

#### Option A: Buffalo Only (Legacy Mode)
```bash
# Dataset structure: buffalo/Murrah/, buffalo/Mehsana/, etc.
# config.yaml: root_dir: "buffalo"

python main.py
```

#### Option B: Hybrid Mode (Cattle + Buffalo) ⭐ RECOMMENDED
```bash
# Dataset structure: dataset/buffalo/, dataset/cattle/
# config.yaml: root_dir: "dataset"

python main_hybrid.py
```

### 3. Start API & Frontend

```bash
# Terminal 1: Start API
python api.py

# API runs at: http://localhost:8000
# Docs at: http://localhost:8000/docs
# Frontend at: http://localhost:8000/frontend
```

---

## 📁 Dataset Structure

### Legacy Mode (Buffalo Only)
```
buffalo/
├── Murrah/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Mehsana/
├── Jaffarabadi/
└── ... (17 breeds)
```

### Hybrid Mode (Recommended)
```
dataset/
├── buffalo/
│   ├── Murrah/
│   ├── Mehsana/
│   ├── Jaffarabadi/
│   └── ... (buffalo breeds)
└── cattle/
    ├── Gir/
    ├── Sahiwal/
    ├── Red Sindhi/
    └── ... (cattle breeds)
```

---

## 🎯 Common Tasks

### Train Model
```bash
# Buffalo only
python main.py

# Hybrid (cattle + buffalo)
python main_hybrid.py
```

### Test Inference
```bash
# Buffalo only
python test_inference.py

# Hybrid
python test_inference_hybrid.py
```

### Start API Server
```bash
python api.py
# Access: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Use Web Frontend
```bash
# API must be running first
# Open browser: http://localhost:8000/frontend
```

---

## 🔧 Configuration

Edit `config.yaml`:

```yaml
dataset:
  root_dir: "dataset"  # or "buffalo" for legacy mode
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  architecture: "mobilenet_v2"
  pretrained: true
  dropout: 0.3

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  use_weighted_loss: true
```

---

## 📊 API Usage Examples

### Single Image Prediction
```bash
curl -X POST "http://localhost:8000/predict_single" \
  -F "file=@buffalo_image.jpg"
```

### Multi-Image Prediction
```bash
curl -X POST "http://localhost:8000/predict_multi" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "aggregation=average"
```

### Get Breed Information
```bash
curl "http://localhost:8000/breed_info/Murrah"
```

---

## 🎨 Web Frontend Features

1. **Upload Images**: Drag & drop or click to upload
2. **Single/Multi Mode**: Choose prediction mode
3. **Real-time Results**: Instant breed identification
4. **Decision Support**: ACCEPTED/REVIEW/REJECTED recommendations
5. **Breed Information**: Detailed characteristics
6. **Download Results**: Export as JSON

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors
```bash
# Ensure virtual environment is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### API Not Starting
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Use different port
# Edit api.py: uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Model Not Found
```bash
# Train model first
python main.py  # or python main_hybrid.py

# Check models directory
dir models
```

---

## 📚 Documentation

- **README.md** - Complete project documentation
- **INSTALLATION.md** - Detailed installation guide
- **RESULTS.md** - System overview and features
- **DATASET_UPGRADE_SUMMARY.md** - Dataset changes

---

## 🎯 Next Steps

1. ✅ Setup environment
2. ✅ Prepare dataset
3. ✅ Train model
4. ✅ Test inference
5. ✅ Start API
6. ✅ Use frontend
7. 🚀 Deploy to production

---

## 💡 Tips

- **Use GPU**: Training is 10-20x faster with GPU
- **Multi-Image Mode**: Use 3-5 images for better accuracy
- **Class Imbalance**: Enable weighted loss in config.yaml
- **Fine-Tuning**: Enabled by default after epoch 20
- **Early Stopping**: Prevents overfitting (patience=10)

---

## 📞 Support

For issues or questions:
1. Check logs: `logs/training.log`
2. Review API docs: `http://localhost:8000/docs`
3. Verify config: `config.yaml`
4. Run setup check: `python setup.py`

---

**🌟 System is production-ready and fully operational! 🌟**
