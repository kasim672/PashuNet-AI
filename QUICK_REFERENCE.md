# 🚀 QUICK REFERENCE CARD

## One-Page Guide to the Hybrid System

---

## 📁 File Structure

### ✅ USE THESE (Hybrid System)
```
src/
├── model_hybrid.py          # Two-stage models
├── train_hybrid.py          # Training pipeline
├── dataset_hybrid.py        # Dataset management
└── inference_hybrid.py      # Inference system

main_hybrid.py               # Training script
test_inference_hybrid.py     # Testing script
api.py                       # API server
config.yaml                  # Configuration
```

### ⚠️ DEPRECATED (Legacy System)
```
src/
├── model.py                 # Single-stage model
├── train.py                 # Legacy training
├── dataset.py               # Legacy dataset
└── inference.py             # Partially deprecated

main.py                      # Legacy training
```

---

## ⚡ Quick Commands

### Training
```bash
python main_hybrid.py        # Train hybrid system
```

### Testing
```bash
python test_inference_hybrid.py
```

### API
```bash
python api.py                # Start server
# Access: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### API Calls
```bash
# Single image
curl -X POST "http://localhost:8000/predict_single" \
  -F "file=@image.jpg"

# Multi-image
curl -X POST "http://localhost:8000/predict_multi" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "aggregation=average"
```

---

## 📊 System Flow

```
Image(s) → Binary Classifier → Breed Classifier → Decision Support → Output
         (Cattle/Buffalo)     (Specific Breed)   (Accept/Review/Reject)
```

---

## 🎯 Decision Rules

| Confidence | Decision | Action |
|-----------|----------|--------|
| ≥ 70% | ACCEPTED | Proceed with registration |
| 50-70% | REVIEW | Manual verification needed |
| < 50% | REJECTED | Retake photos |

---

## 📂 Dataset Structure

```
dataset/
├── buffalo/
│   ├── Murrah/
│   ├── Mehsana/
│   └── ...
└── cattle/
    ├── Gir/
    ├── Sahiwal/
    └── ...
```

---

## ⚙️ Configuration

**config.yaml:**
```yaml
mode: "hybrid"
dataset:
  root_dir: "dataset"
  hybrid_mode: true
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python main_hybrid.py` |
| Import error | Check `src/inference_hybrid.py` exists |
| API error | Verify `models/hybrid/metadata.json` |
| GPU not detected | Reinstall PyTorch with CUDA |

---

## 📚 Documentation

- **QUICKSTART.md** - Quick start guide
- **MIGRATION_GUIDE.md** - Legacy to hybrid migration
- **SYSTEM_INTEGRATION_COMPLETE.md** - Integration details
- **INTEGRATION_FIXES_SUMMARY.md** - What was fixed
- **RESULTS.md** - System capabilities

---

## 🎯 Output Format

```json
{
  "animal_type": "buffalo",
  "final_prediction": "Murrah",
  "confidence": 0.92,
  "decision": "ACCEPTED",
  "recommendation": "Proceed with registration",
  "top_predictions": [...],
  "breed_info": {...}
}
```

---

## ✅ Quick Checklist

**Before Training:**
- [ ] Dataset in `dataset/buffalo/` and `dataset/cattle/`
- [ ] `config.yaml` set to `mode: "hybrid"`
- [ ] Virtual environment activated
- [ ] Dependencies installed

**After Training:**
- [ ] Models in `models/hybrid/`
- [ ] `metadata.json` exists
- [ ] Test with `test_inference_hybrid.py`
- [ ] Start API with `python api.py`

---

**Need Help?** Check the full documentation or logs in `logs/training.log`
