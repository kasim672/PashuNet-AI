# 🔄 MIGRATION GUIDE - Legacy to Hybrid System

## Overview

This guide helps you migrate from the legacy single-stage system to the new hybrid two-stage classification system.

---

## 🆚 System Comparison

| Feature | Legacy System | Hybrid System |
|---------|--------------|---------------|
| **Classification** | Single-stage (breed only) | Two-stage (animal type → breed) |
| **Animal Types** | Buffalo only | Buffalo + Cattle |
| **Models** | 1 model | 3 models (binary + 2 breed) |
| **Accuracy** | Good | Better (specialized models) |
| **Scalability** | Limited | High (easy to add animals) |
| **Files** | `src/model.py`, `src/inference.py` | `src/model_hybrid.py`, `src/inference_hybrid.py` |

---

## 📋 Migration Steps

### Step 1: Update Dataset Structure

**Legacy Structure:**
```
buffalo/
├── Murrah/
├── Mehsana/
└── ...
```

**Hybrid Structure:**
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

**Action:**
```bash
# Create new structure
mkdir -p dataset/buffalo
mkdir -p dataset/cattle

# Move buffalo data
mv buffalo/* dataset/buffalo/

# Add cattle data (if available)
# Copy cattle breed folders to dataset/cattle/
```

### Step 2: Update Configuration

**Edit `config.yaml`:**
```yaml
# Change this:
mode: "legacy"
dataset:
  root_dir: "buffalo"

# To this:
mode: "hybrid"
dataset:
  root_dir: "dataset"
  hybrid_mode: true
  binary_classifier: true
```

### Step 3: Train Hybrid Models

```bash
# Old command (legacy):
python main.py

# New command (hybrid):
python main_hybrid.py
```

**Output:**
- `models/hybrid/binary_classifier.pth`
- `models/hybrid/buffalo_classifier.pth`
- `models/hybrid/cattle_classifier.pth` (if cattle data available)
- `models/hybrid/metadata.json`

### Step 4: Update API Code (if custom)

**Legacy Code:**
```python
from src.inference import HybridBreedPredictor

predictor = HybridBreedPredictor(
    model_path='models/best_model.pth',
    class_names=class_names,
    config=config,
    device=device,
    animal_type='buffalo'
)
```

**Hybrid Code:**
```python
from src.inference_hybrid import HybridBreedPredictor

predictor = HybridBreedPredictor(
    model_dir='models/hybrid',
    config=config,
    device=device
)
```

**Note:** The provided `api.py` already handles this automatically!

### Step 5: Test the System

```bash
# Test hybrid inference
python test_inference_hybrid.py

# Start API
python api.py

# Test API endpoint
curl -X POST "http://localhost:8000/predict_single" \
  -F "file=@test_image.jpg"
```

---

## 🔧 Code Changes Required

### If Using Custom Inference Code

**Legacy:**
```python
from src.inference import HybridBreedPredictor

predictor = HybridBreedPredictor(model_path, class_names, config, device)
result = predictor.predict_with_decision_support(image_path)
```

**Hybrid:**
```python
from src.inference_hybrid import HybridBreedPredictor

predictor = HybridBreedPredictor(model_dir, config, device)
result = predictor.predict_with_decision_support(image_path)
```

**Output format is identical** - no changes needed to result processing!

---

## 📊 Expected Results

### Legacy System Output:
```json
{
  "final_prediction": "Murrah",
  "confidence": 0.92,
  "animal_type": "buffalo",
  "decision": "ACCEPTED",
  ...
}
```

### Hybrid System Output:
```json
{
  "final_prediction": "Murrah",
  "confidence": 0.92,
  "animal_type": "buffalo",
  "animal_confidence": 0.98,  // NEW: Animal type confidence
  "decision": "ACCEPTED",
  ...
}
```

**Additional field:** `animal_confidence` - confidence in animal type classification

---

## ⚠️ Important Notes

### 1. Backward Compatibility
- Legacy files are **NOT deleted**
- Legacy system still works if needed
- API automatically detects which model to use

### 2. Model Files
- Legacy: `models/best_model.pth`, `models/class_names.json`
- Hybrid: `models/hybrid/*.pth`, `models/hybrid/metadata.json`
- Both can coexist

### 3. Performance
- Hybrid system may be slightly slower (two stages)
- But accuracy is typically better
- Especially when mixing cattle and buffalo

### 4. Buffalo-Only Mode
- If you only have buffalo data, hybrid system still works
- Binary classifier won't be trained
- System automatically falls back to buffalo-only mode

---

## 🐛 Troubleshooting

### Issue: "Metadata not found"
**Solution:**
```bash
# Train hybrid system first
python main_hybrid.py
```

### Issue: "Cattle classifier not available"
**Solution:**
- This is normal if you only have buffalo data
- System will work in buffalo-only mode
- To add cattle: add cattle breed folders to `dataset/cattle/`

### Issue: API still using legacy model
**Solution:**
```bash
# Check model directory
ls models/hybrid/

# Should see:
# - metadata.json
# - binary_classifier.pth (optional)
# - buffalo_classifier.pth
# - cattle_classifier.pth (optional)

# If missing, train hybrid system:
python main_hybrid.py
```

### Issue: Import errors
**Solution:**
```bash
# Ensure all files are present
ls src/inference_hybrid.py
ls src/model_hybrid.py
ls src/dataset_hybrid.py
ls src/train_hybrid.py

# If missing, check git status or re-download
```

---

## 📚 Additional Resources

- **QUICKSTART.md** - Quick start guide
- **SYSTEM_INTEGRATION_COMPLETE.md** - Integration details
- **RESULTS.md** - System features and capabilities
- **INSTALLATION.md** - Installation instructions

---

## ✅ Migration Checklist

- [ ] Dataset restructured to `dataset/buffalo/` and `dataset/cattle/`
- [ ] `config.yaml` updated with `mode: "hybrid"`
- [ ] Hybrid models trained with `python main_hybrid.py`
- [ ] Models saved to `models/hybrid/`
- [ ] Testing completed with `python test_inference_hybrid.py`
- [ ] API tested with sample images
- [ ] Custom code updated (if any)
- [ ] Documentation reviewed

---

**Migration Time:** ~30 minutes (excluding training time)  
**Training Time:** 2-4 hours (depending on dataset size and GPU)

**Status:** Ready to migrate! 🚀
