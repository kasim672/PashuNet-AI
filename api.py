"""
Enhanced FastAPI deployment for Hybrid Cattle+Buffalo Breed Recognition
Features: Multi-image prediction, Decision support, Grad-CAM explainability
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import torch
import json
import os
import tempfile
from pathlib import Path
import logging

from src.utils import load_config, get_device
from src.inference_hybrid import HybridBreedPredictor
from src.inference import format_prediction_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bharat Pashudhan App - Breed Recognition API",
    description="AI-powered cattle and buffalo breed identification with decision support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global variables
predictor = None
config = None


class SinglePredictionResponse(BaseModel):
    """Response model for single image prediction"""
    success: bool
    final_prediction: str
    confidence: float
    confidence_percent: str
    confidence_level: str
    animal_type: str
    decision: str
    decision_message: str
    recommendation: str
    reasoning: str
    top_predictions: List[dict]
    breed_info: Optional[dict] = None


class MultiPredictionResponse(BaseModel):
    """Response model for multi-image prediction"""
    success: bool
    final_prediction: str
    confidence: float
    confidence_percent: str
    confidence_level: str
    animal_type: str
    decision: str
    decision_message: str
    recommendation: str
    reasoning: str
    top_predictions: List[dict]
    breed_info: Optional[dict] = None
    aggregation_method: str
    images_processed: int
    images_successful: int


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global predictor, config

    try:
        # Load config
        config = load_config('config.yaml')
        device = get_device()

        # Check for hybrid models
        hybrid_model_dir = os.path.join(
            config['output']['model_dir'], 'hybrid')
        legacy_model_dir = config['output']['model_dir']

        # Try hybrid mode first
        if os.path.exists(os.path.join(hybrid_model_dir, 'metadata.json')):
            logger.info("Loading hybrid two-stage classification system...")
            predictor = HybridBreedPredictor(hybrid_model_dir, config, device)
            logger.info("✓ Hybrid model loaded successfully")
            logger.info(f"  Device: {device}")
            logger.info(f"  Mode: Two-stage (Animal Type → Breed)")

        # Fallback to legacy single-stage model
        elif os.path.exists(os.path.join(legacy_model_dir, 'class_names.json')):
            logger.info("Loading legacy single-stage model...")
            from src.inference import HybridBreedPredictor as LegacyPredictor

            with open(os.path.join(legacy_model_dir, 'class_names.json'), 'r') as f:
                class_names = json.load(f)

            model_path = os.path.join(legacy_model_dir, 'best_model.pth')
            predictor = LegacyPredictor(
                model_path, class_names, config, device, animal_type='buffalo')
            logger.info("✓ Legacy model loaded successfully")
            logger.info(f"  Device: {device}")
            logger.info(f"  Classes: {len(class_names)}")

        else:
            logger.error("No trained models found")
            logger.warning(
                "⚠️ Model not loaded. Please train the model first:")
            logger.warning("   - Hybrid mode: python main_hybrid.py")
            logger.warning("   - Legacy mode: python main.py")
            return

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning(
            "⚠️ API will run in limited mode without prediction capability")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Bharat Pashudhan App - Breed Recognition API",
        "version": "2.0.0",
        "status": "active",
        "model_loaded": predictor is not None,
        "features": [
            "Single image prediction",
            "Multi-image prediction with aggregation",
            "Decision support engine",
            "Domain intelligence",
            "Breed information database"
        ],
        "endpoints": {
            "/predict_single": "POST - Single image breed prediction",
            "/predict_multi": "POST - Multi-image aggregated prediction",
            "/predict_batch": "POST - Batch prediction (independent images)",
            "/health": "GET - Health check",
            "/breeds": "GET - List all supported breeds",
            "/breed_info/{breed_name}": "GET - Get breed information",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "gpu_available": torch.cuda.is_available(),
        "device": str(predictor.device) if predictor else "N/A"
    }


@app.get("/breeds")
async def get_breeds():
    """Get list of supported breeds"""
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first.")

    return {
        "total_breeds": len(predictor.class_names),
        "animal_type": predictor.animal_type,
        "breeds": sorted(predictor.class_names)
    }


@app.get("/breed_info/{breed_name}")
async def get_breed_info(breed_name: str):
    """Get detailed information about a specific breed"""
    from src.inference import BREED_FEATURES

    breed_name = breed_name.replace("_", " ").title()

    if breed_name not in BREED_FEATURES:
        raise HTTPException(
            status_code=404,
            detail=f"Breed '{breed_name}' not found in database"
        )

    return {
        "breed": breed_name,
        "info": BREED_FEATURES[breed_name]
    }


@app.post("/predict_single", response_model=SinglePredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    """
    Predict breed from a single image with full decision support

    Args:
        file: Image file (JPG, JPEG, PNG)

    Returns:
        Complete prediction with decision support and breed information
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first.")

    # Validate file type
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    file_ext = Path(file.filename).suffix
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
        )

    # Validate file size
    max_size = config['deployment']['max_image_size']
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents)/1e6:.1f}MB). Max size: {max_size / 1e6:.1f} MB"
        )

    # Save temporary file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        # Run inference with decision support
        result = predictor.predict_with_decision_support(
            tmp_path,
            top_k=config['evaluation']['top_k']
        )

        # Clean up
        os.unlink(tmp_path)

        return SinglePredictionResponse(
            success=True,
            **result
        )

    except Exception as e:
        # Clean up on error
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_multi", response_model=MultiPredictionResponse)
async def predict_multi(
    files: List[UploadFile] = File(...),
    aggregation: str = Form("average")
):
    """
    Predict breed from multiple images with aggregation

    Args:
        files: List of image files (2-10 images recommended)
        aggregation: Aggregation method ('average' or 'voting')

    Returns:
        Aggregated prediction with decision support
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first.")

    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Multi-image prediction requires at least 2 images"
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per request"
        )

    if aggregation not in ['average', 'voting']:
        raise HTTPException(
            status_code=400,
            detail="Aggregation method must be 'average' or 'voting'"
        )

    # Save temporary files
    tmp_paths = []
    try:
        for file in files:
            # Validate file type
            file_ext = Path(file.filename).suffix
            if file_ext not in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for '{file.filename}'"
                )

            contents = await file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(contents)
                tmp_paths.append(tmp_file.name)

        # Run multi-image inference
        result = predictor.predict_multi(
            tmp_paths,
            top_k=config['evaluation']['top_k'],
            aggregation=aggregation
        )

        # Clean up
        for tmp_path in tmp_paths:
            os.unlink(tmp_path)

        return MultiPredictionResponse(
            success=True,
            **result
        )

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        logger.error(f"Multi-image prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict breeds for multiple images independently (not aggregated)

    Args:
        files: List of image files (max 10)

    Returns:
        List of independent predictions for each image
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first.")

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )

    results = []

    for file in files:
        tmp_path = None
        try:
            # Validate file type
            file_ext = Path(file.filename).suffix
            if file_ext not in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid file type"
                })
                continue

            contents = await file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name

            # Run inference
            result = predictor.predict_with_decision_support(
                tmp_path,
                top_k=config['evaluation']['top_k']
            )

            os.unlink(tmp_path)

            results.append({
                "filename": file.filename,
                "success": True,
                **result
            })

        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {"results": results, "total_processed": len(results)}


@app.get("/frontend")
async def serve_frontend():
    """Serve the web frontend"""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return {
            "message": "Frontend not found",
            "note": "Please ensure frontend/index.html exists"
        }


# Legacy endpoint for backward compatibility
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    """Legacy prediction endpoint (redirects to predict_single)"""
    return await predict_single(file)


if __name__ == "__main__":
    import uvicorn

    config = load_config('config.yaml')

    logger.info("="*60)
    logger.info("Starting Bharat Pashudhan App - Breed Recognition API")
    logger.info("="*60)
    logger.info(f"Host: {config['deployment']['api_host']}")
    logger.info(f"Port: {config['deployment']['api_port']}")
    logger.info(
        f"Docs: http://localhost:{config['deployment']['api_port']}/docs")
    logger.info("="*60)

    uvicorn.run(
        app,
        host=config['deployment']['api_host'],
        port=config['deployment']['api_port'],
        log_level="info"
    )
