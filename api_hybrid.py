"""
Production-Grade FastAPI for Hybrid Breed Recognition
Endpoints: Single/Multi-image prediction, Grad-CAM explanation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
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
