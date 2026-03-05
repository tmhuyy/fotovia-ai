from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI(
    title="Fotovia AI Image Classification Service",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    
)

# Load model once when server starts
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define photography categories
LABELS = [
    "film photography",
    "hong kong vibe",
    "wedding photography",
    "food photography",
    "interior photography",
    "portrait photography",
    "street photography",
    "cinematic photography"
]

@app.get("/")
def healthcheck():
    return {"message": "Fotovia AI Service is running 🚀"}


