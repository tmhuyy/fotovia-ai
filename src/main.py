from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io

from .model_loader import model, img_transforms, device
from .config import LABELS

app = FastAPI(
    title="Fotovia AI Image Classification Service",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@app.get("/")
def healthcheck():
    return {"message": "Fotovia AI Service is running 🚀"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = img_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probs, 3)

        results = []
        for i in range(3):
            results.append(
                {
                    "tag": LABELS[top_indices[0][i].item()],
                    "confidence": round(top_probs[0][i].item(), 4),
                }
            )
            
        return {"predictions": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})