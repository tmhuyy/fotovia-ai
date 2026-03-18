from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
from model_loader import model, processor

app = FastAPI(
    title="Fotovia AI Image Classification Service",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Load model once when server starts
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define photography categories
LABELS = [
    "film photography",
    "hong kong vibe",
    "wedding photography",
    "food photography",
    "interior photography",
    "portrait photography",
    "street photography",
    "cinematic photography",
]


@app.get("/")
def healthcheck():
    return {"message": "Fotovia AI Service is running 🚀"}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(text=LABELS, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # confidence, predicted_index = torch.max(probs, dim=1)

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

        # result = {
        #     "predicted_tag": LABELS[predicted_index.item()],
        #     "confidence_score": round(confidence.item(), 4),
        # }

        # return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
