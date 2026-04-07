import io
from typing import Any

import httpx
import torch
from PIL import Image

from .config import LABELS
from .model_loader import device, img_transforms, model


def _normalize_top_k(top_k: int) -> int:
    return max(1, min(top_k, len(LABELS)))


def predict_from_pil_image(image: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
    applied_top_k = _normalize_top_k(top_k)

    rgb_image = image.convert("RGB")
    input_tensor = img_transforms(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, applied_top_k)

    predictions: list[dict[str, Any]] = []
    for index in range(applied_top_k):
        predictions.append(
            {
                "label": LABELS[top_indices[0][index].item()],
                "confidence": round(top_probs[0][index].item(), 4),
            }
        )

    return predictions


def classify_image_bytes(image_bytes: bytes, top_k: int = 3) -> list[dict[str, Any]]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return predict_from_pil_image(image=image, top_k=top_k)


async def fetch_image_bytes_from_url(client: httpx.AsyncClient, url: str) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    return response.content


async def classify_image_from_url(
    client: httpx.AsyncClient,
    url: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    image_bytes = await fetch_image_bytes_from_url(client=client, url=url)
    return classify_image_bytes(image_bytes=image_bytes, top_k=top_k)
