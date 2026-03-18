# model_loader.py

from transformers import CLIPProcessor, CLIPModel

print("Loading AI model once...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Model loaded 🚀")