# Fotovia AI Image Classification Service
## Overview
The Fotovia AI Image Classification Service is a standalone AI microservice responsible for automatically analyzing and categorizing photographer portfolio images.
## Tech Stack
- Python 3.10+
- FastAPI
- PyTorch
- HuggingFace Transformers
- OpenAI CLIP Model
- Pillow
## How to run
step 1: source venv/bin/activate
step 2: uvicorn main:app --reload
URL http://127.0.0.1:8000/swagger
step 3: deactivate