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
- step 1: run <code>python3 -m venv venv</code> to create virtical env
- step 2: run <code>pip install -r requirements.txt</code> to install dependencies
- step 3: run <code>source venv/bin/activate</code> to activate
- step 4: run <code>uvicorn src.main:app</code> or <code>uvicorn src.main:app --reload</code>
-- URL http://127.0.0.1:8000/swagger
- step 3: deactivate