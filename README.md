# Fotovia AI Image Classification Service

## Overview

The Fotovia AI Image Classification Service is a standalone FastAPI service used to classify photography images into the current Fotovia style classes.

At the current stage of the project, this service is designed to support:

- single-image file upload classification for legacy/demo flows
- single-image URL classification for backend-to-backend integration
- batch image URL classification for portfolio item processing

## Current runtime stack

- Python 3.10+
- FastAPI
- PyTorch
- torchvision
- Pillow
- httpx

## Current model setup

The current runtime loads one local trained model when the service starts.

- model name: `resnext`
- number of classes: `10`
- image size: `224`
- weights path: `resnext_best.pth`

Current labels:

- aerial
- architecture
- event
- fashion
- food
- nature
- sports
- street
- wedding
- wildlife

## How to run
- step 1: run <code>python3 -m venv venv</code> to create virtical env
- step 2: run <code>pip install -r requirements.txt</code> to install dependencies
- step 3: run <code>source venv/bin/activate</code> to activate
- step 4: run <code>uvicorn src.main:app</code> or <code>uvicorn src.main:app --reload</code>
-- URL http://127.0.0.1:8000/swagger
- step 3: deactivate

## API endpoints

### 1. Healthcheck

`GET /`

Example response:

```json
{
  "statusCode": 200,
  "data": {
    "message": "Fotovia AI Service is running",
    "modelName": "resnext",
    "numClasses": 10,
    "labels": [
      "aerial",
      "architecture",
      "event",
      "fashion",
      "food",
      "nature",
      "sports",
      "street",
      "wedding",
      "wildlife"
    ]
  }
}

