from typing import List

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import LABELS, MODEL_NAME, NUM_CLASSES
from .inference import classify_image_bytes, classify_image_from_url
from .schemas import (
    ClassifiedImageResult,
    ClassifyBatchUrlsData,
    ClassifyBatchUrlsRequest,
    ClassifyBatchUrlsResponse,
    ClassifyUrlData,
    ClassifyUrlRequest,
    ClassifyUrlResponse,
    HealthcheckData,
    HealthcheckResponse,
    LegacyClassifyData,
    LegacyClassifyResponse,
    LegacyPrediction,
    Prediction,
    RemoteImageInput,
)

app = FastAPI(
    title="Fotovia AI Image Classification Service",
    version="1.1.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


def _applied_top_k(top_k: int) -> int:
    return max(1, min(top_k, len(LABELS)))


def _to_prediction_models(raw_predictions: list[dict]) -> List[Prediction]:
    return [
        Prediction(
            label=str(item["label"]),
            confidence=float(item["confidence"]),
        )
        for item in raw_predictions
    ]


def _build_completed_result(
    image: RemoteImageInput,
    raw_predictions: list[dict],
) -> ClassifiedImageResult:
    return ClassifiedImageResult(
        imageKey=image.imageKey,
        role=image.role,
        status="completed",
        predictions=_to_prediction_models(raw_predictions),
        error=None,
    )


def _build_failed_result(
    image: RemoteImageInput, error: Exception
) -> ClassifiedImageResult:
    return ClassifiedImageResult(
        imageKey=image.imageKey,
        role=image.role,
        status="failed",
        predictions=[],
        error=str(error),
    )


@app.get("/", response_model=HealthcheckResponse)
def healthcheck():
    return HealthcheckResponse(
        statusCode=200,
        data=HealthcheckData(
            message="Fotovia AI Service is running",
            modelName=MODEL_NAME,
            numClasses=NUM_CLASSES,
            labels=LABELS,
        ),
    )


@app.post("/classify", response_model=LegacyClassifyResponse)
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        raw_predictions = classify_image_bytes(image_bytes=image_bytes, top_k=3)

        legacy_predictions = [
            LegacyPrediction(
                tag=str(item["label"]),
                confidence=float(item["confidence"]),
            )
            for item in raw_predictions
        ]

        return LegacyClassifyResponse(
            statusCode=200,
            data=LegacyClassifyData(predictions=legacy_predictions),
        )
    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={
                "statusCode": 500,
                "data": None,
                "error": str(error),
            },
        )


@app.post("/classify/url", response_model=ClassifyUrlResponse)
async def classify_image_by_url(payload: ClassifyUrlRequest):
    applied_top_k = _applied_top_k(payload.topK)

    try:
        timeout = httpx.Timeout(payload.timeoutSeconds)
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            raw_predictions = await classify_image_from_url(
                client=client,
                url=payload.image.url,
                top_k=applied_top_k,
            )

        result = _build_completed_result(
            image=payload.image,
            raw_predictions=raw_predictions,
        )
    except Exception as error:
        result = _build_failed_result(payload.image, error)

    return ClassifyUrlResponse(
        statusCode=200,
        data=ClassifyUrlData(
            modelName=MODEL_NAME,
            numClasses=NUM_CLASSES,
            topKApplied=applied_top_k,
            result=result,
        ),
    )


@app.post("/classify/batch-urls", response_model=ClassifyBatchUrlsResponse)
async def classify_batch_images_by_url(payload: ClassifyBatchUrlsRequest):
    if not payload.images:
        raise HTTPException(status_code=400, detail="images must not be empty")

    applied_top_k = _applied_top_k(payload.topK)
    results: List[ClassifiedImageResult] = []

    timeout = httpx.Timeout(payload.timeoutSeconds)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
    ) as client:
        for image in payload.images:
            try:
                raw_predictions = await classify_image_from_url(
                    client=client,
                    url=image.url,
                    top_k=applied_top_k,
                )
                results.append(
                    _build_completed_result(
                        image=image,
                        raw_predictions=raw_predictions,
                    )
                )
            except Exception as error:
                results.append(_build_failed_result(image, error))

    completed_count = sum(1 for result in results if result.status == "completed")
    failed_count = sum(1 for result in results if result.status == "failed")

    return ClassifyBatchUrlsResponse(
        statusCode=200,
        data=ClassifyBatchUrlsData(
            modelName=MODEL_NAME,
            numClasses=NUM_CLASSES,
            topKApplied=applied_top_k,
            totalImages=len(payload.images),
            completedCount=completed_count,
            failedCount=failed_count,
            results=results,
        ),
    )
