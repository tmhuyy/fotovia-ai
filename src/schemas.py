from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ImageRole = Literal["cover", "gallery"]
ClassificationStatus = Literal["completed", "failed"]


class HealthcheckData(BaseModel):
    message: str
    modelName: str
    numClasses: int
    labels: List[str]


class HealthcheckResponse(BaseModel):
    statusCode: int
    data: HealthcheckData


class LegacyPrediction(BaseModel):
    tag: str
    confidence: float


class LegacyClassifyData(BaseModel):
    predictions: List[LegacyPrediction]


class LegacyClassifyResponse(BaseModel):
    statusCode: int
    data: LegacyClassifyData


class Prediction(BaseModel):
    label: str
    confidence: float


class RemoteImageInput(BaseModel):
    imageKey: str = Field(
        ...,
        description="Client-provided identifier for the image, e.g. cover or gallery-1.",
    )
    url: str = Field(
        ...,
        description="Signed URL or public URL pointing to the image to classify.",
    )
    role: ImageRole = Field(
        default="gallery",
        description="The business role of the image inside the portfolio item.",
    )


class ClassifyUrlRequest(BaseModel):
    image: RemoteImageInput
    topK: int = Field(default=3, ge=1, le=10)
    timeoutSeconds: float = Field(default=20.0, gt=0, le=120.0)


class ClassifyBatchUrlsRequest(BaseModel):
    images: List[RemoteImageInput]
    topK: int = Field(default=3, ge=1, le=10)
    timeoutSeconds: float = Field(default=20.0, gt=0, le=120.0)


class ClassifiedImageResult(BaseModel):
    imageKey: str
    role: ImageRole
    status: ClassificationStatus
    predictions: List[Prediction] = Field(default_factory=list)
    error: Optional[str] = None


class ClassificationMetadata(BaseModel):
    modelName: str
    numClasses: int
    topKApplied: int


class ClassifyUrlData(ClassificationMetadata):
    result: ClassifiedImageResult


class ClassifyUrlResponse(BaseModel):
    statusCode: int
    data: ClassifyUrlData


class ClassifyBatchUrlsData(ClassificationMetadata):
    totalImages: int
    completedCount: int
    failedCount: int
    results: List[ClassifiedImageResult]


class ClassifyBatchUrlsResponse(BaseModel):
    statusCode: int
    data: ClassifyBatchUrlsData
