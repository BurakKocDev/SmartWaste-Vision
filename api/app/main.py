from __future__ import annotations

import io
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, UnidentifiedImageError

from .predictor import PredictorProtocol, predictor_from_environment


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


class CategoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float = Field(ge=0, le=1)
    display_name_tr: str
    display_name_en: str
    bin_color_tr: str
    bin_color_en: str
    bin_color_hex: str


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: CategoryResponse
    uncertain: bool
    confidence_threshold: float = Field(ge=0, le=1)
    top_k: list[CategoryResponse]
    inference_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: str
    model: dict[str, Any]


def create_app(predictor: PredictorProtocol | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.predictor = predictor or predictor_from_environment()
        yield

    app = FastAPI(
        title="SmartWaste Vision API",
        version="0.1.0",
        description="Small PyTorch FP32 reference API for one dominant waste item.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> dict[str, Any]:
        active_predictor: PredictorProtocol = request.app.state.predictor
        return {"status": "ok", "model": active_predictor.metadata}

    @app.post(
        "/v1/classify",
        response_model=ClassificationResponse,
        status_code=status.HTTP_200_OK,
    )
    async def classify(
        request: Request,
        file: UploadFile = File(..., description="JPEG or PNG waste photo"),
    ) -> dict[str, Any]:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only image/jpeg and image/png are supported.",
            )
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Image exceeds the 10 MiB upload limit.",
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as source:
                    source.load()
                    if source.width * source.height > MAX_IMAGE_PIXELS:
                        raise ValueError("Image pixel count exceeds the limit")
                    image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The upload is not a valid supported image.",
            ) from None
        active_predictor: PredictorProtocol = request.app.state.predictor
        return await run_in_threadpool(active_predictor.predict, image)

    return app


app = create_app()
