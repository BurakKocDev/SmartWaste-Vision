from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from api.app.main import create_app


class FakePredictor:
    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": "fake",
            "runtime": "test",
            "classes": ["plastic"],
            "confidence_threshold": 0.51,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        assert image.mode == "RGB"
        category = {
            "label": "plastic",
            "confidence": 0.93,
            "display_name_tr": "Plastik",
            "display_name_en": "Plastic",
            "bin_color_tr": "Sarı",
            "bin_color_en": "Yellow",
            "bin_color_hex": "#F7C600",
        }
        return {
            "prediction": category,
            "uncertain": False,
            "confidence_threshold": 0.51,
            "top_k": [category],
            "inference_ms": 1.25,
        }


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (220, 30, 20)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_loaded_model() -> None:
    with TestClient(create_app(FakePredictor())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model"]["model"] == "fake"


def test_classify_returns_prediction_contract() -> None:
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post(
            "/v1/classify",
            files={"file": ("waste.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["label"] == "plastic"
    assert body["prediction"]["bin_color_tr"] == "Sarı"
    assert body["uncertain"] is False


def test_classify_rejects_unsupported_content_type() -> None:
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post(
            "/v1/classify",
            files={"file": ("waste.gif", b"GIF89a", "image/gif")},
        )

    assert response.status_code == 415


def test_classify_rejects_invalid_image_bytes() -> None:
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post(
            "/v1/classify",
            files={"file": ("waste.png", b"not an image", "image/png")},
        )

    assert response.status_code == 422
