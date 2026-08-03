from __future__ import annotations

import os
import threading
import time
from pathlib import Path, PosixPath
from typing import Any, Protocol

import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


LABELS = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "organic",
    "general_waste",
]

CATEGORY_METADATA = {
    "plastic": {
        "display_name_tr": "Plastik",
        "display_name_en": "Plastic",
        "bin_color_tr": "Sarı",
        "bin_color_en": "Yellow",
        "bin_color_hex": "#F7C600",
    },
    "glass": {
        "display_name_tr": "Cam",
        "display_name_en": "Glass",
        "bin_color_tr": "Yeşil",
        "bin_color_en": "Green",
        "bin_color_hex": "#238636",
    },
    "metal": {
        "display_name_tr": "Metal",
        "display_name_en": "Metal",
        "bin_color_tr": "Açık gri",
        "bin_color_en": "Light gray",
        "bin_color_hex": "#B0BEC5",
    },
    "paper_cardboard": {
        "display_name_tr": "Kâğıt / karton",
        "display_name_en": "Paper / cardboard",
        "bin_color_tr": "Mavi",
        "bin_color_en": "Blue",
        "bin_color_hex": "#1976D2",
    },
    "organic": {
        "display_name_tr": "Organik",
        "display_name_en": "Organic",
        "bin_color_tr": "Kahverengi",
        "bin_color_en": "Brown",
        "bin_color_hex": "#795548",
    },
    "general_waste": {
        "display_name_tr": "Genel atık",
        "display_name_en": "General waste",
        "bin_color_tr": "Koyu gri",
        "bin_color_en": "Dark gray",
        "bin_color_hex": "#455A64",
    },
}


class PredictorProtocol(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def predict(self, image: Image.Image) -> dict[str, Any]: ...


class TorchPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        threads: int = 4,
        confidence_threshold: float = 0.51,
    ) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits this setting before inter-op work starts.
            pass
        with torch.serialization.safe_globals([PosixPath]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            len(LABELS),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self._model = model.eval()
        self._transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
        self._threshold = confidence_threshold
        self._lock = threading.Lock()
        self._checkpoint_epoch = int(checkpoint["epoch"])
        self._checkpoint_path = checkpoint_path

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": "MobileNetV3-Small",
            "runtime": "PyTorch FP32 CPU",
            "checkpoint_epoch": self._checkpoint_epoch,
            "classes": LABELS,
            "confidence_threshold": self._threshold,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self._transform(image.convert("RGB")).unsqueeze(0)
        started = time.perf_counter_ns()
        with self._lock, torch.inference_mode():
            logits = self._model(tensor)
        inference_ms = (time.perf_counter_ns() - started) / 1_000_000
        probabilities = torch.softmax(logits[0], dim=0)
        values, indices = torch.topk(probabilities, k=3)
        ranking = []
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            label = LABELS[index]
            ranking.append(
                {
                    "label": label,
                    "confidence": value,
                    **CATEGORY_METADATA[label],
                }
            )
        best = ranking[0]
        return {
            "prediction": best,
            "uncertain": best["confidence"] < self._threshold,
            "confidence_threshold": self._threshold,
            "top_k": ranking,
            "inference_ms": inference_ms,
        }


def predictor_from_environment() -> TorchPredictor:
    checkpoint_path = Path(
        os.environ.get(
            "SMARTWASTE_MODEL_PATH",
            "artifacts/training/mobilenet_v3_small_baseline/best_model.pth",
        )
    )
    threads = int(os.environ.get("SMARTWASTE_TORCH_THREADS", "4"))
    threshold = float(os.environ.get("SMARTWASTE_CONFIDENCE_THRESHOLD", "0.51"))
    return TorchPredictor(
        checkpoint_path,
        threads=threads,
        confidence_threshold=threshold,
    )
