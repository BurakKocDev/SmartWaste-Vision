#!/usr/bin/env python3
"""Compare a Dart mobile input tensor with the PyTorch evaluation transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

import export_trained_model as trained_export
import run_conversion_canary as conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--dart-tensor", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/training/mobilenet_v3_small_baseline/best_model.pth"),
    )
    parser.add_argument(
        "--litert-model",
        type=Path,
        default=Path("artifacts/export/baseline/smartwaste_fp16.tflite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/mobile_preprocessing_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mobile = np.fromfile(args.dart_tensor, dtype="<f4").reshape(1, 3, 224, 224)
    with Image.open(args.image) as image:
        reference = (
            MobileNet_V3_Small_Weights.DEFAULT.transforms()(image.convert("RGB"))
            .unsqueeze(0)
            .numpy()
        )
    difference = np.abs(reference - mobile)
    model, _ = trained_export.load_model(args.checkpoint)
    reference_logits = conversion.run_pytorch(model, [reference])
    mobile_pytorch_logits = conversion.run_pytorch(model, [mobile])
    mobile_litert_logits = conversion.run_litert(
        args.litert_model, [mobile], threads=4
    )
    report = {
        "schema_version": 1,
        "image": args.image.as_posix(),
        "tensor_comparison": {
            "mean_absolute_error": float(difference.mean()),
            "p95_absolute_error": float(np.percentile(difference, 95)),
            "max_absolute_error": float(difference.max()),
        },
        "pytorch_reference_top1": conversion.LABELS[
            int(reference_logits.argmax(axis=1)[0])
        ],
        "pytorch_mobile_preprocess_top1": conversion.LABELS[
            int(mobile_pytorch_logits.argmax(axis=1)[0])
        ],
        "litert_mobile_preprocess_top1": conversion.LABELS[
            int(mobile_litert_logits.argmax(axis=1)[0])
        ],
        "mobile_preprocess_logit_comparison": conversion.comparison(
            reference_logits, mobile_pytorch_logits
        ),
        "mobile_runtime_comparison": conversion.comparison(
            mobile_pytorch_logits, mobile_litert_logits
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
