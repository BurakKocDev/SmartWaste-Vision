#!/usr/bin/env python3
"""Load the real API predictor and classify one local image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.app.predictor import TorchPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/training/mobilenet_v3_small_baseline/best_model.pth"),
    )
    args = parser.parse_args()
    predictor = TorchPredictor(args.checkpoint)
    with Image.open(args.image) as image:
        result = predictor.predict(image.convert("RGB"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
