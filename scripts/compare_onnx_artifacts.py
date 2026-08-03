#!/usr/bin/env python3
"""Compare existing canary ONNX artifacts with the deterministic PyTorch model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_conversion_canary import (
    build_inputs,
    comparison,
    create_model,
    run_onnx,
    run_pytorch,
    select_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/realwaste/realwaste-main/RealWaste"),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    validation, _ = select_samples(args.data_root, 3, 5)
    model, weights = create_model()
    inputs = build_inputs(validation, weights.transforms())
    reference = run_pytorch(model, inputs)
    print("pytorch top1:", np.argmax(reference, axis=1).tolist())
    for artifact in args.artifacts:
        output = run_onnx(artifact, inputs, args.threads)
        print(artifact, comparison(reference, output))
        print("  top1:", np.argmax(output, axis=1).tolist())


if __name__ == "__main__":
    main()
