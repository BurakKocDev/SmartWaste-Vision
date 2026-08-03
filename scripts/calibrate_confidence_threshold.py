#!/usr/bin/env python3
"""Calibrate the mobile low-confidence threshold on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from torchvision.models import MobileNet_V3_Small_Weights

import run_conversion_canary as conversion


LABELS = conversion.LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/realwaste_split.csv"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/realwaste/realwaste-main/RealWaste"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/export/baseline/smartwaste_fp16.tflite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/confidence_threshold.json"),
    )
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def load_validation_samples(
    manifest_path: Path, data_root: Path
) -> tuple[list[conversion.Sample], np.ndarray]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(
            (row for row in csv.DictReader(handle) if row["split"] == "validation"),
            key=lambda row: row["relative_path"].casefold(),
        )
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    samples = [
        conversion.Sample(
            row["target_label"], row["original_class"], data_root / row["relative_path"]
        )
        for row in rows
    ]
    targets = np.asarray(
        [label_to_index[row["target_label"]] for row in rows], dtype=np.int64
    )
    return samples, targets


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def threshold_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    accepted = confidences >= threshold
    accepted_count = int(accepted.sum())
    correct = predictions == targets
    return {
        "threshold": round(float(threshold), 4),
        "accepted_images": accepted_count,
        "uncertain_images": int(len(targets) - accepted_count),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": (
            float(correct[accepted].mean()) if accepted_count else None
        ),
        "overall_accuracy": float(correct.mean()),
    }


def select_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    target_accepted_accuracy: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        threshold_metrics(targets, probabilities, threshold)
        for threshold in np.arange(0.50, 0.951, 0.01)
    ]
    eligible = [
        item
        for item in candidates
        if item["accepted_accuracy"] is not None
        and item["accepted_accuracy"] >= target_accepted_accuracy
    ]
    if eligible:
        # Lowest threshold gives the greatest coverage for the quality target.
        return eligible[0], candidates
    # Keep the most accurate candidate if the requested target is unattainable.
    fallback = max(
        candidates,
        key=lambda item: (item["accepted_accuracy"] or 0.0, item["coverage"]),
    )
    return fallback, candidates


def main() -> None:
    args = parse_args()
    samples, targets = load_validation_samples(args.manifest, args.data_root)
    inputs = conversion.build_inputs(
        samples, MobileNet_V3_Small_Weights.DEFAULT.transforms()
    )
    logits = conversion.run_litert(args.model, inputs, threads=args.threads)
    probabilities = softmax(logits)
    selected, candidates = select_threshold(
        targets, probabilities, args.target_accepted_accuracy
    )
    report = {
        "schema_version": 1,
        "selection_split": "validation",
        "validation_images": len(samples),
        "model": args.model.as_posix(),
        "target_accepted_accuracy": args.target_accepted_accuracy,
        "selected": selected,
        "candidates": candidates,
        "notes": [
            "Threshold selection never uses the frozen test split.",
            "Confidence is the maximum softmax probability.",
            "Accepted accuracy is measured only on predictions above the threshold.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selected, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
