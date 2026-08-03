#!/usr/bin/env python3
"""Evaluate tensors produced by the exact Dart mobile preprocessing code."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import calibrate_confidence_threshold as calibration
import evaluate_exported_models as evaluation
import run_conversion_canary as conversion


LABELS = conversion.LABELS
INPUT_VALUES = 3 * 224 * 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/realwaste_split.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/export/baseline/smartwaste_fp16.tflite"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
    )
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def load_targets(manifest_path: Path, split: str) -> np.ndarray:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    for label in LABELS:
        selected.extend(
            sorted(
                (
                    row
                    for row in rows
                    if row["split"] == split and row["target_label"] == label
                ),
                key=lambda row: row["relative_path"].casefold(),
            )
        )
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    return np.asarray(
        [label_to_index[row["target_label"]] for row in selected], dtype=np.int64
    )


def main() -> None:
    args = parse_args()
    targets = load_targets(args.manifest, args.split)
    values = np.fromfile(args.tensor, dtype="<f4")
    expected_values = len(targets) * INPUT_VALUES
    if values.size != expected_values:
        raise ValueError(
            f"Tensor file contains {values.size} values; expected {expected_values}"
        )
    inputs = values.reshape(len(targets), 1, 3, 224, 224)
    logits = conversion.run_litert(args.model, inputs, threads=args.threads)
    metrics: dict[str, Any] = evaluation.classification_metrics(targets, logits)
    report: dict[str, Any] = {
        "schema_version": 1,
        "split": args.split,
        "images": len(targets),
        "preprocessing": "apps/mobile/lib/src/inference/image_preprocessor.dart",
        "runtime": "LiteRT FP16 via Python smoke runtime",
        "metrics": metrics,
    }
    if args.split == "validation":
        probabilities = calibration.softmax(logits)
        selected, candidates = calibration.select_threshold(
            targets, probabilities, args.target_accepted_accuracy
        )
        report["confidence_threshold"] = {
            "target_accepted_accuracy": args.target_accepted_accuracy,
            "selected": selected,
            "candidates": candidates,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"mobile_preprocessing_{args.split}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "images": len(targets),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "confidence_threshold": report.get("confidence_threshold", {}).get(
                    "selected"
                ),
                "output": output_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
