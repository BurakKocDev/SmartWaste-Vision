#!/usr/bin/env python3
"""Evaluate all trained/exported artifacts on the frozen 598-image test split."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from torchvision.models import MobileNet_V3_Small_Weights

import export_trained_model as trained_export
import run_conversion_canary as conversion


LABELS = conversion.LABELS


def classification_metrics(targets: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(actual[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
    }


def load_test_samples(
    manifest_path: Path, data_root: Path
) -> tuple[list[conversion.Sample], np.ndarray]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(
            (row for row in csv.DictReader(handle) if row["split"] == "test"),
            key=lambda row: row["relative_path"].casefold(),
        )
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    samples = [
        conversion.Sample(
            row["target_label"], row["original_class"], data_root / row["relative_path"]
        )
        for row in rows
    ]
    targets = np.asarray([label_to_index[row["target_label"]] for row in rows])
    return samples, targets


def main() -> None:
    data_root = Path("data/raw/realwaste/realwaste-main/RealWaste")
    export_dir = Path("artifacts/export/baseline")
    output_path = Path("artifacts/evaluation/baseline_export_metrics.json")
    samples, targets = load_test_samples(
        Path("data/manifests/realwaste_split.csv"), data_root
    )
    inputs = conversion.build_inputs(
        samples, MobileNet_V3_Small_Weights.DEFAULT.transforms()
    )
    model, checkpoint = trained_export.load_model(
        Path("artifacts/training/mobilenet_v3_small_baseline/best_model.pth")
    )
    runners: dict[str, Callable[[], np.ndarray]] = {
        "pytorch_fp32": lambda: conversion.run_pytorch(model, inputs),
        "onnx_fp32": lambda: conversion.run_onnx(
            export_dir / "smartwaste_fp32.onnx", inputs, threads=4
        ),
        "onnx_int8_full_ptq_rejected": lambda: conversion.run_onnx(
            export_dir / "smartwaste_int8.onnx", inputs, threads=4
        ),
        "litert_fp16": lambda: conversion.run_litert(
            export_dir / "smartwaste_fp16.tflite", inputs, threads=4
        ),
        "litert_int8_weight_only": lambda: conversion.run_litert(
            export_dir / "smartwaste_int8_weight_only.tflite", inputs, threads=4
        ),
        "litert_int8_full_ptq_rejected": lambda: conversion.run_litert(
            export_dir / "smartwaste_int8.tflite", inputs, threads=4
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "test_images": len(samples),
        "checkpoint_epoch": checkpoint["epoch"],
        "models": {},
    }
    reference_logits: np.ndarray | None = None
    for name, runner in runners.items():
        logits = runner()
        if reference_logits is None:
            reference_logits = logits
        metrics = classification_metrics(targets, logits)
        metrics["parity_vs_pytorch"] = conversion.comparison(reference_logits, logits)
        report["models"][name] = metrics
        print(
            name,
            "accuracy",
            round(metrics["accuracy"], 6),
            "macro_f1",
            round(metrics["macro_f1"], 6),
            "agreement",
            round(metrics["parity_vs_pytorch"]["top1_agreement"], 6),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
