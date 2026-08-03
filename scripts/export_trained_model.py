#!/usr/bin/env python3
"""Export and parity-test the trained SmartWaste Vision baseline."""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

import run_conversion_canary as conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/training/mobilenet_v3_small_baseline/best_model.pth"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/realwaste/realwaste-main/RealWaste"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/manifests/realwaste_split.csv"),
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("data/manifests/int8_calibration.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/export/baseline")
    )
    parser.add_argument("--parity-per-class", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_model(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["labels"] != conversion.LABELS:
        raise ValueError(
            f"Checkpoint label order mismatch: {checkpoint['labels']} != {conversion.LABELS}"
        )
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(conversion.LABELS))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), checkpoint


def select_parity_samples(
    manifest_path: Path, data_root: Path, per_class: int
) -> list[conversion.Sample]:
    rows = [row for row in read_csv(manifest_path) if row["split"] == "validation"]
    result: list[conversion.Sample] = []
    for label in conversion.LABELS:
        candidates = sorted(
            (row for row in rows if row["target_label"] == label),
            key=lambda row: (row["sha256"], row["relative_path"]),
        )
        if len(candidates) < per_class:
            raise ValueError(f"Not enough validation rows for {label}")
        for row in candidates[:per_class]:
            path = data_root / row["relative_path"]
            result.append(conversion.Sample(label, row["original_class"], path))
    return result


def select_calibration_samples(
    manifest_path: Path, data_root: Path
) -> list[conversion.Sample]:
    rows = read_csv(manifest_path)
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Calibration manifest must contain train rows only")
    return [
        conversion.Sample(
            row["target_label"],
            (data_root / row["relative_path"]).parent.name,
            data_root / row["relative_path"],
        )
        for row in rows
    ]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "export_report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "status": "running",
        "source_checkpoint": conversion.artifact_record(args.checkpoint.resolve()),
        "stages": [],
    }
    active_stage = "initialization"
    started = time.perf_counter()

    def record(name: str, stage_started: float, **details: Any) -> None:
        report["stages"].append(
            {
                "name": name,
                "status": "pass",
                "seconds": time.perf_counter() - stage_started,
                **details,
            }
        )
        conversion.write_json(report_path, report)

    try:
        torch.set_num_threads(args.threads)
        model, checkpoint = load_model(args.checkpoint)
        parity_samples = select_parity_samples(
            args.split_manifest, data_root, args.parity_per_class
        )
        calibration_samples = select_calibration_samples(
            args.calibration_manifest, data_root
        )
        transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
        parity_inputs = conversion.build_inputs(parity_samples, transform)
        calibration_inputs = conversion.build_inputs(calibration_samples, transform)
        sample_tensor = torch.from_numpy(parity_inputs[0])
        reference = conversion.run_pytorch(model, parity_inputs)
        np.save(output_dir / "pytorch_reference_logits.npy", reference)
        conversion.write_json(
            output_dir / "labels.json",
            {"schema_version": 1, "labels": conversion.LABELS},
        )
        conversion.write_json(
            output_dir / "preprocess.json",
            {"schema_version": 1, **checkpoint["preprocess"]},
        )
        record(
            active_stage,
            started,
            parity_samples=len(parity_samples),
            calibration_samples=len(calibration_samples),
            checkpoint_epoch=checkpoint["epoch"],
        )

        active_stage = "onnx_fp32"
        started = time.perf_counter()
        onnx_fp32 = output_dir / "smartwaste_fp32.onnx"
        conversion.export_onnx(model, sample_tensor, onnx_fp32)
        output = conversion.run_onnx(onnx_fp32, parity_inputs, args.threads)
        onnx_fp32_comparison = conversion.comparison(reference, output)
        record(active_stage, started, comparison=onnx_fp32_comparison)

        active_stage = "onnx_int8_ptq"
        started = time.perf_counter()
        onnx_int8 = output_dir / "smartwaste_int8.onnx"
        conversion.quantize_onnx(onnx_fp32, onnx_int8, calibration_inputs)
        output = conversion.run_onnx(onnx_int8, parity_inputs, args.threads)
        onnx_int8_comparison = conversion.comparison(reference, output)
        record(active_stage, started, comparison=onnx_int8_comparison)

        active_stage = "litert_fp32_intermediate"
        started = time.perf_counter()
        litert_fp32 = output_dir / "smartwaste_fp32_intermediate.tflite"
        conversion.convert_litert_fp32(model, sample_tensor, litert_fp32)
        output = conversion.run_litert(litert_fp32, parity_inputs, args.threads)
        litert_fp32_comparison = conversion.comparison(reference, output)
        record(
            active_stage,
            started,
            comparison=litert_fp32_comparison,
            inspection=conversion.inspect_litert(litert_fp32, args.threads),
        )

        active_stage = "litert_fp16"
        started = time.perf_counter()
        litert_fp16 = output_dir / "smartwaste_fp16.tflite"
        conversion.quantize_litert_fp16(litert_fp32, litert_fp16)
        fp16_inspection = conversion.inspect_litert(litert_fp16, args.threads)
        if fp16_inspection["constant_tensor_type_counts"].get("FLOAT16", 0) == 0:
            raise RuntimeError("FP16 export contains no FLOAT16 constant tensors")
        output = conversion.run_litert(litert_fp16, parity_inputs, args.threads)
        litert_fp16_comparison = conversion.comparison(reference, output)
        record(
            active_stage,
            started,
            comparison=litert_fp16_comparison,
            inspection=fp16_inspection,
        )

        active_stage = "litert_int8_ptq"
        started = time.perf_counter()
        litert_int8 = output_dir / "smartwaste_int8.tflite"
        conversion.quantize_litert_int8(
            litert_fp32, litert_int8, calibration_inputs, args.threads
        )
        int8_inspection = conversion.inspect_litert(litert_int8, args.threads)
        output = conversion.run_litert(litert_int8, parity_inputs, args.threads)
        litert_int8_comparison = conversion.comparison(reference, output)
        record(
            active_stage,
            started,
            comparison=litert_int8_comparison,
            inspection=int8_inspection,
        )

        artifacts = [onnx_fp32, onnx_int8, litert_fp16, litert_int8]
        report["artifacts"] = {
            path.name: conversion.artifact_record(path) for path in artifacts
        }
        checks = {
            "onnx_fp32_top1_at_least_0_99": onnx_fp32_comparison["top1_agreement"] >= 0.99,
            "onnx_int8_top1_at_least_0_97": onnx_int8_comparison["top1_agreement"] >= 0.97,
            "litert_fp32_top1_at_least_0_99": litert_fp32_comparison["top1_agreement"] >= 0.99,
            "litert_fp16_top1_at_least_0_99": litert_fp16_comparison["top1_agreement"] >= 0.99,
            "litert_int8_top1_at_least_0_97": litert_int8_comparison["top1_agreement"] >= 0.97,
            "litert_int8_default_delegate_compatible": int8_inspection["runtime_backend"] == "auto",
        }
        report["acceptance_checks"] = checks
        report["status"] = "pass" if all(checks.values()) else "partial"
        report["finished_at_utc"] = utc_now()
        conversion.write_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}, indent=2))
        return 0 if report["status"] == "pass" else 2
    except Exception as error:
        report["stages"].append(
            {
                "name": active_stage,
                "status": "fail",
                "seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        report["status"] = "error"
        report["finished_at_utc"] = utc_now()
        conversion.write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
