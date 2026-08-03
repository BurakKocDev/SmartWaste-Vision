#!/usr/bin/env python3
"""Probe usable mixed/dynamic INT8 fallbacks for the trained baseline."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from torchvision.models import MobileNet_V3_Small_Weights

from ai_edge_quantizer import quantizer, recipe

import export_trained_model as trained_export
import run_conversion_canary as conversion


def save_quantized(result: object, path: Path) -> None:
    path.write_bytes(bytes(result.quantized_model))


def main() -> None:
    data_root = Path("data/raw/realwaste/realwaste-main/RealWaste")
    export_dir = Path("artifacts/export/baseline")
    fp32_tflite = export_dir / "smartwaste_fp32_intermediate.tflite"
    fp32_onnx = export_dir / "smartwaste_fp32.preprocessed.onnx"
    model, _ = trained_export.load_model(
        Path("artifacts/training/mobilenet_v3_small_baseline/best_model.pth")
    )
    samples = trained_export.select_parity_samples(
        Path("data/manifests/realwaste_split.csv"), data_root, per_class=10
    )
    inputs = conversion.build_inputs(
        samples, MobileNet_V3_Small_Weights.DEFAULT.transforms()
    )
    reference = conversion.run_pytorch(model, inputs)

    weight_only_path = export_dir / "smartwaste_int8_weight_only.tflite"
    weight_only_quantizer = quantizer.Quantizer(fp32_tflite)
    weight_only_quantizer.load_quantization_recipe(recipe.weight_only_wi8_afp32())
    save_quantized(
        weight_only_quantizer.quantize(
            enable_progress_bar=False, enable_progress_report=False
        ),
        weight_only_path,
    )

    dynamic_path = export_dir / "smartwaste_int8_dynamic.tflite"
    dynamic_quantizer = quantizer.Quantizer(fp32_tflite)
    dynamic_quantizer.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    save_quantized(
        dynamic_quantizer.quantize(
            enable_progress_bar=False, enable_progress_report=False
        ),
        dynamic_path,
    )

    dynamic_onnx_path = export_dir / "smartwaste_dynamic_int8.onnx"
    onnx_creation_error: Exception | None = None
    try:
        quantize_dynamic(
            str(fp32_onnx),
            str(dynamic_onnx_path),
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
    except Exception as error:  # Preserve the two LiteRT fallback results.
        onnx_creation_error = error

    for path in (weight_only_path, dynamic_path):
        output = conversion.run_litert(path, inputs, threads=4)
        print(path.name, conversion.comparison(reference, output))
        print(conversion.inspect_litert(path, threads=4))
        print("bytes", path.stat().st_size)

    try:
        if onnx_creation_error is not None:
            raise onnx_creation_error
        output = conversion.run_onnx(dynamic_onnx_path, inputs, threads=4)
        print(dynamic_onnx_path.name, conversion.comparison(reference, output))
        print("bytes", dynamic_onnx_path.stat().st_size)
    except Exception as error:
        print(dynamic_onnx_path.name, "runtime_fail", type(error).__name__, str(error))


if __name__ == "__main__":
    main()
