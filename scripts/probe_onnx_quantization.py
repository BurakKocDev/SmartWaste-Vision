#!/usr/bin/env python3
"""Probe ONNX Runtime static-INT8 settings on the fixed conversion canary."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from run_conversion_canary import (
    ArrayCalibrationReader,
    build_inputs,
    comparison,
    create_model,
    run_onnx,
    run_pytorch,
    select_samples,
)


def main() -> None:
    data_root = Path("data/raw/realwaste/realwaste-main/RealWaste")
    source = Path("artifacts/conversion_canary/canary_fp32.preprocessed.onnx")
    output_dir = Path("artifacts/conversion_canary/onnx_probes")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_samples, calibration_samples = select_samples(data_root, 3, 50)
    model, weights = create_model()
    transform = weights.transforms()
    validation_inputs = build_inputs(validation_samples, transform)
    calibration_inputs = build_inputs(calibration_samples, transform)
    validation_reference = run_pytorch(model, validation_inputs)
    calibration_reference = run_pytorch(model, calibration_inputs)
    print(
        "calibration reference range/top1:",
        float(calibration_reference.min()),
        float(calibration_reference.max()),
        dict(Counter(np.argmax(calibration_reference, axis=1).tolist())),
    )

    input_name = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name
    graph = onnx.load(str(source))
    early_nodes = [
        node.name
        for node in graph.graph.node
        if any(
            any(f"features.{index}" in input_name for index in range(4))
            for input_name in node.input
        )
    ]
    print("excluded early nodes:", early_nodes)
    variants = {
        "qdq_s8s8_exclude_features_0_to_3": {
            "quant_format": QuantFormat.QDQ,
            "activation_type": QuantType.QInt8,
            "weight_type": QuantType.QInt8,
            "per_channel": True,
            "extra_options": {"WeightSymmetric": True},
            "nodes_to_exclude": early_nodes,
        },
    }
    for name, variant_options in variants.items():
        output_path = output_dir / f"{name}.onnx"
        reader = ArrayCalibrationReader(input_name, calibration_inputs)
        options = {"reduce_range": False, **variant_options}
        quantize_static(
            str(source),
            str(output_path),
            reader,
            **options,
        )
        output = run_onnx(output_path, validation_inputs, threads=4)
        print(name, comparison(validation_reference, output))
        print("  top1:", np.argmax(output, axis=1).tolist())


if __name__ == "__main__":
    main()
