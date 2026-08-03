#!/usr/bin/env python3
"""Probe PyTorch PT2E post-training quantization before LiteRT conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchao.quantization.pt2e import move_exported_model_to_eval, quantize_pt2e

from litert_torch.quantize import pt2e_quantizer
import litert_torch

from run_conversion_canary import (
    build_inputs,
    comparison,
    create_model,
    inspect_litert,
    run_litert,
    run_pytorch,
    select_samples,
)


def main() -> None:
    data_root = Path("data/raw/realwaste/realwaste-main/RealWaste")
    output_dir = Path("artifacts/conversion_canary/pt2e_probes")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_samples, calibration_samples = select_samples(data_root, 3, 20)
    model, weights = create_model()
    transform = weights.transforms()
    validation_inputs = build_inputs(validation_samples, transform)
    calibration_inputs = build_inputs(calibration_samples, transform)
    reference = run_pytorch(model, validation_inputs)
    sample_args = (torch.from_numpy(validation_inputs[0]),)

    exported = torch.export.export(model, sample_args).module()
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)
    with torch.inference_mode():
        for value in calibration_inputs:
            prepared(torch.from_numpy(value))
    quantized = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    quantized = move_exported_model_to_eval(quantized)

    tflite_path = output_dir / "canary_int8_pt2e.tflite"
    litert_torch.convert(quantized, sample_args, strict_export=False).export(
        str(tflite_path)
    )
    tflite_output = run_litert(tflite_path, validation_inputs, threads=4)
    print("tflite", comparison(reference, tflite_output))
    print("inspection", inspect_litert(tflite_path, threads=4))
    print("reference top1", np.argmax(reference, axis=1).tolist())
    print("tflite top1", np.argmax(tflite_output, axis=1).tolist())

    onnx_path = output_dir / "canary_int8_pt2e.onnx"
    try:
        torch.onnx.export(
            quantized,
            sample_args,
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        print("onnx export: pass", onnx_path)
    except Exception as error:  # The probe must preserve the TFLite result.
        print("onnx export: fail", type(error).__name__, str(error))


if __name__ == "__main__":
    main()
