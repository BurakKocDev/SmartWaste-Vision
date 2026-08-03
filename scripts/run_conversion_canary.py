#!/usr/bin/env python3
"""Run the SmartWaste Vision PyTorch -> ONNX -> LiteRT conversion gate.

The canary deliberately uses an untrained six-class head. It proves graph,
preprocessing, quantization, metadata and runtime compatibility; it does not
measure waste-classification accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


SEED = 20260803
LABELS = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "organic",
    "general_waste",
]
SOURCE_CLASS_BY_LABEL = {
    "plastic": ("Plastic",),
    "glass": ("Glass",),
    "metal": ("Metal",),
    "paper_cardboard": ("Cardboard", "Paper"),
    "organic": ("Food Organics",),
    "general_waste": ("Miscellaneous Trash",),
}
# Semantically adjacent ImageNet rows keep the untrained canary head numerically
# well-conditioned. They are initialization only and are not target labels.
IMAGENET_HEAD_ROW_BY_LABEL = {
    "plastic": 898,  # water bottle
    "glass": 440,  # beer bottle
    "metal": 473,  # can opener
    "paper_cardboard": 478,  # carton
    "organic": 954,  # banana
    "general_waste": 412,  # ashcan / trash can
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Sample:
    label: str
    source_class: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/realwaste/realwaste-main/RealWaste"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/conversion_canary"),
    )
    parser.add_argument("--validation-per-class", type=int, default=3)
    parser.add_argument("--calibration-per-class", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def choose_evenly(items: Sequence[Path], count: int) -> list[Path]:
    if len(items) < count:
        raise ValueError(f"Need {count} samples, found only {len(items)}")
    if count == 1:
        return [items[len(items) // 2]]
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(index)] for index in indices]


def select_samples(
    data_root: Path, validation_per_class: int, calibration_per_class: int
) -> tuple[list[Sample], list[Sample]]:
    validation: list[Sample] = []
    calibration: list[Sample] = []
    for label in LABELS:
        source_classes = SOURCE_CLASS_BY_LABEL[label]
        candidates: list[Path] = []
        for source_class in source_classes:
            class_dir = data_root / source_class
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing RealWaste class directory: {class_dir}")
            candidates.extend(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        candidates.sort(key=lambda path: path.as_posix().casefold())
        validation_selected = choose_evenly(candidates, validation_per_class)
        validation_set = set(validation_selected)
        calibration_candidates = [path for path in candidates if path not in validation_set]
        calibration_selected = choose_evenly(calibration_candidates, calibration_per_class)
        for path in validation_selected:
            validation.append(Sample(label, path.parent.name, path))
        for path in calibration_selected:
            calibration.append(Sample(label, path.parent.name, path))

    return validation, calibration


def create_model() -> tuple[nn.Module, MobileNet_V3_Small_Weights]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    imagenet_head = model.classifier[-1]
    selected_rows = [IMAGENET_HEAD_ROW_BY_LABEL[label] for label in LABELS]
    canary_head = nn.Linear(imagenet_head.in_features, len(LABELS))
    with torch.no_grad():
        canary_head.weight.copy_(imagenet_head.weight[selected_rows])
        canary_head.bias.copy_(imagenet_head.bias[selected_rows])
    model.classifier[-1] = canary_head
    return model.eval(), weights


def load_image(path: Path, transform: Any) -> np.ndarray:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).cpu().numpy().astype(np.float32, copy=False)


def build_inputs(samples: Iterable[Sample], transform: Any) -> list[np.ndarray]:
    return [load_image(sample.path, transform) for sample in samples]


def run_pytorch(model: nn.Module, inputs: Sequence[np.ndarray]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for value in inputs:
            outputs.append(model(torch.from_numpy(value)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def run_onnx(path: Path, inputs: Sequence[np.ndarray], threads: int) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    return np.concatenate(
        [session.run(None, {input_name: value})[0] for value in inputs], axis=0
    )


def quantize_for_detail(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if np.issubdtype(dtype, np.floating):
        return value.astype(dtype, copy=False)
    scales = detail["quantization_parameters"]["scales"]
    zero_points = detail["quantization_parameters"]["zero_points"]
    if len(scales) != 1 or len(zero_points) != 1 or float(scales[0]) == 0.0:
        raise ValueError(f"Unsupported boundary quantization: {detail}")
    quantized = np.rint(value / float(scales[0]) + int(zero_points[0]))
    limits = np.iinfo(dtype)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def dequantize_from_detail(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32, copy=False)
    scales = detail["quantization_parameters"]["scales"]
    zero_points = detail["quantization_parameters"]["zero_points"]
    if len(scales) != 1 or len(zero_points) != 1:
        raise ValueError(f"Unsupported output quantization: {detail}")
    return (value.astype(np.float32) - int(zero_points[0])) * float(scales[0])


def match_litert_layout(value: np.ndarray, expected_shape: Sequence[int]) -> np.ndarray:
    shape = tuple(int(item) for item in expected_shape)
    if tuple(value.shape) == shape:
        return value
    nhwc = value.transpose(0, 2, 3, 1)
    if tuple(nhwc.shape) == shape:
        return nhwc
    raise ValueError(f"Cannot match NCHW input {value.shape} to LiteRT input {shape}")


def run_litert(path: Path, inputs: Sequence[np.ndarray], threads: int) -> np.ndarray:
    runner, _, _ = create_litert_interpreter(path, threads)
    input_details = runner.get_input_details()
    output_details = runner.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("Canary expects exactly one LiteRT input and one output")
    input_detail = input_details[0]
    output_detail = output_details[0]
    outputs: list[np.ndarray] = []
    for value in inputs:
        laid_out = match_litert_layout(value, input_detail["shape"])
        runner.set_tensor(input_detail["index"], quantize_for_detail(laid_out, input_detail))
        runner.invoke()
        raw_output = runner.get_tensor(output_detail["index"])
        outputs.append(dequantize_from_detail(raw_output, output_detail))
    return np.concatenate(outputs, axis=0)


def create_litert_interpreter(path: Path, threads: int) -> tuple[Any, str, str | None]:
    """Allocate LiteRT, recording a reference-kernel fallback if XNNPACK rejects it."""
    from ai_edge_litert import interpreter as tfl_interpreter

    try:
        runner = tfl_interpreter.Interpreter(model_path=str(path), num_threads=threads)
        runner.allocate_tensors()
        return runner, "auto", None
    except RuntimeError as error:
        runner = tfl_interpreter.Interpreter(
            model_path=str(path),
            num_threads=threads,
            experimental_op_resolver_type=(
                tfl_interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        runner.allocate_tensors()
        return runner, "builtin_without_default_delegates", str(error)


def tensor_detail(detail: dict[str, Any]) -> dict[str, Any]:
    params = detail["quantization_parameters"]
    return {
        "name": detail["name"],
        "shape": [int(item) for item in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "scales": [float(item) for item in params["scales"]],
        "zero_points": [int(item) for item in params["zero_points"]],
        "quantized_dimension": int(params["quantized_dimension"]),
    }


def inspect_litert(path: Path, threads: int) -> dict[str, Any]:
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    runner, runtime_backend, delegate_error = create_litert_interpreter(path, threads)
    model = tfl_flatbuffer_utils.read_model(path)
    type_names = {
        value: name
        for name, value in vars(qtyping.TensorType).items()
        if name.isupper() and isinstance(value, int)
    }
    all_types: Counter[str] = Counter()
    constant_types: Counter[str] = Counter()
    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            name = type_names.get(tensor.type, str(tensor.type))
            all_types[name] += 1
            data = tfl_flatbuffer_utils.get_tensor_data(tensor, model.buffers)
            if data is not None and data.size:
                constant_types[name] += 1
    return {
        "inputs": [tensor_detail(item) for item in runner.get_input_details()],
        "outputs": [tensor_detail(item) for item in runner.get_output_details()],
        "tensor_type_counts": dict(sorted(all_types.items())),
        "constant_tensor_type_counts": dict(sorted(constant_types.items())),
        "signature_list": runner.get_signature_list(),
        "runtime_backend": runtime_backend,
        "default_delegate_error": delegate_error,
    }


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Output shape mismatch: {reference.shape} != {candidate.shape}")
    finite = bool(np.isfinite(candidate).all())
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    agreement = float(
        np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
    )
    return {
        "sample_count": int(reference.shape[0]),
        "finite": finite,
        "top1_agreement": agreement,
        "max_abs_logit_error": float(delta.max()),
        "mean_abs_logit_error": float(delta.mean()),
    }


class ArrayCalibrationReader:
    def __init__(self, input_name: str, values: Sequence[np.ndarray]) -> None:
        self.input_name = input_name
        self.values = list(values)
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.values):
            return None
        value = self.values[self.index]
        self.index += 1
        return {self.input_name: value}

    def rewind(self) -> None:
        self.index = 0


def export_onnx(model: nn.Module, sample: torch.Tensor, output_path: Path) -> None:
    torch.onnx.export(
        model,
        (sample,),
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    checked = onnx.load(str(output_path))
    onnx.checker.check_model(checked)


def quantize_onnx(
    fp32_path: Path, output_path: Path, calibration: Sequence[np.ndarray]
) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared_path = fp32_path.with_name(f"{fp32_path.stem}.preprocessed.onnx")
    quant_pre_process(
        str(fp32_path),
        str(prepared_path),
        skip_symbolic_shape=False,
        skip_onnx_shape=False,
        skip_optimization=False,
    )
    session = ort.InferenceSession(str(prepared_path), providers=["CPUExecutionProvider"])
    reader = ArrayCalibrationReader(session.get_inputs()[0].name, calibration)
    quantize_static(
        str(prepared_path),
        str(output_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"WeightSymmetric": True},
    )
    checked = onnx.load(str(output_path))
    onnx.checker.check_model(checked)


def convert_litert_fp32(model: nn.Module, sample: torch.Tensor, output_path: Path) -> None:
    import litert_torch

    converted = litert_torch.convert(model, (sample,), strict_export=False)
    converted.export(str(output_path))


def quantize_litert_fp16(fp32_path: Path, output_path: Path) -> None:
    from ai_edge_quantizer import qtyping, quantizer

    qt = quantizer.Quantizer(fp32_path)
    fp16_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
            dtype=qtyping.TensorDataType.FLOAT,
        ),
        compute_precision=qtyping.ComputePrecision.FLOAT,
        explicit_dequantize=True,
    )
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=fp16_config,
        algorithm_key="float_casting",
    )
    result = qt.quantize(enable_progress_bar=False, enable_progress_report=False)
    output_path.write_bytes(bytes(result.quantized_model))


def litert_calibration_data(
    fp32_path: Path, values: Sequence[np.ndarray], threads: int
) -> dict[str, list[dict[str, np.ndarray]]]:
    from ai_edge_litert import interpreter as tfl_interpreter

    runner = tfl_interpreter.Interpreter(model_path=str(fp32_path), num_threads=threads)
    runner.allocate_tensors()
    signature_list = runner.get_signature_list()
    if len(signature_list) != 1:
        raise ValueError(f"Expected one LiteRT signature, got {signature_list}")
    signature_key, signature = next(iter(signature_list.items()))
    if len(signature["inputs"]) != 1:
        raise ValueError(f"Expected one signature input, got {signature}")
    signature_input_name = signature["inputs"][0]
    input_detail = runner.get_input_details()[0]
    samples = [
        {
            signature_input_name: match_litert_layout(value, input_detail["shape"]).astype(
                np.float32, copy=False
            )
        }
        for value in values
    ]
    return {signature_key: samples}


def quantize_litert_int8(
    fp32_path: Path,
    output_path: Path,
    calibration: Sequence[np.ndarray],
    threads: int,
) -> None:
    from ai_edge_quantizer import quantizer, recipe

    qt = quantizer.Quantizer(fp32_path)
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    calibration_data = litert_calibration_data(fp32_path, calibration, threads)
    calibration_result = qt.calibrate(calibration_data, num_threads=threads)
    result = qt.quantize(
        calibration_result, enable_progress_bar=False, enable_progress_report=False
    )
    output_path.write_bytes(bytes(result.quantized_model))


def artifact_record(path: Path) -> dict[str, Any]:
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root if args.data_root.is_absolute() else project_root / args.data_root
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    )
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "conversion_report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "status": "running",
        "purpose": "conversion_compatibility_only_not_accuracy",
        "seed": SEED,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "onnx",
                    "onnxruntime",
                    "litert-torch",
                    "ai-edge-quantizer",
                    "ai-edge-litert",
                )
            },
        },
        "stages": [],
    }

    active_stage = "initialization"
    stage_started = time.perf_counter()

    def finish_stage(name: str, started: float, **details: Any) -> None:
        report["stages"].append(
            {"name": name, "status": "pass", "seconds": time.perf_counter() - started, **details}
        )
        write_json(report_path, report)

    try:
        torch.set_num_threads(args.threads)
        validation_samples, calibration_samples = select_samples(
            data_root, args.validation_per_class, args.calibration_per_class
        )
        model, weights = create_model()
        transform = weights.transforms()
        validation_inputs = build_inputs(validation_samples, transform)
        calibration_inputs = build_inputs(calibration_samples, transform)
        sample_tensor = torch.from_numpy(validation_inputs[0])

        labels_path = output_dir / "labels.json"
        preprocess_path = output_dir / "preprocess.json"
        manifest_path = output_dir / "canary_samples.json"
        write_json(labels_path, {"schema_version": 1, "labels": LABELS})
        write_json(
            preprocess_path,
            {
                "schema_version": 1,
                "input_name": "image",
                "input_layout": "NCHW",
                "input_dtype": "float32",
                "input_shape": [1, 3, 224, 224],
                "resize_shorter_side": 256,
                "center_crop": [224, 224],
                "interpolation": "bilinear",
                "pixel_scale": "uint8_to_0_1",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "output": "six_class_logits",
                "softmax_location": "postprocess",
            },
        )
        write_json(
            manifest_path,
            {
                "validation": [
                    {**asdict(item), "path": item.path.relative_to(data_root).as_posix()}
                    for item in validation_samples
                ],
                "calibration": [
                    {**asdict(item), "path": item.path.relative_to(data_root).as_posix()}
                    for item in calibration_samples
                ],
            },
        )
        finish_stage(
            active_stage,
            stage_started,
            validation_samples=len(validation_samples),
            calibration_samples=len(calibration_samples),
        )

        active_stage = "pytorch_fp32"
        stage_started = time.perf_counter()
        checkpoint_path = output_dir / "canary.pth"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": "mobilenet_v3_small",
                "backbone_weights": str(weights),
                "labels": LABELS,
                "imagenet_head_row_by_label": IMAGENET_HEAD_ROW_BY_LABEL,
                "seed": SEED,
                "trained": False,
                "purpose": "conversion_canary",
            },
            checkpoint_path,
        )
        reference_logits = run_pytorch(model, validation_inputs)
        calibration_reference_logits = run_pytorch(model, calibration_inputs)
        np.save(output_dir / "pytorch_reference_logits.npy", reference_logits)
        np.save(output_dir / "pytorch_calibration_logits.npy", calibration_reference_logits)
        finish_stage(active_stage, stage_started, finite=bool(np.isfinite(reference_logits).all()))

        active_stage = "onnx_fp32"
        stage_started = time.perf_counter()
        onnx_fp32_path = output_dir / "canary_fp32.onnx"
        export_onnx(model, sample_tensor, onnx_fp32_path)
        onnx_fp32_logits = run_onnx(onnx_fp32_path, validation_inputs, args.threads)
        np.save(output_dir / "onnx_fp32_logits.npy", onnx_fp32_logits)
        onnx_fp32_comparison = comparison(reference_logits, onnx_fp32_logits)
        finish_stage(active_stage, stage_started, comparison=onnx_fp32_comparison)

        active_stage = "onnx_int8"
        stage_started = time.perf_counter()
        onnx_int8_path = output_dir / "canary_int8.onnx"
        quantize_onnx(onnx_fp32_path, onnx_int8_path, calibration_inputs)
        onnx_int8_logits = run_onnx(onnx_int8_path, validation_inputs, args.threads)
        np.save(output_dir / "onnx_int8_logits.npy", onnx_int8_logits)
        onnx_int8_comparison = comparison(reference_logits, onnx_int8_logits)
        finish_stage(active_stage, stage_started, comparison=onnx_int8_comparison)

        active_stage = "litert_fp32_intermediate"
        stage_started = time.perf_counter()
        litert_fp32_path = output_dir / "canary_fp32_intermediate.tflite"
        convert_litert_fp32(model, sample_tensor, litert_fp32_path)
        litert_fp32_logits = run_litert(litert_fp32_path, validation_inputs, args.threads)
        np.save(output_dir / "litert_fp32_logits.npy", litert_fp32_logits)
        litert_fp32_comparison = comparison(reference_logits, litert_fp32_logits)
        finish_stage(
            active_stage,
            stage_started,
            comparison=litert_fp32_comparison,
            inspection=inspect_litert(litert_fp32_path, args.threads),
        )

        active_stage = "litert_fp16"
        stage_started = time.perf_counter()
        litert_fp16_path = output_dir / "canary_fp16.tflite"
        quantize_litert_fp16(litert_fp32_path, litert_fp16_path)
        litert_fp16_inspection = inspect_litert(litert_fp16_path, args.threads)
        if litert_fp16_inspection["constant_tensor_type_counts"].get("FLOAT16", 0) == 0:
            raise RuntimeError("FP16 artifact contains no FLOAT16 constant tensors")
        litert_fp16_logits = run_litert(litert_fp16_path, validation_inputs, args.threads)
        np.save(output_dir / "litert_fp16_logits.npy", litert_fp16_logits)
        litert_fp16_comparison = comparison(reference_logits, litert_fp16_logits)
        finish_stage(
            active_stage,
            stage_started,
            comparison=litert_fp16_comparison,
            inspection=litert_fp16_inspection,
        )

        active_stage = "litert_int8"
        stage_started = time.perf_counter()
        litert_int8_path = output_dir / "canary_int8.tflite"
        quantize_litert_int8(
            litert_fp32_path, litert_int8_path, calibration_inputs, args.threads
        )
        litert_int8_inspection = inspect_litert(litert_int8_path, args.threads)
        if litert_int8_inspection["tensor_type_counts"].get("INT8", 0) == 0:
            raise RuntimeError("INT8 artifact contains no INT8 tensors")
        litert_int8_logits = run_litert(litert_int8_path, validation_inputs, args.threads)
        np.save(output_dir / "litert_int8_logits.npy", litert_int8_logits)
        litert_int8_comparison = comparison(reference_logits, litert_int8_logits)
        finish_stage(
            active_stage,
            stage_started,
            comparison=litert_int8_comparison,
            inspection=litert_int8_inspection,
        )

        deploy_artifacts = [
            checkpoint_path,
            onnx_fp32_path,
            onnx_int8_path,
            litert_fp16_path,
            litert_int8_path,
        ]
        report["artifacts"] = {
            path.name: artifact_record(path) for path in deploy_artifacts
        }
        report["metadata_artifacts"] = {
            path.name: artifact_record(path)
            for path in (labels_path, preprocess_path, manifest_path)
        }
        checks = {
            "onnx_fp32_top1_at_least_0_99": onnx_fp32_comparison["top1_agreement"] >= 0.99,
            "onnx_int8_top1_at_least_0_97": onnx_int8_comparison["top1_agreement"] >= 0.97,
            "litert_fp32_top1_at_least_0_99": litert_fp32_comparison["top1_agreement"] >= 0.99,
            "litert_fp16_top1_at_least_0_99": litert_fp16_comparison["top1_agreement"] >= 0.99,
            "litert_int8_top1_at_least_0_97": litert_int8_comparison["top1_agreement"] >= 0.97,
            "litert_int8_default_delegate_compatible": (
                litert_int8_inspection["runtime_backend"] == "auto"
            ),
            "all_outputs_finite": all(
                item["finite"]
                for item in (
                    onnx_fp32_comparison,
                    onnx_int8_comparison,
                    litert_fp32_comparison,
                    litert_fp16_comparison,
                    litert_int8_comparison,
                )
            ),
            "five_deploy_artifacts_exist": all(path.is_file() for path in deploy_artifacts),
        }
        report["acceptance_checks"] = checks
        report["status"] = "pass" if all(checks.values()) else "fail"
        report["decision"] = (
            {
                "conversion_gate": "pass",
                "next_step": "prepare_reproducible_dataset_split",
            }
            if report["status"] == "pass"
            else {
                "conversion_gate": "partial",
                "accepted": ["pytorch_fp32", "onnx_fp32", "litert_fp16"],
                "not_accepted": ["onnx_int8_ptq", "litert_int8_ptq"],
                "next_step": "train_fp32_baseline_then_run_short_qat_finetune_for_int8",
            }
        )
        report["finished_at_utc"] = utc_now()
        write_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}, indent=2))
        return 0 if report["status"] == "pass" else 2
    except Exception as error:
        report["stages"].append(
            {
                "name": active_stage,
                "status": "fail",
                "seconds": time.perf_counter() - stage_started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        report["status"] = "error"
        report["finished_at_utc"] = utc_now()
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
