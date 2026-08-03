#!/usr/bin/env python3
"""Benchmark SmartWaste artifacts on WSL CPU in isolated worker processes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


RESULT_PREFIX = "SMARTWASTE_BENCHMARK_RESULT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--runtime")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/cpu_baseline.json"),
    )
    return parser.parse_args()


def current_rss_mib() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found in /proc/self/status")


def peak_rss_mib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def make_pytorch_runner(artifact: Path, input_path: Path, threads: int) -> Callable[[], None]:
    import numpy as np
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_small

    torch.set_num_threads(threads)
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    value = torch.from_numpy(np.load(input_path).astype(np.float32, copy=False))

    def run() -> None:
        with torch.inference_mode():
            model(value)

    return run


def make_onnx_runner(artifact: Path, input_path: Path, threads: int) -> Callable[[], None]:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(artifact), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    value = np.load(input_path).astype(np.float32, copy=False)

    def run() -> None:
        session.run(None, {input_name: value})

    return run


def make_litert_runner(
    artifact: Path, input_path: Path, threads: int
) -> tuple[Callable[[], None], str, str | None]:
    import numpy as np
    from ai_edge_litert import interpreter as tfl_interpreter

    delegate_error: str | None = None
    backend = "auto"
    try:
        interpreter = tfl_interpreter.Interpreter(
            model_path=str(artifact), num_threads=threads
        )
        interpreter.allocate_tensors()
    except RuntimeError as error:
        delegate_error = str(error)
        backend = "builtin_without_default_delegates"
        interpreter = tfl_interpreter.Interpreter(
            model_path=str(artifact),
            num_threads=threads,
            experimental_op_resolver_type=(
                tfl_interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    value = np.load(input_path).astype(np.float32, copy=False)
    dtype = np.dtype(input_detail["dtype"])
    if not np.issubdtype(dtype, np.floating):
        scale, zero_point = input_detail["quantization"]
        limits = np.iinfo(dtype)
        value = np.clip(np.rint(value / scale + zero_point), limits.min, limits.max).astype(
            dtype
        )
    else:
        value = value.astype(dtype, copy=False)
    interpreter.set_tensor(input_detail["index"], value)

    def run() -> None:
        interpreter.invoke()

    return run, backend, delegate_error


def run_worker(args: argparse.Namespace) -> int:
    if not args.runtime or not args.artifact or not args.input:
        raise ValueError("Worker requires runtime, artifact and input")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    start_rss = current_rss_mib()
    backend = args.runtime
    delegate_error = None
    if args.runtime == "pytorch_fp32":
        runner = make_pytorch_runner(args.artifact, args.input, args.threads)
    elif args.runtime.startswith("onnx"):
        runner = make_onnx_runner(args.artifact, args.input, args.threads)
    elif args.runtime.startswith("litert"):
        runner, backend, delegate_error = make_litert_runner(
            args.artifact, args.input, args.threads
        )
    else:
        raise ValueError(f"Unknown runtime: {args.runtime}")
    loaded_rss = current_rss_mib()
    for _ in range(args.warmup):
        runner()
    latencies: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        runner()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    post_inference_rss = current_rss_mib()
    result = {
        "runtime": args.runtime,
        "backend": backend,
        "delegate_error": delegate_error,
        "artifact": args.artifact.as_posix(),
        "model_bytes": args.artifact.stat().st_size,
        "threads": args.threads,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "memory_mib": {
            "process_start_rss": start_rss,
            "model_loaded_rss": loaded_rss,
            "post_inference_rss": post_inference_rss,
            "load_delta_rss": loaded_rss - start_rss,
            "peak_rss": peak_rss_mib(),
        },
    }
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
    return 0


def prepare_input(path: Path) -> None:
    import csv
    import numpy as np
    from PIL import Image
    from torchvision.models import MobileNet_V3_Small_Weights

    with Path("data/manifests/realwaste_split.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        first_test = next(row for row in csv.DictReader(handle) if row["split"] == "test")
    image_path = Path("data/raw/realwaste/realwaste-main/RealWaste") / first_test[
        "relative_path"
    ]
    with Image.open(image_path) as image:
        tensor = MobileNet_V3_Small_Weights.DEFAULT.transforms()(image.convert("RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.unsqueeze(0).numpy().astype(np.float32, copy=False))


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def run_coordinator(args: argparse.Namespace) -> int:
    input_path = Path("artifacts/benchmarks/benchmark_input.npy")
    prepare_input(input_path)
    jobs = [
        (
            "pytorch_fp32",
            Path("artifacts/training/mobilenet_v3_small_baseline/best_model.pth"),
        ),
        ("onnx_fp32", Path("artifacts/export/baseline/smartwaste_fp32.onnx")),
        (
            "onnx_int8_full_ptq_rejected",
            Path("artifacts/export/baseline/smartwaste_int8.onnx"),
        ),
        ("litert_fp16", Path("artifacts/export/baseline/smartwaste_fp16.tflite")),
        (
            "litert_int8_weight_only",
            Path("artifacts/export/baseline/smartwaste_int8_weight_only.tflite"),
        ),
        (
            "litert_int8_full_ptq_rejected",
            Path("artifacts/export/baseline/smartwaste_int8.tflite"),
        ),
    ]
    results: list[dict[str, Any]] = []
    for runtime, artifact in jobs:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--runtime",
                runtime,
                "--artifact",
                str(artifact),
                "--input",
                str(input_path),
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
                "--threads",
                str(args.threads),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result_line = next(
            line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)
        )
        result = json.loads(result_line.removeprefix(RESULT_PREFIX))
        results.append(result)
        print(
            runtime,
            "mean_ms",
            round(result["latency_ms"]["mean"], 3),
            "p95_ms",
            round(result["latency_ms"]["p95"], 3),
            "peak_mib",
            round(result["memory_mib"]["peak_rss"], 1),
        )
    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "threads": args.threads,
        "input_shape": [1, 3, 224, 224],
        "batch_size": 1,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.worker else run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
