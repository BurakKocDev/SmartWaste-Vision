#!/usr/bin/env python3
"""Fine-tune the trained baseline with LiteRT-compatible PT2E QAT."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
    move_exported_model_to_eval,
    quantize_pt2e,
)

from litert_torch.quantize import pt2e_quantizer, pt2e_quantizer_utils

import train_baseline as baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("artifacts/training/mobilenet_v3_small_baseline/best_model.pth"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/realwaste/realwaste-main/RealWaste"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/realwaste_split.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/training/qat")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=baseline.SEED)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prepared_qat_model(source_checkpoint: Path) -> torch.fx.GraphModule:
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["labels"] != baseline.LABELS:
        raise ValueError("Source checkpoint label contract mismatch")
    model = baseline.build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    # Trace with batch 2 so export does not specialize the batch dimension to 1.
    sample_args = (torch.zeros(2, 3, 224, 224, dtype=torch.float32),)
    batch_dimension = torch.export.Dim("batch", min=1, max=64)
    exported = torch.export.export(
        model, sample_args, dynamic_shapes=({0: batch_dimension},)
    ).module()
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=True, is_qat=True
        )
    )
    # LiteRT Torch 0.9.2's QAT-only conv+bn pattern helper passes a function to
    # torch.export, which PyTorch 2.12 rejects. This model was exported in eval
    # mode and contains no train-mode BatchNorm pattern, so skipping only these
    # two no-op pattern scans preserves the remaining QAT annotations.
    skipped_annotators = {
        name: pt2e_quantizer_utils.OP_TO_ANNOTATOR[name]
        for name in ("conv_bn", "conv_bn_relu")
    }

    def skip_absent_eval_bn_pattern(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    for name in skipped_annotators:
        pt2e_quantizer_utils.OP_TO_ANNOTATOR[name] = skip_absent_eval_bn_pattern
    try:
        prepared = quantize_pt2e.prepare_qat_pt2e(exported, quantizer)
    finally:
        pt2e_quantizer_utils.OP_TO_ANNOTATOR.update(skipped_annotators)
    allow_exported_model_train_eval(prepared)
    return prepared


def save_qat_checkpoint(
    path: Path,
    model: torch.fx.GraphModule,
    epoch: int,
    validation_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "quantization": "pt2e_qat_int8_per_channel_weights",
            "prepared_state_dict": model.state_dict(),
            "labels": baseline.LABELS,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "source_checkpoint": args.source_checkpoint.as_posix(),
            "training_args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = 1
        args.max_eval_batches = 1
        args.num_workers = 0
        if args.output_dir == Path("artifacts/training/qat"):
            args.output_dir = Path("artifacts/training/qat_smoke")

    baseline.seed_everything(args.seed)
    device = baseline.choose_device(args.device)
    rows = baseline.load_manifest(args.manifest)
    loaders, datasets = baseline.make_loaders(
        rows, args.data_root, args.batch_size, args.num_workers, args.seed
    )
    model = build_prepared_qat_model(args.source_checkpoint).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(
        weight=baseline.class_weights(datasets["train"], device),
        label_smoothing=args.label_smoothing,
    )
    # PT2E fake-quant observers require float32; CUDA autocast is disabled for QAT.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_qat_state.pth"
    history_path = args.output_dir / "history.json"
    result_path = args.output_dir / "result.json"
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = -1

    print(
        json.dumps(
            {
                "event": "qat_start",
                "device": str(device),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "smoke_test": args.smoke_test,
            }
        )
    )
    for epoch in range(args.epochs):
        train_metrics = baseline.run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            scaler,
            args.max_train_batches,
            amp_enabled=False,
        )
        validation_metrics = baseline.run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            max_batches=args.max_eval_batches,
            amp_enabled=False,
        )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        baseline.write_json(history_path, history)
        print(
            json.dumps(
                {
                    "event": "qat_epoch",
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                }
            )
        )
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch + 1
            save_qat_checkpoint(
                checkpoint_path, model, best_epoch, validation_metrics, args
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["prepared_state_dict"], strict=True)
    move_exported_model_to_eval(model)
    test_metrics = baseline.run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        optimizer=None,
        scaler=scaler,
        max_batches=args.max_eval_batches,
        amp_enabled=False,
    )
    result = {
        "schema_version": 1,
        "finished_at_utc": utc_now(),
        "status": "smoke_pass" if args.smoke_test else "complete",
        "valid_for_model_selection": not args.smoke_test,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "test_fake_quant": test_metrics,
        "qat_state": checkpoint_path.as_posix(),
        "source_checkpoint": args.source_checkpoint.as_posix(),
        "labels": baseline.LABELS,
    }
    baseline.write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
