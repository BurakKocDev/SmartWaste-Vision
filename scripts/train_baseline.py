#!/usr/bin/env python3
"""Train the SmartWaste Vision six-class MobileNetV3-Small FP32 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import InterpolationMode


SEED = 20260803
LABELS = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "organic",
    "general_waste",
]
EXPECTED_SPLITS = {"train", "validation", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/training/mobilenet_v3_small_baseline"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one train/validation/test batch and mark outputs as smoke-only.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty split manifest: {path}")
    labels = {row["target_label"] for row in rows}
    splits = {row["split"] for row in rows}
    if labels != set(LABELS):
        raise ValueError(f"Label contract mismatch: {sorted(labels)} != {LABELS}")
    if splits != EXPECTED_SPLITS:
        raise ValueError(f"Split contract mismatch: {sorted(splits)}")
    if len(rows) != 3998:
        raise ValueError(f"Expected 3998 manifest rows, got {len(rows)}")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate relative_path values in training manifest")
    return rows


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        split: str,
        data_root: Path,
        transform: Any,
    ) -> None:
        self.rows = [row for row in rows if row["split"] == split]
        self.split = split
        self.data_root = data_root.resolve()
        self.transform = transform
        self.label_to_index = {label: index for index, label in enumerate(LABELS)}
        if not self.rows:
            raise ValueError(f"No rows for split: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = (self.data_root / row["relative_path"]).resolve()
        if not image_path.is_relative_to(self.data_root):
            raise ValueError(f"Manifest path escapes data root: {row['relative_path']}")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[row["target_label"]]


def build_transforms() -> tuple[Any, Any]:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.70, 1.0),
                ratio=(0.80, 1.25),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return train_transform, weights.transforms()


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(
    rows: list[dict[str, str]],
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, DataLoader[Any]], dict[str, ManifestDataset]]:
    train_transform, eval_transform = build_transforms()
    datasets = {
        "train": ManifestDataset(rows, "train", data_root, train_transform),
        "validation": ManifestDataset(rows, "validation", data_root, eval_transform),
        "test": ManifestDataset(rows, "test", data_root, eval_transform),
    }
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_seed,
        "persistent_workers": num_workers > 0,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"], shuffle=True, generator=generator, **common
        ),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, datasets


def build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(LABELS))
    nn.init.normal_(model.classifier[-1].weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.classifier[-1].bias)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def class_weights(dataset: ManifestDataset, device: torch.device) -> torch.Tensor:
    counts = Counter(row["target_label"] for row in dataset.rows)
    total = sum(counts.values())
    values = [total / (len(LABELS) * counts[label]) for label in LABELS]
    return torch.tensor(values, dtype=torch.float32, device=device)


def empty_confusion_matrix() -> torch.Tensor:
    return torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)


def update_confusion_matrix(
    matrix: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor
) -> None:
    flat = targets.detach().cpu() * len(LABELS) + predictions.detach().cpu()
    matrix += torch.bincount(flat, minlength=len(LABELS) ** 2).reshape(
        len(LABELS), len(LABELS)
    )


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, Any]:
    values = matrix.to(torch.float64)
    true_positive = values.diag()
    actual = values.sum(dim=1)
    predicted = values.sum(dim=0)
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(actual > 0, true_positive / actual, 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    total = values.sum()
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    max_batches: int | None,
    amp_enabled: bool | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if training and hasattr(model, "features") and not any(
        parameter.requires_grad for parameter in model.features.parameters()
    ):
        # Frozen-backbone warmup must not update BatchNorm running statistics.
        model.features.eval()
    matrix = empty_confusion_matrix()
    total_loss = 0.0
    total_examples = 0
    started = time.perf_counter()
    if amp_enabled is None:
        amp_enabled = device.type == "cuda"

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = targets.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        update_confusion_matrix(matrix, targets, logits.argmax(dim=1))

    if total_examples == 0:
        raise RuntimeError("No batches were processed")
    metrics = metrics_from_confusion(matrix)
    metrics.update(
        {
            "loss": total_loss / total_examples,
            "examples": total_examples,
            "seconds": time.perf_counter() - started,
        }
    )
    return metrics


def optimizer_for_phase(model: nn.Module, args: argparse.Namespace, full: bool) -> Any:
    if full:
        return torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": args.backbone_learning_rate},
                {"params": model.classifier.parameters(), "lr": args.head_learning_rate / 5},
            ],
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    validation_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "state_dict": model.state_dict(),
            "labels": LABELS,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "preprocess": {
                "input_shape": [1, 3, 224, 224],
                "layout": "NCHW",
                "dtype": "float32",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize_shorter_side": 256,
                "center_crop": [224, 224],
            },
            "training_args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.warmup_epochs = 1
        args.max_train_batches = 1
        args.max_eval_batches = 1
        args.num_workers = 0
        if args.output_dir == Path("artifacts/training/mobilenet_v3_small_baseline"):
            args.output_dir = Path("artifacts/training/smoke")
    if args.epochs < 1 or not 0 <= args.warmup_epochs <= args.epochs:
        raise ValueError("Require epochs >= 1 and 0 <= warmup_epochs <= epochs")

    seed_everything(args.seed)
    device = choose_device(args.device)
    rows = load_manifest(args.manifest)
    loaders, datasets = make_loaders(
        rows, args.data_root, args.batch_size, args.num_workers, args.seed
    )
    model = build_model().to(device)
    set_backbone_trainable(model, trainable=args.warmup_epochs == 0)
    optimizer = optimizer_for_phase(model, args, full=args.warmup_epochs == 0)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(datasets["train"], device),
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pth"
    history_path = args.output_dir / "history.json"
    result_path = args.output_dir / "result.json"
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "train_images": len(datasets["train"]),
                "validation_images": len(datasets["validation"]),
                "test_images": len(datasets["test"]),
                "smoke_test": args.smoke_test,
            }
        )
    )

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            optimizer = optimizer_for_phase(model, args, full=True)
        phase = "head_warmup" if epoch < args.warmup_epochs else "full_finetune"
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            scaler,
            args.max_train_batches,
        )
        validation_metrics = run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            max_batches=args.max_eval_batches,
        )
        record = {
            "epoch": epoch + 1,
            "phase": phase,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        write_json(history_path, history)
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "phase": phase,
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
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, best_epoch, validation_metrics, args)
        else:
            epochs_without_improvement += 1
            if not args.smoke_test and epochs_without_improvement >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        optimizer=None,
        scaler=scaler,
        max_batches=args.max_eval_batches,
    )
    result = {
        "schema_version": 1,
        "finished_at_utc": utc_now(),
        "status": "smoke_pass" if args.smoke_test else "complete",
        "valid_for_model_selection": not args.smoke_test,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "test": test_metrics,
        "checkpoint": checkpoint_path.as_posix(),
        "labels": LABELS,
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
