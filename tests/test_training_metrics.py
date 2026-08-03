from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline.py"
SPEC = importlib.util.spec_from_file_location("train_baseline", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_metrics_are_perfect_for_identity_confusion_matrix() -> None:
    matrix = torch.eye(len(MODULE.LABELS), dtype=torch.int64) * 3
    metrics = MODULE.metrics_from_confusion(matrix)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_confusion_matrix_rows_are_actual_and_columns_are_predicted() -> None:
    matrix = MODULE.empty_confusion_matrix()
    targets = torch.tensor([0, 0, 1, 2])
    predictions = torch.tensor([0, 1, 1, 0])
    MODULE.update_confusion_matrix(matrix, targets, predictions)
    assert matrix[0, 0] == 1
    assert matrix[0, 1] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 0] == 1


def test_training_and_conversion_label_contracts_match() -> None:
    conversion_path = SCRIPT_PATH.parent / "run_conversion_canary.py"
    conversion_spec = importlib.util.spec_from_file_location(
        "run_conversion_canary_for_training_test", conversion_path
    )
    assert conversion_spec and conversion_spec.loader
    conversion = importlib.util.module_from_spec(conversion_spec)
    sys.modules[conversion_spec.name] = conversion
    conversion_spec.loader.exec_module(conversion)
    assert MODULE.LABELS == conversion.LABELS
