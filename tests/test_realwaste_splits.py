from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_realwaste_splits.py"
SPEC = importlib.util.spec_from_file_location("build_realwaste_splits", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(path: str, group: str) -> dict[str, str]:
    return {"relative_path": path, "group_id": group}


def test_target_counts_sum_and_match_expected_ratio() -> None:
    assert MODULE.target_counts(100) == {"train": 70, "validation": 15, "test": 15}
    assert sum(MODULE.target_counts(461).values()) == 461


def test_group_allocation_is_deterministic_and_never_splits_a_group() -> None:
    groups = [
        [row("a.jpg", "a"), row("b.jpg", "a")],
        *[[row(f"{index}.jpg", str(index))] for index in range(2, 30)],
    ]
    first = MODULE.allocate_groups(groups, MODULE.SEED, "organic|Food Organics")
    second = MODULE.allocate_groups(groups, MODULE.SEED, "organic|Food Organics")
    assert first == second
    assert first["a.jpg"] == first["b.jpg"]


def test_canonical_mapping_has_six_labels() -> None:
    assert set(MODULE.TARGET_BY_SOURCE_CLASS.values()) == {
        "plastic",
        "glass",
        "metal",
        "paper_cardboard",
        "organic",
        "general_waste",
    }
