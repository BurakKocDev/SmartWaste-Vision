from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_conversion_canary.py"
SPEC = importlib.util.spec_from_file_location("run_conversion_canary", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_label_contract_is_stable_and_unique() -> None:
    assert MODULE.LABELS == [
        "plastic",
        "glass",
        "metal",
        "paper_cardboard",
        "organic",
        "general_waste",
    ]
    assert len(MODULE.LABELS) == len(set(MODULE.LABELS))


def test_source_mapping_covers_exactly_the_label_contract() -> None:
    assert set(MODULE.SOURCE_CLASS_BY_LABEL) == set(MODULE.LABELS)
    assert MODULE.SOURCE_CLASS_BY_LABEL["paper_cardboard"] == ("Cardboard", "Paper")
    assert MODULE.SOURCE_CLASS_BY_LABEL["organic"] == ("Food Organics",)
    assert "Textile Trash" not in {
        source
        for sources in MODULE.SOURCE_CLASS_BY_LABEL.values()
        for source in sources
    }
    assert "Vegetation" not in {
        source
        for sources in MODULE.SOURCE_CLASS_BY_LABEL.values()
        for source in sources
    }


def test_canary_head_rows_cover_exactly_the_label_contract() -> None:
    assert set(MODULE.IMAGENET_HEAD_ROW_BY_LABEL) == set(MODULE.LABELS)
    assert len(set(MODULE.IMAGENET_HEAD_ROW_BY_LABEL.values())) == len(MODULE.LABELS)


def test_even_sample_selection_is_deterministic() -> None:
    items = [Path(str(index)) for index in range(10)]
    assert MODULE.choose_evenly(items, 3) == [Path("0"), Path("4"), Path("9")]


def test_validation_selection_does_not_depend_on_calibration_count(tmp_path: Path) -> None:
    for source_classes in MODULE.SOURCE_CLASS_BY_LABEL.values():
        for source_class in source_classes:
            class_dir = tmp_path / source_class
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(20):
                (class_dir / f"{index:02d}.jpg").touch()
    validation_a, _ = MODULE.select_samples(tmp_path, 3, 2)
    validation_b, _ = MODULE.select_samples(tmp_path, 3, 7)
    assert [sample.path for sample in validation_a] == [
        sample.path for sample in validation_b
    ]
