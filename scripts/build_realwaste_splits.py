#!/usr/bin/env python3
"""Build deterministic, stratified and capture-group-aware RealWaste splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SEED = 20260803
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
TARGET_BY_SOURCE_CLASS = {
    "Plastic": "plastic",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper_cardboard",
    "Cardboard": "paper_cardboard",
    "Food Organics": "organic",
    "Miscellaneous Trash": "general_waste",
}
EXCLUDED_SOURCE_CLASSES = {"Textile Trash", "Vegetation"}
KNOWN_CAPTURE_GROUPS = {
    "capture_food_organics_081_082": {
        "Food Organics/Food Organics_81.jpg",
        "Food Organics/Food Organics_82.jpg",
    },
    "capture_metal_655_658": {
        "Metal/Metal_655.jpg",
        "Metal/Metal_658.jpg",
    },
}
OUTPUT_FIELDS = [
    "source",
    "relative_path",
    "sha256",
    "original_class",
    "target_label",
    "split",
    "group_id",
    "size_bytes",
    "width",
    "height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/manifests/realwaste_file_manifest.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/realwaste_split.csv")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/manifests/realwaste_split_summary.json"),
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=Path("data/manifests/int8_calibration.csv"),
    )
    parser.add_argument("--calibration-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def target_counts(total: int) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        SPLIT_RATIOS,
        key=lambda name: (raw[name] - counts[name], SPLIT_RATIOS[name]),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def allocate_groups(
    groups: list[list[dict[str, str]]], seed: int, stratum: str
) -> dict[str, str]:
    rng = random.Random(stable_seed(seed, stratum))
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    targets = target_counts(sum(len(group) for group in shuffled))
    assigned_counts = Counter({name: 0 for name in SPLIT_RATIOS})
    assignment: dict[str, str] = {}

    for group in shuffled:
        split = max(
            SPLIT_RATIOS,
            key=lambda name: (
                targets[name] - assigned_counts[name],
                -assigned_counts[name],
                -list(SPLIT_RATIOS).index(name),
            ),
        )
        for row in group:
            assignment[row["relative_path"]] = split
        assigned_counts[split] += len(group)
    return assignment


def group_id_for(row: dict[str, str]) -> str:
    relative_path = row["relative_path"]
    for group_id, members in KNOWN_CAPTURE_GROUPS.items():
        if relative_path in members:
            return group_id
    return f"file_{row['sha256'][:16]}"


def load_rows(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    included: list[dict[str, str]] = []
    excluded: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_class = row["original_class"]
            if source_class in EXCLUDED_SOURCE_CLASSES:
                excluded[source_class] += 1
                continue
            if source_class not in TARGET_BY_SOURCE_CLASS:
                raise ValueError(f"Unmapped source class: {source_class}")
            if row["status"] != "ok":
                raise ValueError(f"Non-ok image cannot enter split: {row['relative_path']}")
            enriched = dict(row)
            enriched["target_label"] = TARGET_BY_SOURCE_CLASS[source_class]
            enriched["group_id"] = group_id_for(row)
            included.append(enriched)
    return included, excluded


def build_split(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    by_stratum: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_stratum[(row["target_label"], row["original_class"])].append(row)

    assignment: dict[str, str] = {}
    for (target_label, source_class), stratum_rows in sorted(by_stratum.items()):
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in stratum_rows:
            by_group[row["group_id"]].append(row)
        assignment.update(
            allocate_groups(
                list(by_group.values()), seed, f"{target_label}|{source_class}"
            )
        )

    result: list[dict[str, str]] = []
    for row in rows:
        output_row = {field: row[field] for field in OUTPUT_FIELDS if field != "split"}
        output_row["split"] = assignment[row["relative_path"]]
        result.append({field: output_row[field] for field in OUTPUT_FIELDS})
    result.sort(key=lambda row: row["relative_path"].casefold())
    return result


def validate(rows: list[dict[str, str]]) -> None:
    if len(rows) != 3998:
        raise ValueError(f"Expected 3998 included images, got {len(rows)}")
    paths = [row["relative_path"] for row in rows]
    hashes = [row["sha256"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate relative paths in split manifest")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Exact duplicate image hashes in split manifest")
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_group[row["group_id"]].add(row["split"])
    leaked = {group: splits for group, splits in splits_by_group.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Capture groups leaked across splits: {leaked}")
    expected_labels = set(TARGET_BY_SOURCE_CLASS.values())
    if {row["target_label"] for row in rows} != expected_labels:
        raise ValueError("Target label set does not match the six-class contract")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_calibration(
    rows: list[dict[str, str]], per_class: int, seed: int
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_label[row["target_label"]].append(row)
    selected: list[dict[str, Any]] = []
    for label in sorted(by_label):
        candidates = sorted(
            by_label[label],
            key=lambda row: stable_seed(seed, f"calibration|{row['relative_path']}"),
        )
        if len(candidates) < per_class:
            raise ValueError(f"Not enough train samples for calibration class {label}")
        for order, row in enumerate(candidates[:per_class]):
            selected.append(
                {
                    "calibration_order": order,
                    "target_label": label,
                    "relative_path": row["relative_path"],
                    "sha256": row["sha256"],
                    "split": row["split"],
                }
            )
    return selected


def nested_counts(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    values = sorted({row[key] for row in rows})
    for value in values:
        result[value] = dict(
            sorted(Counter(row["split"] for row in rows if row[key] == value).items())
        )
    return result


def main() -> int:
    args = parse_args()
    rows, excluded = load_rows(args.input)
    split_rows = build_split(rows, args.seed)
    validate(split_rows)
    write_csv(args.output, split_rows, OUTPUT_FIELDS)

    calibration = choose_calibration(split_rows, args.calibration_per_class, args.seed)
    calibration_fields = [
        "calibration_order",
        "target_label",
        "relative_path",
        "sha256",
        "split",
    ]
    write_csv(args.calibration_output, calibration, calibration_fields)

    manifest_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "included_images": len(split_rows),
        "excluded_images": sum(excluded.values()),
        "excluded_by_source_class": dict(sorted(excluded.items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in split_rows).items())),
        "target_label_counts": nested_counts(split_rows, "target_label"),
        "source_class_counts": nested_counts(split_rows, "original_class"),
        "known_capture_group_count": len(KNOWN_CAPTURE_GROUPS),
        "group_leakage_count": 0,
        "calibration_images": len(calibration),
        "calibration_per_class": args.calibration_per_class,
        "split_manifest_sha256": manifest_hash,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
