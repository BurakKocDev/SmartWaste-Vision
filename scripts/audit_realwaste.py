"""Audit a local RealWaste download without modifying the raw dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


EXPECTED_CLASS_COUNTS = {
    "Cardboard": 461,
    "Food Organics": 411,
    "Glass": 420,
    "Metal": 790,
    "Miscellaneous Trash": 495,
    "Paper": 500,
    "Plastic": 921,
    "Textile Trash": 318,
    "Vegetation": 436,
}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dct_basis(size: int) -> np.ndarray:
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    positions = np.arange(size, dtype=np.float64)[None, :]
    basis = np.cos((np.pi / size) * (positions + 0.5) * frequencies)
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    return basis * np.sqrt(2.0 / size)


DCT_32 = dct_basis(32)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    """Return a 64-bit dHash as a fixed-width hexadecimal string."""
    normalized = ImageOps.exif_transpose(image).convert("L")
    resized = normalized.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.get_flattened_data())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | (
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{value:016x}"


def perceptual_hash(image: Image.Image) -> str:
    """Return a 64-bit DCT-based pHash as a fixed-width hexadecimal string."""
    normalized = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(
        normalized.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
    )
    transformed = DCT_32 @ pixels @ DCT_32.T
    low_frequencies = transformed[:8, :8].reshape(-1)
    threshold = float(np.median(low_frequencies[1:]))
    value = 0
    for coefficient in low_frequencies:
        value = (value << 1) | (coefficient > threshold)
    return f"{value:016x}"


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as candidate:
        candidate.verify()
    with Image.open(path) as image:
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format or "unknown",
            "mode": image.mode,
            "dhash": difference_hash(image),
            "phash": perceptual_hash(image),
        }


def duplicate_groups(
    grouped_paths: dict[str, list[str]], *, require_multiple: bool = True
) -> list[dict[str, Any]]:
    groups = []
    for fingerprint, paths in sorted(grouped_paths.items()):
        if require_multiple and len(paths) < 2:
            continue
        groups.append(
            {
                "fingerprint": fingerprint,
                "count": len(paths),
                "paths": sorted(paths),
            }
        )
    return groups


def near_hash_groups(
    hash_paths: dict[str, list[str]], threshold: int, max_examples: int = 500
) -> tuple[int, list[dict[str, Any]], bool]:
    """Find distinct perceptual hash values within a Hamming threshold."""
    values = sorted((int(value, 16), value) for value in hash_paths)
    total_pairs = 0
    examples: list[dict[str, Any]] = []
    for left_index, (left_value, left_hex) in enumerate(values):
        for right_value, right_hex in values[left_index + 1 :]:
            distance = (left_value ^ right_value).bit_count()
            if distance > threshold:
                continue
            total_pairs += 1
            if len(examples) >= max_examples:
                continue
            examples.append(
                {
                    "distance": distance,
                    "left_hash": left_hex,
                    "right_hash": right_hex,
                    "left_count": len(hash_paths[left_hex]),
                    "right_count": len(hash_paths[right_hex]),
                    "left_example": sorted(hash_paths[left_hex])[0],
                    "right_example": sorted(hash_paths[right_hex])[0],
                }
            )
    return total_pairs, examples, total_pairs > len(examples)


def audit(
    dataset_root: Path,
    output_dir: Path,
    dhash_threshold: int,
    phash_threshold: int,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_directories = sorted(
        directory for directory in dataset_root.iterdir() if directory.is_dir()
    )
    class_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    sha_paths: dict[str, list[str]] = defaultdict(list)
    dhash_paths: dict[str, list[str]] = defaultdict(list)
    phash_paths: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    corrupt_files: list[dict[str, str]] = []
    unsupported_files: list[str] = []
    total_bytes = 0

    for class_directory in class_directories:
        for path in sorted(candidate for candidate in class_directory.rglob("*") if candidate.is_file()):
            relative_path = path.relative_to(dataset_root).as_posix()
            suffix = path.suffix.lower()
            suffix_counts[suffix or "<none>"] += 1
            if suffix not in SUPPORTED_SUFFIXES:
                unsupported_files.append(relative_path)
                continue

            file_size = path.stat().st_size
            file_hash = sha256_file(path)
            total_bytes += file_size
            class_counts[class_directory.name] += 1
            sha_paths[file_hash].append(relative_path)

            row: dict[str, Any] = {
                "source": "realwaste_uci_908",
                "relative_path": relative_path,
                "original_class": class_directory.name,
                "size_bytes": file_size,
                "sha256": file_hash,
                "width": "",
                "height": "",
                "format": "",
                "mode": "",
                "dhash": "",
                "phash": "",
                "status": "ok",
                "error": "",
            }
            try:
                metadata = image_metadata(path)
                row.update(metadata)
                format_counts[metadata["format"]] += 1
                mode_counts[metadata["mode"]] += 1
                dimension_counts[f'{metadata["width"]}x{metadata["height"]}'] += 1
                dhash_paths[metadata["dhash"]].append(relative_path)
                phash_paths[metadata["phash"]].append(relative_path)
            except Exception as exc:  # Pillow exposes several decoder exceptions.
                row["status"] = "corrupt"
                row["error"] = f"{type(exc).__name__}: {exc}"
                corrupt_files.append(
                    {"path": relative_path, "error": row["error"]}
                )
            rows.append(row)

    exact_groups = duplicate_groups(sha_paths)
    same_dhash_groups = [
        group
        for group in duplicate_groups(dhash_paths)
        if len({row["sha256"] for row in rows if row["relative_path"] in group["paths"]})
        > 1
    ]
    same_phash_groups = [
        group
        for group in duplicate_groups(phash_paths)
        if len({row["sha256"] for row in rows if row["relative_path"] in group["paths"]})
        > 1
    ]
    near_dhash_total, near_dhash_examples, near_dhash_truncated = near_hash_groups(
        dhash_paths, dhash_threshold
    )
    near_phash_total, near_phash_examples, near_phash_truncated = near_hash_groups(
        phash_paths, phash_threshold
    )

    inventory_digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["relative_path"]):
        inventory_digest.update(
            f'{row["relative_path"]}\t{row["sha256"]}\n'.encode("utf-8")
        )

    expected_mismatches = {
        class_name: {"expected": expected, "actual": class_counts[class_name]}
        for class_name, expected in EXPECTED_CLASS_COUNTS.items()
        if class_counts[class_name] != expected
    }
    unexpected_classes = sorted(set(class_counts) - set(EXPECTED_CLASS_COUNTS))
    missing_classes = sorted(set(EXPECTED_CLASS_COUNTS) - set(class_counts))

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "RealWaste",
        "dataset_root": str(dataset_root),
        "source_url": "https://archive.ics.uci.edu/dataset/908/realwaste",
        "status": "pass"
        if not corrupt_files
        and not unsupported_files
        and not expected_mismatches
        and not unexpected_classes
        and not missing_classes
        else "review_required",
        "inventory_sha256": inventory_digest.hexdigest(),
        "summary": {
            "image_count": len(rows),
            "total_bytes": total_bytes,
            "total_mib": round(total_bytes / (1024 * 1024), 2),
            "class_count": len(class_counts),
            "corrupt_count": len(corrupt_files),
            "unsupported_count": len(unsupported_files),
            "exact_duplicate_group_count": len(exact_groups),
            "exact_duplicate_file_count": sum(group["count"] for group in exact_groups),
            "same_dhash_distinct_file_group_count": len(same_dhash_groups),
            "same_phash_distinct_file_group_count": len(same_phash_groups),
            "near_dhash_pair_count": near_dhash_total,
            "near_dhash_threshold": dhash_threshold,
            "near_phash_pair_count": near_phash_total,
            "near_phash_threshold": phash_threshold,
        },
        "class_counts": dict(sorted(class_counts.items())),
        "expected_class_counts": EXPECTED_CLASS_COUNTS,
        "expected_count_mismatches": expected_mismatches,
        "unexpected_classes": unexpected_classes,
        "missing_classes": missing_classes,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "corrupt_files": corrupt_files,
        "unsupported_files": unsupported_files,
        "exact_duplicate_groups": exact_groups,
        "same_dhash_distinct_file_groups": same_dhash_groups,
        "same_phash_distinct_file_groups": same_phash_groups,
        "near_dhash_pairs": {
            "count": near_dhash_total,
            "threshold": dhash_threshold,
            "examples_truncated": near_dhash_truncated,
            "examples": near_dhash_examples,
        },
        "near_phash_pairs": {
            "count": near_phash_total,
            "threshold": phash_threshold,
            "examples_truncated": near_phash_truncated,
            "examples": near_phash_examples,
        },
    }

    manifest_path = output_dir / "realwaste_file_manifest.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = output_dir / "realwaste_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--dhash-threshold", type=int, default=2)
    parser.add_argument("--phash-threshold", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        args.dataset_root,
        args.output_dir,
        args.dhash_threshold,
        args.phash_threshold,
    )
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
