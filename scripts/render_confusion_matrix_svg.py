#!/usr/bin/env python3
"""Render a dependency-free SVG confusion matrix from the evaluation report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DISPLAY_LABELS = [
    "Plastic",
    "Glass",
    "Metal",
    "Paper/cardboard",
    "Organic",
    "General waste",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/evaluation/baseline_export_metrics.json"),
    )
    parser.add_argument("--model", default="pytorch_fp32")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/confusion_matrix.svg")
    )
    return parser.parse_args()


def color(value: float) -> str:
    start = (239, 246, 255)
    end = (29, 78, 216)
    rgb = tuple(round(start[index] + value * (end[index] - start[index])) for index in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    metrics = report["models"][args.model]
    matrix = metrics["confusion_matrix"]
    width, height = 1040, 900
    left, top, cell = 250, 150, 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#172033}.title{font-size:30px;font-weight:700}.subtitle{font-size:17px;fill:#536177}.label{font-size:15px;font-weight:600}.value{font-size:18px;font-weight:700}.axis{font-size:18px;font-weight:700;fill:#334155}</style>',
        '<text class="title" x="520" y="48" text-anchor="middle">SmartWaste Vision — Test Confusion Matrix</text>',
        f'<text class="subtitle" x="520" y="78" text-anchor="middle">PyTorch FP32 · n={report["test_images"]} · Accuracy {metrics["accuracy"]:.2%} · Macro F1 {metrics["macro_f1"]:.3f}</text>',
        f'<text class="axis" x="{left + cell * 3}" y="115" text-anchor="middle">Predicted label</text>',
        f'<text class="axis" transform="translate(55 {top + cell * 3}) rotate(-90)" text-anchor="middle">Actual label</text>',
    ]
    for index, label in enumerate(DISPLAY_LABELS):
        x = left + index * cell + cell / 2
        y = top - 22
        parts.append(
            f'<text class="label" x="{x}" y="{y}" text-anchor="middle" transform="rotate(-30 {x} {y})">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text class="label" x="{left - 18}" y="{top + index * cell + cell / 2 + 5}" text-anchor="end">{html.escape(label)}</text>'
        )
    for row_index, row in enumerate(matrix):
        row_total = sum(row)
        for column_index, count in enumerate(row):
            ratio = count / row_total if row_total else 0.0
            x = left + column_index * cell
            y = top + row_index * cell
            text_color = "#ffffff" if ratio >= 0.55 else "#172033"
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color(ratio)}" stroke="#ffffff" stroke-width="2" rx="5"/>',
                    f'<text class="value" x="{x + cell / 2}" y="{y + cell / 2 - 3}" text-anchor="middle" style="fill:{text_color}">{count}</text>',
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 20}" text-anchor="middle" style="fill:{text_color};font-size:13px">{ratio:.1%}</text>',
                ]
            )
    parts.append(
        '<text class="subtitle" x="520" y="850" text-anchor="middle">Rows are actual classes; columns are predictions. Percentages are normalized per row.</text>'
    )
    parts.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
