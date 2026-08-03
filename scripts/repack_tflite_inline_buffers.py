"""Repack offset-backed TFLite constants as legacy-compatible inline buffers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    args = parse_args()
    source = args.input.read_bytes()
    model_view = schema.Model.GetRootAsModel(source, 0)
    model = schema.ModelT.InitFromObj(model_view)

    repacked = 0
    repacked_bytes = 0
    for index in range(model_view.BuffersLength()):
        source_buffer = model_view.Buffers(index)
        offset = source_buffer.Offset()
        size = source_buffer.Size()
        if offset == 0 and size == 0:
            continue
        if offset <= 0 or size <= 0 or offset + size > len(source):
            raise ValueError(
                f"Invalid external buffer {index}: offset={offset}, size={size}"
            )
        target_buffer = model.buffers[index]
        target_buffer.data = np.frombuffer(
            source[offset : offset + size], dtype=np.uint8
        ).copy()
        target_buffer.offset = 0
        target_buffer.size = 0
        repacked += 1
        repacked_bytes += size

    if repacked == 0:
        raise ValueError("The model has no offset-backed buffers to repack")

    builder = flatbuffers.Builder(initialSize=len(source))
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    output = bytes(builder.Output())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)

    print(f"Repacked buffers: {repacked}")
    print(f"Repacked bytes: {repacked_bytes}")
    print(f"Input bytes: {len(source)}")
    print(f"Output bytes: {len(output)}")
    print(f"Input SHA-256: {sha256(source)}")
    print(f"Output SHA-256: {sha256(output)}")


if __name__ == "__main__":
    main()
