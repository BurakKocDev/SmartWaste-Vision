"""Inspect tensors and producer/consumer ops in a LiteRT/TFLite model."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

from ai_edge_litert import interpreter as tfl_interpreter
from ai_edge_litert import schema_py_generated as schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tensor", type=int, required=True)
    parser.add_argument("--compare", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interpreter = tfl_interpreter.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    details = {detail["index"]: detail for detail in interpreter.get_tensor_details()}
    print(f"ai-edge-litert: {importlib.metadata.version('ai-edge-litert')}")
    print(f"Tensor: {details.get(args.tensor)}")
    print("Producer/consumer ops:")
    for index, operation in enumerate(interpreter._get_ops_details()):
        inputs = operation["inputs"].tolist()
        outputs = operation["outputs"].tolist()
        if args.tensor in inputs or args.tensor in outputs:
            print(index, operation["op_name"], inputs, outputs)

    model_bytes = args.model.read_bytes()
    model = schema.Model.GetRootAsModel(model_bytes, 0)
    tensor = model.Subgraphs(0).Tensors(args.tensor)
    buffer_index = tensor.Buffer()
    buffer = model.Buffers(buffer_index)
    print(
        "FlatBuffer:",
        {
            "buffer_index": buffer_index,
            "inline_data_length": buffer.DataLength(),
            "offset": buffer.Offset(),
            "size": buffer.Size(),
        },
    )

    if args.compare is not None:
        comparison = tfl_interpreter.Interpreter(model_path=str(args.compare))
        comparison.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        rng = __import__("numpy").random.default_rng(20260804)
        sample = rng.normal(size=input_detail["shape"]).astype(input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        expected = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        comparison_input = comparison.get_input_details()[0]
        comparison.set_tensor(comparison_input["index"], sample)
        comparison.invoke()
        actual = comparison.get_tensor(comparison.get_output_details()[0]["index"])
        difference = abs(expected - actual)
        print(
            "Comparison:",
            {
                "model": str(args.compare),
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "top1_equal": bool(expected.argmax() == actual.argmax()),
            },
        )


if __name__ == "__main__":
    main()
