import 'package:flutter_test/flutter_test.dart';
import 'package:smartwaste_vision/src/inference/model_contract.dart';

void main() {
  test('parses the checked-in model contract', () {
    const source = '''
    {
      "model_id": "baseline",
      "asset": "assets/model.tflite",
      "labels": ["plastic", "glass"],
      "input": {
        "shape": [1, 3, 224, 224],
        "layout": "NCHW",
        "dtype": "float32",
        "resize_shorter_side": 256,
        "resize_interpolation": "pillow_bilinear_antialias",
        "center_crop": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      },
      "output": {
        "shape": [1, 2],
        "dtype": "float32",
        "semantics": "logits"
      },
      "confidence": {"low_confidence_threshold": 0.5}
    }
    ''';

    final contract = ModelContract.fromJsonString(source);

    expect(contract.inputShape, [1, 3, 224, 224]);
    expect(contract.labels, ['plastic', 'glass']);
    expect(contract.lowConfidenceThreshold, 0.5);
  });
}
