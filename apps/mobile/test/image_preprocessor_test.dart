import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as image_lib;
import 'package:smartwaste_vision/src/inference/image_preprocessor.dart';
import 'package:smartwaste_vision/src/inference/model_contract.dart';

const _contract = ModelContract(
  modelId: 'test',
  modelAsset: 'test.tflite',
  labels: [
    'plastic',
    'glass',
    'metal',
    'paper_cardboard',
    'organic',
    'general_waste',
  ],
  inputShape: [1, 3, 224, 224],
  outputShape: [1, 6],
  resizeShorterSide: 256,
  resizeInterpolation: 'pillow_bilinear_antialias',
  cropSize: 224,
  mean: [0.485, 0.456, 0.406],
  standardDeviation: [0.229, 0.224, 0.225],
  lowConfidenceThreshold: 0.5,
);

void main() {
  test('preprocessImage emits normalized NCHW data', () {
    final source = image_lib.Image(width: 300, height: 500);
    image_lib.fill(source, color: image_lib.ColorRgb8(255, 0, 0));

    final tensor = preprocessImage(image_lib.encodePng(source), _contract);
    const planeSize = 224 * 224;

    expect(tensor, hasLength(3 * planeSize));
    expect(tensor[0], closeTo((1 - 0.485) / 0.229, 1e-5));
    expect(tensor[planeSize], closeTo((0 - 0.456) / 0.224, 1e-5));
    expect(tensor[2 * planeSize], closeTo((0 - 0.406) / 0.225, 1e-5));
  });

  test('preprocessImage rejects undecodable bytes', () {
    expect(
      () => preprocessImage(Uint8List.fromList([1, 2, 3]), _contract),
      throwsFormatException,
    );
  });
}
