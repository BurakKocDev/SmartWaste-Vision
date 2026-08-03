import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'classification_result.dart';
import 'image_preprocessor.dart';
import 'model_contract.dart';
import 'waste_classifier.dart';

class TfliteWasteClassifier implements WasteClassifier {
  static const _contractAsset = 'assets/models/model_contract.json';

  Interpreter? _interpreter;
  ModelContract? _contract;

  @override
  Future<void> initialize() async {
    if (_interpreter != null) {
      return;
    }
    final contract = ModelContract.fromJsonString(
      await rootBundle.loadString(_contractAsset),
    );
    final options = InterpreterOptions()..threads = 4;
    final interpreter = await Interpreter.fromAsset(
      contract.modelAsset,
      options: options,
    );
    _validateTensorContract(interpreter, contract);
    _contract = contract;
    _interpreter = interpreter;
  }

  @override
  Future<ClassificationResult> classify(Uint8List encodedImage) async {
    final interpreter = _interpreter;
    final contract = _contract;
    if (interpreter == null || contract == null) {
      throw StateError('Classifier has not been initialized');
    }

    final preprocessingClock = Stopwatch()..start();
    final input = preprocessImage(encodedImage, contract);
    preprocessingClock.stop();

    final output = Float32List(contract.labels.length);
    final inferenceClock = Stopwatch()..start();
    interpreter.run(input.buffer.asUint8List(), output.buffer);
    inferenceClock.stop();

    return ClassificationResult(
      ranking: rankPredictions(
        labels: contract.labels,
        logits: output.toList(),
      ),
      lowConfidenceThreshold: contract.lowConfidenceThreshold,
      preprocessingDuration: preprocessingClock.elapsed,
      inferenceDuration: inferenceClock.elapsed,
    );
  }

  @override
  void close() {
    _interpreter?.close();
    _interpreter = null;
    _contract = null;
  }

  void _validateTensorContract(
    Interpreter interpreter,
    ModelContract contract,
  ) {
    final input = interpreter.getInputTensor(0);
    final output = interpreter.getOutputTensor(0);
    if (!_sameShape(input.shape, contract.inputShape) ||
        input.type != TensorType.float32) {
      interpreter.close();
      throw StateError(
        'Model input contract mismatch: ${input.shape} ${input.type}',
      );
    }
    if (!_sameShape(output.shape, contract.outputShape) ||
        output.type != TensorType.float32 ||
        contract.labels.length != contract.outputShape.last) {
      interpreter.close();
      throw StateError(
        'Model output contract mismatch: ${output.shape} ${output.type}',
      );
    }
  }

  bool _sameShape(List<int> left, List<int> right) {
    if (left.length != right.length) {
      return false;
    }
    for (var index = 0; index < left.length; index++) {
      if (left[index] != right[index]) {
        return false;
      }
    }
    return true;
  }
}
