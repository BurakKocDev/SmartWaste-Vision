import 'dart:convert';

class ModelContract {
  const ModelContract({
    required this.modelId,
    required this.modelAsset,
    required this.labels,
    required this.inputShape,
    required this.outputShape,
    required this.resizeShorterSide,
    required this.resizeInterpolation,
    required this.cropSize,
    required this.mean,
    required this.standardDeviation,
    required this.lowConfidenceThreshold,
  });

  final String modelId;
  final String modelAsset;
  final List<String> labels;
  final List<int> inputShape;
  final List<int> outputShape;
  final int resizeShorterSide;
  final String resizeInterpolation;
  final int cropSize;
  final List<double> mean;
  final List<double> standardDeviation;
  final double lowConfidenceThreshold;

  factory ModelContract.fromJsonString(String source) {
    final root = jsonDecode(source) as Map<String, dynamic>;
    final input = root['input'] as Map<String, dynamic>;
    final output = root['output'] as Map<String, dynamic>;
    final confidence = root['confidence'] as Map<String, dynamic>;
    final crop = (input['center_crop'] as List<dynamic>).cast<num>();
    if (crop.length != 2 || crop[0] != crop[1]) {
      throw const FormatException('Only square center crops are supported');
    }
    if (input['layout'] != 'NCHW' || input['dtype'] != 'float32') {
      throw const FormatException('Expected a float32 NCHW model input');
    }
    if (input['resize_interpolation'] != 'pillow_bilinear_antialias') {
      throw const FormatException('Expected Pillow-compatible bilinear resize');
    }
    if (output['dtype'] != 'float32' || output['semantics'] != 'logits') {
      throw const FormatException('Expected float32 logit model output');
    }
    return ModelContract(
      modelId: root['model_id'] as String,
      modelAsset: root['asset'] as String,
      labels: List<String>.unmodifiable(
        (root['labels'] as List<dynamic>).cast<String>(),
      ),
      inputShape: List<int>.unmodifiable(
        (input['shape'] as List<dynamic>).map(
          (value) => (value as num).toInt(),
        ),
      ),
      outputShape: List<int>.unmodifiable(
        (output['shape'] as List<dynamic>).map(
          (value) => (value as num).toInt(),
        ),
      ),
      resizeShorterSide: (input['resize_shorter_side'] as num).toInt(),
      resizeInterpolation: input['resize_interpolation'] as String,
      cropSize: crop.first.toInt(),
      mean: List<double>.unmodifiable(
        (input['mean'] as List<dynamic>).map(
          (value) => (value as num).toDouble(),
        ),
      ),
      standardDeviation: List<double>.unmodifiable(
        (input['std'] as List<dynamic>).map(
          (value) => (value as num).toDouble(),
        ),
      ),
      lowConfidenceThreshold: (confidence['low_confidence_threshold'] as num)
          .toDouble(),
    );
  }
}
