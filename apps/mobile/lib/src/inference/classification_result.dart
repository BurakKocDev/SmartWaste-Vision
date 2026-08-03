import 'dart:math' as math;

import '../domain/waste_category.dart';

class ClassScore {
  const ClassScore({required this.category, required this.confidence});

  final WasteCategory category;
  final double confidence;
}

class ClassificationResult {
  const ClassificationResult({
    required this.ranking,
    required this.lowConfidenceThreshold,
    required this.preprocessingDuration,
    required this.inferenceDuration,
  });

  final List<ClassScore> ranking;
  final double lowConfidenceThreshold;
  final Duration preprocessingDuration;
  final Duration inferenceDuration;

  ClassScore get best => ranking.first;
  bool get isLowConfidence => best.confidence < lowConfidenceThreshold;
  Duration get totalDuration => preprocessingDuration + inferenceDuration;
}

List<double> softmax(List<double> logits) {
  if (logits.isEmpty) {
    throw ArgumentError.value(logits, 'logits', 'Logits cannot be empty');
  }
  final maximum = logits.reduce(math.max);
  final exponentials = logits
      .map((value) => math.exp(value - maximum))
      .toList();
  final denominator = exponentials.reduce((left, right) => left + right);
  return exponentials.map((value) => value / denominator).toList();
}

List<ClassScore> rankPredictions({
  required List<String> labels,
  required List<double> logits,
}) {
  if (labels.length != logits.length) {
    throw ArgumentError(
      'Label count ${labels.length} does not match logit count ${logits.length}',
    );
  }
  final probabilities = softmax(logits);
  final ranking = List<ClassScore>.generate(
    labels.length,
    (index) => ClassScore(
      category: WasteCategory.fromCanonicalLabel(labels[index]),
      confidence: probabilities[index],
    ),
    growable: false,
  );
  ranking.sort((left, right) => right.confidence.compareTo(left.confidence));
  return List<ClassScore>.unmodifiable(ranking);
}
