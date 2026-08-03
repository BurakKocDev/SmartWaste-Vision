import 'package:flutter_test/flutter_test.dart';
import 'package:smartwaste_vision/src/domain/waste_category.dart';
import 'package:smartwaste_vision/src/inference/classification_result.dart';

void main() {
  test('softmax is stable and sums to one', () {
    final probabilities = softmax([1001, 1002, 1003]);

    expect(
      probabilities.reduce((left, right) => left + right),
      closeTo(1, 1e-12),
    );
    expect(probabilities.last, greaterThan(probabilities.first));
  });

  test('rankPredictions preserves the exported label order', () {
    final ranking = rankPredictions(
      labels: const [
        'plastic',
        'glass',
        'metal',
        'paper_cardboard',
        'organic',
        'general_waste',
      ],
      logits: const [0, 1, 5, 2, -1, -2],
    );

    expect(ranking.first.category, WasteCategory.metal);
    expect(ranking.first.confidence, greaterThan(0.9));
  });

  test('low-confidence status uses the calibrated threshold', () {
    const result = ClassificationResult(
      ranking: [
        ClassScore(category: WasteCategory.glass, confidence: 0.49),
        ClassScore(category: WasteCategory.plastic, confidence: 0.40),
      ],
      lowConfidenceThreshold: 0.50,
      preprocessingDuration: Duration(milliseconds: 8),
      inferenceDuration: Duration(milliseconds: 3),
    );

    expect(result.isLowConfidence, isTrue);
    expect(result.totalDuration, const Duration(milliseconds: 11));
  });
}
