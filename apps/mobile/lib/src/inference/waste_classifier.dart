import 'dart:typed_data';

import 'classification_result.dart';

abstract interface class WasteClassifier {
  Future<void> initialize();

  Future<ClassificationResult> classify(Uint8List encodedImage);

  void close();
}
