import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:smartwaste_vision/src/app.dart';
import 'package:smartwaste_vision/src/inference/classification_result.dart';
import 'package:smartwaste_vision/src/inference/waste_classifier.dart';

class _FakeClassifier implements WasteClassifier {
  bool initialized = false;
  bool closed = false;

  @override
  Future<void> initialize() async => initialized = true;

  @override
  Future<ClassificationResult> classify(Uint8List encodedImage) {
    throw UnimplementedError();
  }

  @override
  void close() => closed = true;
}

void main() {
  testWidgets('shows the offline camera and gallery flow', (tester) async {
    await tester.binding.setSurfaceSize(const Size(430, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final classifier = _FakeClassifier();

    await tester.pumpWidget(SmartWasteApp(classifier: classifier));
    await tester.pumpAndSettle();

    expect(classifier.initialized, isTrue);
    expect(find.text('SmartWaste Vision'), findsOneWidget);
    expect(find.text('Çevrimdışı'), findsOneWidget);
    expect(find.byKey(const Key('cameraButton')), findsOneWidget);
    expect(find.byKey(const Key('galleryButton')), findsOneWidget);
    expect(
      find.textContaining('fotoğraf hiçbir sunucuya gönderilmez'),
      findsOneWidget,
    );
  });
}
