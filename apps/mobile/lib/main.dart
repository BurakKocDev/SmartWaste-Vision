import 'package:flutter/material.dart';

import 'src/app.dart';
import 'src/inference/tflite_waste_classifier.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(SmartWasteApp(classifier: TfliteWasteClassifier()));
}
