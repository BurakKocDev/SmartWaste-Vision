import 'dart:io';

import 'package:smartwaste_vision/src/inference/image_preprocessor.dart';
import 'package:smartwaste_vision/src/inference/model_contract.dart';

Future<void> main(List<String> arguments) async {
  if (arguments.length < 3 || arguments.length > 4) {
    stderr.writeln(
      'Usage: dart run tool/export_preprocessed_tensor.dart '
      '<image> <model-contract.json> <output.f32> '
      '[pillow|linear|average|cubic]',
    );
    exitCode = 64;
    return;
  }
  final imageBytes = await File(arguments[0]).readAsBytes();
  final contract = ModelContract.fromJsonString(
    await File(arguments[1]).readAsString(),
  );
  final resizeMode = _resizeMode(
    arguments.length == 4 ? arguments[3] : 'pillow',
  );
  final tensor = preprocessImage(imageBytes, contract, resizeMode: resizeMode);
  await File(
    arguments[2],
  ).writeAsBytes(tensor.buffer.asUint8List(), flush: true);
  stdout.writeln('${tensor.length} float32 values -> ${arguments[2]}');
}

MobileResizeMode _resizeMode(String name) => switch (name) {
  'pillow' => MobileResizeMode.pillowBilinearAntialias,
  'linear' => MobileResizeMode.linear,
  'average' => MobileResizeMode.average,
  'cubic' => MobileResizeMode.cubic,
  _ => throw ArgumentError.value(name, 'interpolation'),
};
