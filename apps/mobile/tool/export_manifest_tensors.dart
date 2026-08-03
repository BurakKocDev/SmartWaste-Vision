import 'dart:convert';
import 'dart:io';

import 'package:smartwaste_vision/src/inference/image_preprocessor.dart';
import 'package:smartwaste_vision/src/inference/model_contract.dart';

Future<void> main(List<String> arguments) async {
  if (arguments.length < 5 || arguments.length > 6) {
    stderr.writeln(
      'Usage: dart run tool/export_manifest_tensors.dart '
      '<manifest.csv> <data-root> <model-contract.json> <split> <output.f32> '
      '[pillow|linear|average|cubic]',
    );
    exitCode = 64;
    return;
  }
  final manifestPath = arguments[0];
  final dataRoot = arguments[1];
  final contract = ModelContract.fromJsonString(
    await File(arguments[2]).readAsString(),
  );
  final split = arguments[3];
  final outputPath = arguments[4];
  final resizeMode = _resizeMode(
    arguments.length == 6 ? arguments[5] : 'pillow',
  );
  final records = await _readManifest(manifestPath);
  final selected = <Map<String, String>>[];
  for (final label in contract.labels) {
    final candidates =
        records
            .where(
              (row) => row['split'] == split && row['target_label'] == label,
            )
            .toList()
          ..sort(
            (left, right) => left['relative_path']!.toLowerCase().compareTo(
              right['relative_path']!.toLowerCase(),
            ),
          );
    selected.addAll(candidates);
  }
  if (selected.isEmpty) {
    throw StateError('No manifest rows found for split: $split');
  }

  final sink = File(outputPath).openWrite();
  try {
    for (var index = 0; index < selected.length; index++) {
      final path = _joinPath(dataRoot, selected[index]['relative_path']!);
      final tensor = preprocessImage(
        await File(path).readAsBytes(),
        contract,
        resizeMode: resizeMode,
      );
      sink.add(tensor.buffer.asUint8List());
      if ((index + 1) % 50 == 0 || index + 1 == selected.length) {
        stdout.writeln('Processed ${index + 1}/${selected.length}');
      }
    }
    await sink.flush();
  } finally {
    await sink.close();
  }
  stdout.writeln('${selected.length} tensors -> $outputPath');
}

MobileResizeMode _resizeMode(String name) => switch (name) {
  'pillow' => MobileResizeMode.pillowBilinearAntialias,
  'linear' => MobileResizeMode.linear,
  'average' => MobileResizeMode.average,
  'cubic' => MobileResizeMode.cubic,
  _ => throw ArgumentError.value(name, 'interpolation'),
};

Future<List<Map<String, String>>> _readManifest(String path) async {
  final lines = const LineSplitter().convert(await File(path).readAsString());
  if (lines.isEmpty) {
    throw const FormatException('Manifest is empty');
  }
  final headers = _parseCsvRecord(lines.first);
  return lines
      .skip(1)
      .where((line) => line.isNotEmpty)
      .map((line) {
        final values = _parseCsvRecord(line);
        if (values.length != headers.length) {
          throw FormatException('Malformed CSV record: $line');
        }
        return Map<String, String>.fromIterables(headers, values);
      })
      .toList(growable: false);
}

List<String> _parseCsvRecord(String record) {
  final values = <String>[];
  final current = StringBuffer();
  var quoted = false;
  for (var index = 0; index < record.length; index++) {
    final character = record[index];
    if (character == '"') {
      if (quoted && index + 1 < record.length && record[index + 1] == '"') {
        current.write('"');
        index++;
      } else {
        quoted = !quoted;
      }
    } else if (character == ',' && !quoted) {
      values.add(current.toString());
      current.clear();
    } else {
      current.write(character);
    }
  }
  if (quoted) {
    throw FormatException('Unterminated quoted CSV field: $record');
  }
  values.add(current.toString());
  return values;
}

String _joinPath(String root, String relative) {
  final separator = Platform.pathSeparator;
  final normalized = relative
      .replaceAll('/', separator)
      .replaceAll('\\', separator);
  return '$root$separator$normalized';
}
