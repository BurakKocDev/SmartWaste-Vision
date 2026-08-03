import 'dart:typed_data';

import 'package:image/image.dart' as image_lib;

import 'model_contract.dart';

enum MobileResizeMode { pillowBilinearAntialias, linear, average, cubic }

Float32List preprocessImage(
  Uint8List encodedImage,
  ModelContract contract, {
  MobileResizeMode resizeMode = MobileResizeMode.pillowBilinearAntialias,
}) {
  image_lib.Image? decoded;
  try {
    decoded = image_lib.decodeImage(encodedImage);
  } catch (_) {
    throw const FormatException('Fotoğraf okunamadı. JPEG veya PNG kullan.');
  }
  if (decoded == null) {
    throw const FormatException('Fotoğraf okunamadı. JPEG veya PNG kullan.');
  }
  final oriented = image_lib.bakeOrientation(decoded);
  final cropSize = contract.cropSize;
  final pixels = switch (resizeMode) {
    MobileResizeMode.pillowBilinearAntialias => _pillowResizeAndCenterCrop(
      oriented,
      contract.resizeShorterSide,
      cropSize,
    ),
    _ => _packageResizeAndCenterCrop(
      oriented,
      contract.resizeShorterSide,
      cropSize,
      _packageInterpolation(resizeMode),
    ),
  };

  final planeSize = cropSize * cropSize;
  final tensor = Float32List(3 * planeSize);
  for (var spatialIndex = 0; spatialIndex < planeSize; spatialIndex++) {
    final pixelIndex = spatialIndex * 3;
    tensor[spatialIndex] =
        (pixels[pixelIndex] / 255.0 - contract.mean[0]) /
        contract.standardDeviation[0];
    tensor[planeSize + spatialIndex] =
        (pixels[pixelIndex + 1] / 255.0 - contract.mean[1]) /
        contract.standardDeviation[1];
    tensor[2 * planeSize + spatialIndex] =
        (pixels[pixelIndex + 2] / 255.0 - contract.mean[2]) /
        contract.standardDeviation[2];
  }
  return tensor;
}

Uint8List _packageResizeAndCenterCrop(
  image_lib.Image source,
  int shorterSide,
  int cropSize,
  image_lib.Interpolation interpolation,
) {
  final (width, height) = _resizedDimensions(source, shorterSide);
  final resized = image_lib.copyResize(
    source,
    width: width,
    height: height,
    interpolation: interpolation,
  );
  if (resized.width < cropSize || resized.height < cropSize) {
    throw StateError('Resized image is smaller than the requested center crop');
  }
  final cropped = image_lib.copyCrop(
    resized,
    x: _pythonHalfDifference(resized.width - cropSize),
    y: _pythonHalfDifference(resized.height - cropSize),
    width: cropSize,
    height: cropSize,
  );
  return cropped.getBytes(order: image_lib.ChannelOrder.rgb);
}

Uint8List _pillowResizeAndCenterCrop(
  image_lib.Image source,
  int shorterSide,
  int cropSize,
) {
  final (outputWidth, outputHeight) = _resizedDimensions(source, shorterSide);
  final cropLeft = _pythonHalfDifference(outputWidth - cropSize);
  final cropTop = _pythonHalfDifference(outputHeight - cropSize);
  final horizontalWeights = _axisWeights(
    inputSize: source.width,
    outputSize: outputWidth,
    outputStart: cropLeft,
    outputCount: cropSize,
  );
  final verticalWeights = _axisWeights(
    inputSize: source.height,
    outputSize: outputHeight,
    outputStart: cropTop,
    outputCount: cropSize,
  );
  final sourceRgb = source.getBytes(order: image_lib.ChannelOrder.rgb);
  final horizontal = Uint8List(source.height * cropSize * 3);
  for (var y = 0; y < source.height; y++) {
    final sourceRow = y * source.width * 3;
    final destinationRow = y * cropSize * 3;
    for (var x = 0; x < cropSize; x++) {
      final weights = horizontalWeights[x];
      var red = _roundingOffset;
      var green = _roundingOffset;
      var blue = _roundingOffset;
      for (var tap = 0; tap < weights.values.length; tap++) {
        final sourceIndex = sourceRow + (weights.start + tap) * 3;
        final weight = weights.values[tap];
        red += sourceRgb[sourceIndex] * weight;
        green += sourceRgb[sourceIndex + 1] * weight;
        blue += sourceRgb[sourceIndex + 2] * weight;
      }
      final destinationIndex = destinationRow + x * 3;
      horizontal[destinationIndex] = _clip8(red >> _precisionBits);
      horizontal[destinationIndex + 1] = _clip8(green >> _precisionBits);
      horizontal[destinationIndex + 2] = _clip8(blue >> _precisionBits);
    }
  }

  final output = Uint8List(cropSize * cropSize * 3);
  for (var y = 0; y < cropSize; y++) {
    final weights = verticalWeights[y];
    final destinationRow = y * cropSize * 3;
    for (var x = 0; x < cropSize; x++) {
      var red = _roundingOffset;
      var green = _roundingOffset;
      var blue = _roundingOffset;
      for (var tap = 0; tap < weights.values.length; tap++) {
        final sourceIndex = ((weights.start + tap) * cropSize + x) * 3;
        final weight = weights.values[tap];
        red += horizontal[sourceIndex] * weight;
        green += horizontal[sourceIndex + 1] * weight;
        blue += horizontal[sourceIndex + 2] * weight;
      }
      final destinationIndex = destinationRow + x * 3;
      output[destinationIndex] = _clip8(red >> _precisionBits);
      output[destinationIndex + 1] = _clip8(green >> _precisionBits);
      output[destinationIndex + 2] = _clip8(blue >> _precisionBits);
    }
  }
  return output;
}

(int, int) _resizedDimensions(image_lib.Image source, int shorterSide) {
  if (source.width <= 0 || source.height <= 0) {
    throw StateError('Image dimensions must be positive');
  }
  final int width;
  final int height;
  if (source.width <= source.height) {
    width = shorterSide;
    height = (source.height * shorterSide / source.width).toInt();
  } else {
    height = shorterSide;
    width = (source.width * shorterSide / source.height).toInt();
  }
  return (width, height);
}

image_lib.Interpolation _packageInterpolation(MobileResizeMode mode) =>
    switch (mode) {
      MobileResizeMode.linear => image_lib.Interpolation.linear,
      MobileResizeMode.average => image_lib.Interpolation.average,
      MobileResizeMode.cubic => image_lib.Interpolation.cubic,
      MobileResizeMode.pillowBilinearAntialias => throw StateError(
        'Pillow mode does not use package interpolation',
      ),
    };

int _pythonHalfDifference(int difference) {
  if (difference < 0) {
    throw StateError('Center crop exceeds resized image dimensions');
  }
  final lower = difference ~/ 2;
  if (difference.isEven || lower.isEven) {
    return lower;
  }
  return lower + 1;
}

const _precisionBits = 22;
const _roundingOffset = 1 << (_precisionBits - 1);
const _coefficientScale = 1 << _precisionBits;

class _Weights {
  const _Weights(this.start, this.values);

  final int start;
  final Int32List values;
}

List<_Weights> _axisWeights({
  required int inputSize,
  required int outputSize,
  required int outputStart,
  required int outputCount,
}) {
  final scale = inputSize / outputSize;
  final filterScale = scale < 1 ? 1.0 : scale;
  final support = filterScale;
  final inverseFilterScale = 1.0 / filterScale;
  return List<_Weights>.generate(outputCount, (localIndex) {
    final outputIndex = outputStart + localIndex;
    final center = (outputIndex + 0.5) * scale;
    var minimum = (center - support + 0.5).toInt();
    minimum = minimum.clamp(0, inputSize);
    var maximum = (center + support + 0.5).toInt();
    maximum = maximum.clamp(0, inputSize);
    final floating = List<double>.generate(maximum - minimum, (tap) {
      final distance = ((tap + minimum - center + 0.5) * inverseFilterScale)
          .abs();
      return distance < 1.0 ? 1.0 - distance : 0.0;
    }, growable: false);
    final total = floating.fold<double>(0, (sum, value) => sum + value);
    if (total == 0) {
      throw StateError('Pillow bilinear coefficient sum is zero');
    }
    final fixed = Int32List(floating.length);
    for (var tap = 0; tap < floating.length; tap++) {
      fixed[tap] = (floating[tap] / total * _coefficientScale + 0.5).toInt();
    }
    return _Weights(minimum, fixed);
  }, growable: false);
}

int _clip8(int value) {
  if (value < 0) return 0;
  if (value > 255) return 255;
  return value;
}
