import 'dart:io' show Platform;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Result of a MobileNet classification pass.
class ClassificationResult {
  final String label;
  final double confidence;
  final int classIndex;

  const ClassificationResult({
    required this.label,
    required this.confidence,
    required this.classIndex,
  });

  @override
  String toString() =>
      'ClassificationResult(label: $label, '
      'confidence: ${confidence.toStringAsFixed(4)}, '
      'classIndex: $classIndex)';
}

// ── Delegate / accelerator selection ─────────────────────────────────────────

enum MobileNetDelegate {
  cpu,
  nnapi,
  gpu,
}

InterpreterOptions createMobileNetInterpreterOptions({
  int threads = 4,
  MobileNetDelegate delegate = MobileNetDelegate.cpu,
}) {
  final options = InterpreterOptions()..threads = threads;

  if (Platform.isAndroid) {
    switch (delegate) {
      case MobileNetDelegate.nnapi:
        options.useNnApiForAndroid = true;
        break;
      case MobileNetDelegate.gpu:
        options.addDelegate(GpuDelegateV2());
        break;
      case MobileNetDelegate.cpu:
        break;
    }
  }

  return options;
}

/// MobileNetV4-Conv-Small TFLite inference helper.
///
/// Input  : 224×224 RGB image, normalised to [0.0, 1.0] (float32)
///          or quantized uint8 via the tensor's scale/zero-point.
/// Output : [1, numClasses] probability vector (softmax already applied
///          by the TFLite model export).
///
/// Usage:
/// ```dart
/// final model = MobileNetModel();
/// await model.load(modelPath: 'assets/mobilenet.tflite', labels: mobileNetClasses);
/// final results = await model.classify(image);          // top-5 by default
/// ```
class MobileNetModel {
  /// Input resolution expected by MobileNetV4-Conv-Small.
  static const int inputSize = 224;

  /// Only results with confidence ≥ this threshold are returned.
  static const double confThreshold = 0.01;

  /// Standard ImageNet mean/std used by MobileNet variants.
  /// Set [useStandardNorm] = true to apply them; leave false to use
  /// simple [0,1] normalisation (common for TFLite INT8 exports).
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std  = [0.229, 0.224, 0.225];

  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;
  String? _modelPath;
  MobileNetDelegate _delegate = MobileNetDelegate.cpu;

  /// When true, applies ImageNet channel-wise mean/std normalisation
  /// on top of the [0,1] rescale.  Most TFLite exports do NOT need this
  /// (the normalisation is baked in), so it defaults to false.
  final bool useStandardNorm;

  MobileNetModel({this.useStandardNorm = false});

  MobileNetModel.fromInterpreter({
    required Interpreter interpreter,
    required List<String> labels,
    this.useStandardNorm = false,
  }) {
    _interpreter = interpreter;
    _labels = labels;
    _isLoaded = true;
  }

  bool get isLoaded => _isLoaded;
  List<String> get labels => _labels;
  String? get modelPath => _modelPath;
  MobileNetDelegate get delegate => _delegate;

  // ── Load ──────────────────────────────────────────────────────────────────

  Future<void> load({
    required String modelPath,
    required List<String> labels,
    MobileNetDelegate delegate = MobileNetDelegate.cpu,
    int threads = 4,
  }) async {
    try {
      final options = createMobileNetInterpreterOptions(
        threads: threads,
        delegate: delegate,
      );
      _interpreter = await Interpreter.fromAsset(modelPath, options: options);
      _labels = labels;
      _modelPath = modelPath;
      _delegate = delegate;
      _isLoaded = true;
      debugPrint(
        '✅ MobileNet loaded. '
        'Input: ${_interpreter!.getInputTensors().first.shape}  '
        'Output: ${_interpreter!.getOutputTensors().first.shape}',
      );
    } catch (e) {
      debugPrint('❌ Failed to load MobileNet model: $e');
      rethrow;
    }
  }

  // ── Public classify() ─────────────────────────────────────────────────────

  /// Classifies [image] and returns the top [topK] predictions sorted by
  /// confidence (highest first).  Only results ≥ [confThreshold] are included.
  Future<List<ClassificationResult>> classify(
    img.Image image, {
    int topK = 5,
  }) async {
    if (!_isLoaded || _interpreter == null) {
      throw StateError('Model not loaded. Call load() first.');
    }

    // 1) Resize to 224×224
    final resized = _resizeInput(image);

    // 2) Build typed input tensor
    final inputTensor = _interpreter!.getInputTensor(0);
    final shape = inputTensor.shape; // [1, 224, 224, 3]
    final b = shape[0], h = shape[1], w = shape[2], c = shape[3];

    final Object input = switch (inputTensor.type) {
      TensorType.float32 =>
        _toFloat32Input(resized).reshape([b, h, w, c]),
      TensorType.uint8 || TensorType.int8 =>
        _toQuantizedInput(resized, inputTensor).reshape([b, h, w, c]),
      _ => throw UnsupportedError(
          'Unsupported input tensor type: ${inputTensor.type}',
        ),
    };

    // 3) Allocate output buffer from actual tensor shape
    final outShape = _interpreter!.getOutputTensor(0).shape; // [1, numClasses]
    final outputBuffer = List.generate(
      outShape[0],
      (_) => List<double>.filled(outShape[1], 0.0),
    );

    // 4) Run inference
    _interpreter!.run(input, outputBuffer);

    // 5) Decode probabilities
    return _decodeOutput(outputBuffer[0], topK);
  }

  // ── Step 1: Resize to 224×224 ─────────────────────────────────────────────
  //
  // MobileNet classification does NOT need letterboxing — the model is trained
  // on square crops, so a simple bilinear resize is correct and sufficient.

  img.Image _resizeInput(img.Image src) {
    if (src.width == inputSize && src.height == inputSize) return src;
    return img.copyResize(
      src,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );
  }

  // ── Step 2a: Float32 input ────────────────────────────────────────────────
  //
  // Pixels → [0.0, 1.0].  Optionally applies ImageNet channel normalisation.

  Float32List _toFloat32Input(img.Image image) {
    final buffer = Float32List(inputSize * inputSize * 3);
    int idx = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        double r = img.getRed(pixel)   / 255.0;
        double g = img.getGreen(pixel) / 255.0;
        double b = img.getBlue(pixel)  / 255.0;

        if (useStandardNorm) {
          r = (r - _mean[0]) / _std[0];
          g = (g - _mean[1]) / _std[1];
          b = (b - _mean[2]) / _std[2];
        }

        buffer[idx++] = r;
        buffer[idx++] = g;
        buffer[idx++] = b;
      }
    }
    return buffer;
  }

  // ── Step 2b: Quantized (uint8 / int8) input ───────────────────────────────

  Uint8List _toQuantizedInput(img.Image image, Tensor inputTensor) {
    final params = inputTensor.params;
    final scale     = params.scale == 0 ? 1.0 : params.scale;
    final zeroPoint = params.zeroPoint;

    final buffer = Uint8List(inputSize * inputSize * 3);
    int idx = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        buffer[idx++] = _quantize(img.getRed(pixel),   scale, zeroPoint);
        buffer[idx++] = _quantize(img.getGreen(pixel), scale, zeroPoint);
        buffer[idx++] = _quantize(img.getBlue(pixel),  scale, zeroPoint);
      }
    }
    return buffer;
  }

  int _quantize(int channel0to255, double scale, int zeroPoint) {
    final normalized = channel0to255 / 255.0;
    return (normalized / scale + zeroPoint).round().clamp(0, 255);
  }

  // ── Step 3: Decode output probabilities ───────────────────────────────────

  List<ClassificationResult> _decodeOutput(List<double> probs, int topK) {
    // Collect all classes above the confidence threshold.
    final results = <ClassificationResult>[];
    for (int i = 0; i < probs.length; i++) {
      if (probs[i] >= confThreshold) {
        results.add(ClassificationResult(
          label: i < _labels.length ? _labels[i] : 'class_$i',
          confidence: probs[i],
          classIndex: i,
        ));
      }
    }

    // Sort descending by confidence and take top-K.
    results.sort((a, b) => b.confidence.compareTo(a.confidence));
    return results.take(topK).toList();
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isLoaded = false;
  }
}
