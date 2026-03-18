import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Represents a single detected object.
/// [boundingBox] uses normalized [0,1] coordinates relative to the
/// ORIGINAL image dimensions (letterbox already removed).
class Detection {
  final String label;
  final double confidence;
  final Rect boundingBox;

  const Detection({
    required this.label,
    required this.confidence,
    required this.boundingBox,
  });

  @override
  String toString() =>
      'Detection(label: $label, '
      'confidence: ${confidence.toStringAsFixed(2)}, '
      'box: $boundingBox)';
}

class Rect {
  final double left;
  final double top;
  final double right;
  final double bottom;

  const Rect({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  double get width => right - left;
  double get height => bottom - top;
  double get centerX => (left + right) / 2;
  double get centerY => (top + bottom) / 2;

  @override
  String toString() =>
      'Rect(l:${left.toStringAsFixed(3)}, t:${top.toStringAsFixed(3)}, '
      'r:${right.toStringAsFixed(3)}, b:${bottom.toStringAsFixed(3)})';
}

// ── Internal raw detection (pixel coords in letterboxed space) ────────────────
class _RawDetection {
  final double left, top, right, bottom;
  final int classIndex;
  final double confidence;

  _RawDetection({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.classIndex,
    required this.confidence,
  });

  double get width => right - left;
  double get height => bottom - top;
}

// ── Letterbox result ──────────────────────────────────────────────────────────
class _LetterboxResult {
  final img.Image image; // 640×640 padded image
  final double scale; // how much the original was scaled down
  final int padLeft; // grey pixels added on left
  final int padTop; // grey pixels added on top

  const _LetterboxResult({
    required this.image,
    required this.scale,
    required this.padLeft,
    required this.padTop,
  });
}

/// YOLOv8 TFLite inference helper
class YoloModel {
  static const int inputSize = 640;
  static const double confThreshold = 0.45;
  static const double iouThreshold = 0.45;

  // Ultralytics letterbox fill colour — MUST match training preprocessing
  static const int _padValue = 114;

  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;
  List<String> get labels => _labels;

  // ── Load ──────────────────────────────────────────────────────────────────

  Future<void> load({
    required String modelPath,
    required List<String> labels,
  }) async {
    try {
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(modelPath, options: options);
      _labels = labels;
      _isLoaded = true;
      debugPrint(
        '✅ YOLO loaded. '
        'Input: ${_interpreter!.getInputTensors().first.shape}',
      );
    } catch (e) {
      debugPrint('❌ Failed to load model: $e');
      rethrow;
    }
  }

  // ── Public detect() ───────────────────────────────────────────────────────

  Future<List<Detection>> detect(img.Image image) async {
    if (!_isLoaded || _interpreter == null) {
      throw StateError('Model not loaded. Call load() first.');
    }

    // 1) Letterbox
    final lb = _letterbox(image);

    // 2) Flat float buffer
    final flat = _toFloat32Input(lb.image); // length = 640*640*3

    // 3) Get input shape from interpreter
    final inputTensor = _interpreter!.getInputTensor(0);
    final shape = inputTensor.shape; // expect [1, 640, 640, 3]
    final b = shape[0], h = shape[1], w = shape[2], c = shape[3];

    // 4) Build 4‑D nested list [b][h][w][c]
    int idx = 0;
    final input4d = List.generate(
      b,
      (_) => List.generate(
        h,
        (_) => List.generate(
          w,
          (_) => List.generate(c, (_) => flat[idx++].toDouble()),
        ),
      ),
    );

    // 5) Allocate output from actual tensor shape (you already do this)
    final outShape = _interpreter!.getOutputTensor(0).shape;
    final outputBuffer = List.generate(
      outShape[0],
      (_) => List.generate(outShape[1], (_) => List.filled(outShape[2], 0.0)),
    );

    // 6) Run
    _interpreter!.run(input4d, outputBuffer);

    // 7) Decode as before
    return _decodeOutput(
      outputBuffer[0],
      lb.scale,
      lb.padLeft,
      lb.padTop,
      image.width,
      image.height,
    );
  }

  // ── Step 1: Letterbox ─────────────────────────────────────────────────────
  //
  // Scales the image so the LONGER side fits inputSize, then centres it on a
  // grey (114, 114, 114) canvas. This matches Ultralytics' default letterbox
  // used during both training and the Colab export you ran.
  //
  //   scale   = uniform scale factor applied to original dims
  //   padLeft = grey columns added to the left  (right gets the remainder)
  //   padTop  = grey rows added to the top      (bottom gets the remainder)
  //
  _LetterboxResult _letterbox(img.Image src) {
    // Scale so the longer side == inputSize
    final scale = inputSize / max(src.width, src.height);
    final newW = (src.width * scale).round();
    final newH = (src.height * scale).round();

    // Resize with bilinear interpolation (same as cv2.INTER_LINEAR)
    final resized = img.copyResize(
      src,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.linear,
    );

    // Grey canvas filled with 114
    final canvas = img.Image(inputSize, inputSize);
    img.fill(canvas, img.getColor(_padValue, _padValue, _padValue));

    // Centre-paste: integer division keeps pad symmetric;
    // any odd pixel goes to the right / bottom (same as Ultralytics)
    final padLeft = (inputSize - newW) ~/ 2;
    final padTop = (inputSize - newH) ~/ 2;
    img.copyInto(canvas, resized, dstX: padLeft, dstY: padTop);

    return _LetterboxResult(
      image: canvas,
      scale: scale,
      padLeft: padLeft,
      padTop: padTop,
    );
  }

  // ── Step 2: Normalize → Float32 input tensor ──────────────────────────────
  //
  // Pixel values are divided by 255.0 → [0.0, 1.0].
  // Layout is NHWC: [batch=1, height=640, width=640, channels=3].
  // The image package's getPixel() already returns RGB on all platforms
  // so no BGR swap is needed (unlike raw OpenCV frames).
  //
  Float32List _toFloat32Input(img.Image image) {
    final buffer = Float32List(inputSize * inputSize * 3);
    int idx = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        buffer[idx++] = img.getRed(pixel) / 255.0;
        buffer[idx++] = img.getGreen(pixel) / 255.0;
        buffer[idx++] = img.getBlue(pixel) / 255.0;
      }
    }
    // reshape([1, 640, 640, 3]) is called at the call-site so tflite_flutter
    // can handle the tensor layout correctly
    return buffer;
  }

  // ── Step 5: Decode output + unscale boxes ─────────────────────────────────
  //
  // YOLOv8 raw output layout: [4 + numClasses, 8400]
  //   Row 0 : cx  (in letterboxed 640×640 pixel space)
  //   Row 1 : cy
  //   Row 2 : w
  //   Row 3 : h
  //   Row 4+: class scores (already sigmoid-activated by the TFLite export)
  //
  // Unscaling pipeline per box:
  //   1. Convert cx/cy/w/h  →  x1/y1/x2/y2  in letterboxed pixels
  //   2. Subtract pad offset  →  position within the resized (non-padded) image
  //   3. Divide by scale      →  original image pixel coords
  //   4. Divide by orig W/H   →  normalized [0, 1] for the painter
  //
  List<Detection> _decodeOutput(
    List<List<double>> output, // [4+numClasses, 8400]
    double scale,
    int padLeft,
    int padTop,
    int origW,
    int origH,
  ) {
    final numClasses = _labels.length;
    final numAnchors = output[0].length; // 8400
    final rawDets = <_RawDetection>[];

    for (int a = 0; a < numAnchors; a++) {
      // ── Find best class ──────────────────────────────────────────────────
      double bestScore = 0;
      int bestClass = 0;
      for (int c = 0; c < numClasses; c++) {
        final score = output[4 + c][a];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }
      if (bestScore < confThreshold) continue;

      // ── Box in letterboxed pixel space ───────────────────────────────────
      // ── Box in letterboxed pixel space (model gives normalized [0,1]) ──
      const int inputSize = YoloModel.inputSize; // 640

      final cx = output[0][a] * inputSize;
      final cy = output[1][a] * inputSize;
      final w = output[2][a] * inputSize;
      final h = output[3][a] * inputSize;

      final lbX1 = cx - w / 2;
      final lbY1 = cy - h / 2;
      final lbX2 = cx + w / 2;
      final lbY2 = cy + h / 2;

      // ── Remove letterbox padding, then undo scale ────────────────────────
      // Dividing by (scale * origW) combines the two steps:
      //   pixel_in_original = (letterboxed_pixel - pad) / scale
      //   normalized        = pixel_in_original / origDim
      final left = (lbX1 - padLeft) / (scale * origW);
      final top = (lbY1 - padTop) / (scale * origH);
      final right = (lbX2 - padLeft) / (scale * origW);
      final bottom = (lbY2 - padTop) / (scale * origH);

      rawDets.add(
        _RawDetection(
          left: left,
          top: top,
          right: right,
          bottom: bottom,
          classIndex: bestClass,
          confidence: bestScore,
        ),
      );
    }

    // ── NMS ─────────────────────────────────────────────────────────────────
    final nmsResults = _nonMaxSuppression(rawDets);

    // ── Clamp to [0,1] and wrap in Detection ─────────────────────────────────
    return nmsResults.map((d) {
      final label = d.classIndex < _labels.length
          ? _labels[d.classIndex]
          : 'class_${d.classIndex}';
      return Detection(
        label: label,
        confidence: d.confidence,
        boundingBox: Rect(
          left: d.left.clamp(0.0, 1.0),
          top: d.top.clamp(0.0, 1.0),
          right: d.right.clamp(0.0, 1.0),
          bottom: d.bottom.clamp(0.0, 1.0),
        ),
      );
    }).toList();
  }

  // ── NMS (greedy, per-class) ───────────────────────────────────────────────

  List<_RawDetection> _nonMaxSuppression(List<_RawDetection> detections) {
    if (detections.isEmpty) return [];

    final Map<int, List<_RawDetection>> byClass = {};
    for (final d in detections) {
      byClass.putIfAbsent(d.classIndex, () => []).add(d);
    }

    final results = <_RawDetection>[];
    for (final group in byClass.values) {
      group.sort((a, b) => b.confidence.compareTo(a.confidence));
      final kept = <_RawDetection>[];
      for (final det in group) {
        bool suppressed = false;
        for (final keep in kept) {
          if (_iou(det, keep) > iouThreshold) {
            suppressed = true;
            break;
          }
        }
        if (!suppressed) kept.add(det);
      }
      results.addAll(kept);
    }
    return results;
  }

  double _iou(_RawDetection a, _RawDetection b) {
    final iL = max(a.left, b.left);
    final iT = max(a.top, b.top);
    final iR = min(a.right, b.right);
    final iB = min(a.bottom, b.bottom);

    if (iR <= iL || iB <= iT) return 0.0;

    final inter = (iR - iL) * (iB - iT);
    final union = a.width * a.height + b.width * b.height - inter;
    return union == 0 ? 0.0 : inter / union;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isLoaded = false;
  }
}

// ── Label lists ───────────────────────────────────────────────────────────────

const List<String> customLabels = [
  'Crossroads',
  'Hospital Ahead',
  'Junction Ahead',
  'Mosque Ahead',
  'No Pedestrians',
  'No Vehicle Entry',
  'Pedestrians Crossing',
  'School Ahead',
  'Sharp Left Turn',
  'Sharp Right Turn',
  'Side Road On Left',
  'Side Road On Right',
  'Speed Breaker',
  'Speed Limit 20 km',
  'Speed Limit 40Km',
  'Speed Limit 80Km',
  'Traffic Merges From Left',
  'Traffic Merges From Right',
  'U Turn',
  'bicycle',
  'bus',
  'car',
  'cng',
  'motorcycle',
  'other-vehicle',
  'person',
  'rickshaw',
];
