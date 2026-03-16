// lib/yolo_detector.dart
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class YoloDetector {
  Interpreter? _interpreter;
  List<String>? _labels;

  static const int inputSize = 640; // YOLOv8 default
  static const double confidenceThreshold = 0.5;
  static const double iouThreshold = 0.5;

  Future<void> loadModel() async {
    try {
      // Load model (use same logical path as in pubspec.yaml)
      _interpreter = await Interpreter.fromAsset('assets/best_float16.tflite');

      // Query input tensor info (no manual resizing; model already has correct shape)
      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape;

      // Load class labels - FIXED VERSION
      _labels = await _loadLabels();

      print('✅ Model loaded successfully');
      print('Input shape: $inputShape');
      print('Input type: ${inputTensor.type}');
      print('Output shape: ${_interpreter!.getOutputTensor(0).shape}');
    } catch (e) {
      // Propagate error so callers know loading failed
      print('❌ Error loading model: $e');
      rethrow;
    }
  }

  Future<List<String>> _loadLabels() async {
    // Use rootBundle instead of DefaultAssetBundle
    final labelsData = await rootBundle.loadString('assets/labels.txt');
    return labelsData.split('\n').where((label) => label.isNotEmpty).toList();
  }

  List<Detection> detect(img.Image image) {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }

    // Get input tensor shape to determine format
    final inputTensor = _interpreter!.getInputTensor(0);
    final inputShape = inputTensor.shape;
    
    // Preprocess image based on expected input format
    // tflite_flutter accepts flat Float32List and reshapes internally
    final inputImage = _preprocessImage(image, inputShape);

    // Get actual output tensor shape
    final outputTensor = _interpreter!.getOutputTensor(0);
    final outputShape = outputTensor.shape;

    // YOLOv8 output shape: [1, 8400, num_classes+4] or [1, 84, 8400]
    // Create output buffer - tflite_flutter will reshape based on tensor shape
    final outputSize = outputShape.fold(1, (a, b) => a * b);
    final output = List.filled(outputSize, 0.0);

    try {
      _interpreter!.run(inputImage, output);
    } catch (e) {
      print('❌ Error running inference: $e');
      print('Input shape: $inputShape');
      print('Input length: ${inputImage.length}');
      print('Expected input size: ${inputShape.fold(1, (a, b) => a * b)}');
      print('Output shape: $outputShape');
      return [];
    }
    
    // Reshape output for processing into 2D [dim1][dim2] where
    // dim1 = outputShape[1], dim2 = outputShape[2]
    final reshapedOutput = _reshapeOutput(output, outputShape);

    // Post-process detections based on output format
    if (outputShape.length >= 3 && outputShape[1] > outputShape[2]) {
      // Format: [1, num_detections, num_values] - each row is a detection
      return _postProcessDetectionsRowMajor(
        reshapedOutput,
        image.width,
        image.height,
      );
    } else {
      // Format: [1, num_values, num_detections]
      return _postProcessDetectionsColumnMajor(
        reshapedOutput,
        image.width,
        image.height,
      );
    }
  }

  /// Reshape flat output buffer into 2D [dim1][dim2] using output tensor shape.
  /// Assumes outputShape is of the form [1, dim1, dim2].
  List<List<double>> _reshapeOutput(List<double> output, List<int> outputShape) {
    if (outputShape.length < 3) {
      throw Exception('Unexpected output shape: $outputShape');
    }

    final dim1 = outputShape[1];
    final dim2 = outputShape[2];

    if (dim1 * dim2 > output.length) {
      throw Exception(
        'Output buffer too small for shape $outputShape (length=${output.length})',
      );
    }

    final reshaped = List.generate(
      dim1,
      (_) => List<double>.filled(dim2, 0.0),
    );

    int index = 0;
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        reshaped[i][j] = output[index++];
      }
    }

    return reshaped;
  }

  Float32List _preprocessImage(img.Image image, List<int> inputShape) {
    // Resize to 640x640
    final resized = img.copyResize(image, width: inputSize, height: inputSize);

    // Determine format from input shape
    // [1, 640, 640, 3] = HWC format (Height, Width, Channel) - TFLite standard
    // [1, 3, 640, 640] = CHW format (Channel, Height, Width) - PyTorch format
    final isHWC = inputShape.length == 4 && inputShape[3] == 3;
    
    final input = Float32List(1 * 3 * inputSize * inputSize);

    if (isHWC) {
      // HWC format: [1, Height, Width, Channels]
      // Layout: [R, G, B, R, G, B, ...] for each pixel
      int pixelIndex = 0;
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          input[pixelIndex++] = img.getRed(pixel) / 255.0;
          input[pixelIndex++] = img.getGreen(pixel) / 255.0;
          input[pixelIndex++] = img.getBlue(pixel) / 255.0;
        }
      }
    } else {
      // CHW format: [1, Channels, Height, Width]
      // Layout: All R values, then all G values, then all B values
      int pixelIndex = 0;

      // Red channel
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          input[pixelIndex++] = img.getRed(pixel) / 255.0;
        }
      }

      // Green channel
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          input[pixelIndex++] = img.getGreen(pixel) / 255.0;
        }
      }

      // Blue channel
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          input[pixelIndex++] = img.getBlue(pixel) / 255.0;
        }
      }
    }

    return input;
  }

  List<Detection> _postProcessDetectionsRowMajor(
    List<List<double>> outputs,
    int imageWidth,
    int imageHeight,
  ) {
    List<Detection> detections = [];

    for (var detection in outputs) {
      if (detection.length < 5) continue;

      // YOLOv8 format: [x, y, w, h, class_scores...]
      // x, y, w, h are in normalized format (0-1)
      final x = detection[0];
      final y = detection[1];
      final w = detection[2];
      final h = detection[3];

      // Get class with highest score
      final classScores = detection.sublist(4);
      int classId = 0;
      double maxScore = classScores[0];

      for (int i = 1; i < classScores.length; i++) {
        if (classScores[i] > maxScore) {
          maxScore = classScores[i];
          classId = i;
        }
      }

      // Confidence is the max class score
      final conf = maxScore;
      if (conf < confidenceThreshold) continue;

      // Convert from normalized center format to pixel coordinates
      // YOLOv8 outputs center_x, center_y, width, height (all normalized)
      final centerX = x * imageWidth;
      final centerY = y * imageHeight;
      final boxWidth = w * imageWidth;
      final boxHeight = h * imageHeight;

      detections.add(
        Detection(
          bbox: BoundingBox(
            centerX - boxWidth / 2,
            centerY - boxHeight / 2,
            boxWidth,
            boxHeight,
          ),
          confidence: conf,
          classId: classId,
          className: _labels?[classId] ?? 'Unknown',
        ),
      );
    }

    // Apply Non-Maximum Suppression
    return _nonMaxSuppression(detections);
  }

  List<Detection> _postProcessDetectionsColumnMajor(
    List<List<double>> outputs,
    int imageWidth,
    int imageHeight,
  ) {
    List<Detection> detections = [];

    // outputs is [num_values, num_detections]
    final numDetections = outputs[0].length;

    for (int i = 0; i < numDetections; i++) {
      if (outputs.length < 5) continue;

      // Extract values for this detection
      final x = outputs[0][i];
      final y = outputs[1][i];
      final w = outputs[2][i];
      final h = outputs[3][i];

      // Get class with highest score
      int classId = 0;
      double maxScore = outputs[4][i];

      for (int j = 5; j < outputs.length; j++) {
        if (outputs[j][i] > maxScore) {
          maxScore = outputs[j][i];
          classId = j - 4;
        }
      }

      if (maxScore < confidenceThreshold) continue;

      // Convert from normalized to pixel coordinates
      final centerX = x * imageWidth;
      final centerY = y * imageHeight;
      final boxWidth = w * imageWidth;
      final boxHeight = h * imageHeight;

      detections.add(
        Detection(
          bbox: BoundingBox(
            centerX - boxWidth / 2,
            centerY - boxHeight / 2,
            boxWidth,
            boxHeight,
          ),
          confidence: maxScore,
          classId: classId,
          className: _labels?[classId] ?? 'Unknown',
        ),
      );
    }

    // Apply Non-Maximum Suppression
    return _nonMaxSuppression(detections);
  }

  List<Detection> _nonMaxSuppression(List<Detection> detections) {
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<Detection> result = [];

    while (detections.isNotEmpty) {
      final best = detections.removeAt(0);
      result.add(best);

      detections.removeWhere((detection) {
        return _iou(best.bbox, detection.bbox) > iouThreshold;
      });
    }

    return result;
  }

  double _iou(BoundingBox a, BoundingBox b) {
    final intersectionX = math.min(a.x + a.w, b.x + b.w) - math.max(a.x, b.x);
    final intersectionY = math.min(a.y + a.h, b.y + b.h) - math.max(a.y, b.y);

    if (intersectionX <= 0 || intersectionY <= 0) return 0;

    final intersectionArea = intersectionX * intersectionY;
    final unionArea = (a.w * a.h) + (b.w * b.h) - intersectionArea;

    return intersectionArea / unionArea;
  }

  void dispose() {
    _interpreter?.close();
  }
}

class Detection {
  final BoundingBox bbox;
  final double confidence;
  final int classId;
  final String className;

  Detection({
    required this.bbox,
    required this.confidence,
    required this.classId,
    required this.className,
  });
}

class BoundingBox {
  final double x, y, w, h;

  BoundingBox(this.x, this.y, this.w, this.h);

  double get centerX => x + w / 2;
  double get centerY => y + h / 2;
}
