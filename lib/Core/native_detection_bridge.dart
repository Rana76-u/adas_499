import 'dart:async';
import 'package:flutter/services.dart';
import 'yolo_model.dart';

/// Dart-side bridge to the native Android inference engine.
///
/// All heavy work (camera, YUV→RGB, letterbox, TFLite, NMS) runs entirely
/// in Kotlin on a background thread. This class only:
///   1. Tells native to load the model via [MethodChannel].
///   2. Tells native to start / stop the camera.
///   3. Returns the Flutter [textureId] from [startCamera] so the UI
///      can display a [Texture] widget backed by the native preview.
///   4. Exposes a [detectionStream] that emits [NativeFrame] objects.
///   5. Supports one-shot [runOnImage] for the still-image tab.
class NativeDetectionBridge {
  static const _method = MethodChannel('com.example.adas_499/control');
  static const _events = EventChannel('com.example.adas_499/detections');

  Stream<NativeFrame>? _stream;

  // ── Model + camera lifecycle ───────────────────────────────────────────────

  Future<void> loadModel({
    required String modelPath,
    required List<String> labels,
    String delegate = 'gpu',
  }) async {
    await _method.invokeMethod<bool>('loadModel', {
      'modelPath': modelPath,
      'labels':    labels,
      'delegate':  delegate,
    });
  }

  Future<int> startCamera() async {
    final id = await _method.invokeMethod<int>('startCamera');
    return id ?? -1;
  }

  Future<void> stopCamera() async {
    await _method.invokeMethod<bool>('stopCamera');
  }

  Future<void> dispose() async {
    await _method.invokeMethod<bool>('dispose');
    _stream = null;
  }

  // ── Still-image inference ──────────────────────────────────────────────────

  /// Run inference on a still image file (absolute path).
  /// Returns a list of [Detection] objects, or an empty list on error.
  Future<List<Detection>> runOnImage(String absolutePath) async {
    final raw = await _method.invokeMethod<List<dynamic>>(
        'runOnImage', {'path': absolutePath});
    if (raw == null) return const [];
    return raw
        .whereType<Map<Object?, Object?>>()
        .map(Detection.fromMap)
        .toList(growable: false);
  }

  // ── Detection stream ───────────────────────────────────────────────────────

  Stream<NativeFrame> get detectionStream {
    _stream ??= _events
        .receiveBroadcastStream()
        .map((event) => NativeFrame.fromMap(event as Map<Object?, Object?>));
    return _stream!;
  }
}

/// One frame worth of results sent from Kotlin to Dart.
class NativeFrame {
  final List<Detection> detections;
  final int    inferMs;
  final double fps;

  const NativeFrame({
    required this.detections,
    required this.inferMs,
    required this.fps,
  });

  factory NativeFrame.fromMap(Map<Object?, Object?> map) {
    final rawList = map['detections'] as List<Object?>? ?? [];
    final detections = rawList
        .whereType<Map<Object?, Object?>>()
        .map(Detection.fromMap)
        .toList(growable: false);

    return NativeFrame(
      detections: detections,
      inferMs:    (map['inferMs'] as num?)?.toInt()    ?? 0,
      fps:        (map['fps']     as num?)?.toDouble() ?? 0.0,
    );
  }
}
