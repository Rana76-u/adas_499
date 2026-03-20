import 'dart:async';
import 'package:flutter/services.dart';
import 'yolo_model.dart';

/// Dart-side bridge to the native Android inference engine.
///
/// All heavy work (camera, YUV conversion, letterbox, TFLite, NMS) runs
/// entirely in Kotlin on a background thread.  This class only:
///   1. Tells native to load the model via [MethodChannel].
///   2. Tells native to start / stop the camera.
///   3. Exposes a [detectionStream] that emits [NativeFrame] objects
///      as fast as native can produce them.
class NativeDetectionBridge {
  static const _method = MethodChannel('com.example.adas_499/control');
  static const _events = EventChannel('com.example.adas_499/detections');

  Stream<NativeFrame>? _stream;

  // ── Model + camera lifecycle ───────────────────────────────────────────────

  /// Load the TFLite model on the native side.
  ///
  /// [modelPath] must be a Flutter asset path, e.g.
  ///   `'assets/models/yolo11n_int8.tflite'`
  /// [labels] is the ordered class-name list.
  /// [delegate] is `'gpu'` (default), `'nnapi'`, or `'cpu'`.
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

  /// Start the native CameraX pipeline.  Detections will flow through
  /// [detectionStream] immediately after this returns.
  Future<void> startCamera() async {
    await _method.invokeMethod<bool>('startCamera');
  }

  /// Stop the camera (inference stops too).
  Future<void> stopCamera() async {
    await _method.invokeMethod<bool>('stopCamera');
  }

  /// Release all native resources (interpreter, delegates, camera).
  Future<void> dispose() async {
    await _method.invokeMethod<bool>('dispose');
    _stream = null;
  }

  // ── Detection stream ───────────────────────────────────────────────────────

  /// Broadcast stream of [NativeFrame] objects.
  /// Each object carries the detections and timing for one camera frame.
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
  final int inferMs;
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
      inferMs: (map['inferMs'] as num?)?.toInt() ?? 0,
      fps: (map['fps'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
