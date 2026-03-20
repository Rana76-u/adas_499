import 'dart:async';
import 'dart:isolate';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'mobilenet_model.dart';

/// Runs [MobileNetModel.classify] on a background [Isolate] so that
/// inference never blocks the UI thread.
///
/// Usage:
/// ```dart
/// final isolate = MobileNetInferenceIsolate(
///   modelPath: 'assets/mobilenet.tflite',
///   labels:    mobileNetClasses,
/// );
/// await isolate.start();
///
/// // Later, on every camera frame:
/// final results = await isolate.infer(frame, topK: 3);
///
/// // On dispose:
/// await isolate.dispose();
/// ```
class MobileNetInferenceIsolate {
  final String modelPath;
  final List<String> labels;
  final int threads;
  final MobileNetDelegate delegate;
  final bool useStandardNorm;

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  int _nextId = 0;
  final Map<int, Completer<List<ClassificationResult>>> _pending = {};

  MobileNetInferenceIsolate({
    required this.modelPath,
    required this.labels,
    this.threads = 4,
    this.delegate = MobileNetDelegate.cpu,
    this.useStandardNorm = false,
  });

  // ── start() ───────────────────────────────────────────────────────────────

  Future<void> start() async {
    if (_isolate != null) return;

    // Asset channels are only available on the main isolate, so we load
    // the model bytes here and transfer them to the worker via
    // TransferableTypedData (zero-copy across the isolate boundary).
    final modelData  = await rootBundle.load(modelPath);
    final modelBytes = modelData.buffer.asUint8List();

    _isolate = await Isolate.spawn<_WorkerInit>(
      _workerMain,
      _WorkerInit(
        sendPort:        _receivePort.sendPort,
        modelBytes:      TransferableTypedData.fromList([modelBytes]),
        labels:          labels,
        threads:         threads,
        delegate:        delegate,
        useStandardNorm: useStandardNorm,
      ),
      debugName: 'mobilenet-inference',
    );

    _receivePort.listen((message) {
      // Worker sends its own SendPort first.
      if (message is SendPort) {
        _sendPort = message;
        return;
      }

      if (message is Map) {
        final int id   = message['id'] as int;
        final pending  = _pending.remove(id);
        if (pending == null) return;

        final err = message['error'];
        if (err != null) {
          pending.completeError(StateError(err.toString()));
          return;
        }

        final List<dynamic> list =
            (message['results'] as List<dynamic>? ?? const []);
        final results = list.map((e) {
          final m = e as Map;
          return ClassificationResult(
            label:       m['label']       as String,
            confidence:  (m['confidence'] as num).toDouble(),
            classIndex:  m['classIndex']  as int,
          );
        }).toList();

        pending.complete(results);
      }
    });

    // Wait for the worker's SendPort (max 3 s).
    final sw = Stopwatch()..start();
    while (_sendPort == null) {
      if (sw.elapsedMilliseconds > 3000) {
        throw TimeoutException('Timed out starting MobileNet inference isolate');
      }
      await Future<void>.delayed(const Duration(milliseconds: 5));
    }
  }

  // ── infer() ───────────────────────────────────────────────────────────────

  /// Sends a YUV420 camera frame to the worker and returns the top-[topK]
  /// classification results.
  Future<List<ClassificationResult>> infer(
    CameraFrameLike frame, {
    int topK = 5,
  }) async {
    final sendPort = _sendPort;
    if (sendPort == null) throw StateError('Isolate not started.');

    final id = _nextId++;
    final c  = Completer<List<ClassificationResult>>();
    _pending[id] = c;

    sendPort.send({
      'id':           id,
      'topK':         topK,
      'width':        frame.width,
      'height':       frame.height,
      'yRowStride':   frame.yRowStride,
      'uvRowStride':  frame.uvRowStride,
      'uvPixelStride':frame.uvPixelStride,
      'y': TransferableTypedData.fromList([frame.y]),
      'u': TransferableTypedData.fromList([frame.u]),
      'v': TransferableTypedData.fromList([frame.v]),
    });

    return c.future;
  }

  // ── dispose() ─────────────────────────────────────────────────────────────

  Future<void> dispose() async {
    for (final p in _pending.values) {
      p.completeError(StateError('Disposed'));
    }
    _pending.clear();
    _receivePort.close();
    _isolate?.kill(priority: Isolate.immediate);
    _isolate   = null;
    _sendPort  = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Isolate-friendly camera frame representation (YUV420)
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal, isolate-safe representation of a YUV420 camera frame.
/// Mirror of the one used by the YOLO pipeline so callers can share the same
/// conversion helper.
class CameraFrameLike {
  final int width;
  final int height;
  final int yRowStride;
  final int uvRowStride;
  final int uvPixelStride;
  final Uint8List y;
  final Uint8List u;
  final Uint8List v;

  const CameraFrameLike({
    required this.width,
    required this.height,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
    required this.y,
    required this.u,
    required this.v,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker isolate internals
// ─────────────────────────────────────────────────────────────────────────────

class _WorkerInit {
  final SendPort sendPort;
  final TransferableTypedData modelBytes;
  final List<String> labels;
  final int threads;
  final MobileNetDelegate delegate;
  final bool useStandardNorm;

  const _WorkerInit({
    required this.sendPort,
    required this.modelBytes,
    required this.labels,
    required this.threads,
    required this.delegate,
    required this.useStandardNorm,
  });
}

void _workerMain(_WorkerInit init) async {
  final options    = createMobileNetInterpreterOptions(
    threads:  init.threads,
    delegate: init.delegate,
  );
  final modelBytes = init.modelBytes.materialize().asUint8List();
  final interpreter = Interpreter.fromBuffer(modelBytes, options: options);
  final model = MobileNetModel.fromInterpreter(
    interpreter:     interpreter,
    labels:          init.labels,
    useStandardNorm: init.useStandardNorm,
  );

  final port = ReceivePort();
  init.sendPort.send(port.sendPort);

  await for (final message in port) {
    if (message is! Map) continue;
    final int id = message['id'] as int;
    try {
      final width          = message['width']          as int;
      final height         = message['height']         as int;
      final yRowStride     = message['yRowStride']     as int;
      final uvRowStride    = message['uvRowStride']    as int;
      final uvPixelStride  = message['uvPixelStride']  as int;
      final topK           = (message['topK'] as int?) ?? 5;

      final y = (message['y'] as TransferableTypedData).materialize().asUint8List();
      final u = (message['u'] as TransferableTypedData).materialize().asUint8List();
      final v = (message['v'] as TransferableTypedData).materialize().asUint8List();

      final rgb = _yuv420ToImage(
        width:          width,
        height:         height,
        y:              y,
        u:              u,
        v:              v,
        yRowStride:     yRowStride,
        uvRowStride:    uvRowStride,
        uvPixelStride:  uvPixelStride,
      );

      final results = await model.classify(rgb, topK: topK);

      init.sendPort.send({
        'id':      id,
        'results': results
            .map((r) => {
                  'label':      r.label,
                  'confidence': r.confidence,
                  'classIndex': r.classIndex,
                })
            .toList(),
      });
    } catch (e) {
      init.sendPort.send({'id': id, 'error': e.toString()});
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// YUV420 → RGB conversion  (identical to the YOLO pipeline)
// ─────────────────────────────────────────────────────────────────────────────

img.Image _yuv420ToImage({
  required int width,
  required int height,
  required Uint8List y,
  required Uint8List u,
  required Uint8List v,
  required int yRowStride,
  required int uvRowStride,
  required int uvPixelStride,
}) {
  final out = img.Image(width, height);
  for (int row = 0; row < height; row++) {
    final int uvRow       = row >> 1;
    final int yRowOffset  = row * yRowStride;
    final int uvRowOffset = uvRow * uvRowStride;
    for (int col = 0; col < width; col++) {
      final int yIndex  = yRowOffset + col;
      final int uvIndex = uvRowOffset + (col >> 1) * uvPixelStride;

      final int yVal = y[yIndex];
      final int uVal = u[uvIndex] - 128;
      final int vVal = v[uvIndex] - 128;

      final int r = (yVal + 1.402  * vVal).clamp(0, 255).toInt();
      final int g = (yVal - 0.344  * uVal - 0.714 * vVal).clamp(0, 255).toInt();
      final int b = (yVal + 1.772  * uVal).clamp(0, 255).toInt();
      out.setPixelRgba(col, row, r, g, b);
    }
  }
  return out;
}
