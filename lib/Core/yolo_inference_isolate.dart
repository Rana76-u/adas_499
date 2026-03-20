import 'dart:async';
import 'dart:isolate';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'yolo_model.dart';

class YoloInferenceIsolate {
  final String modelPath;
  final List<String> labels;
  final int threads;
  final YoloDelegate delegate;

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  int _nextId = 0;
  final Map<int, Completer<List<Detection>>> _pending = {};

  YoloInferenceIsolate({
    required this.modelPath,
    required this.labels,
    this.threads = 4,
    this.delegate = YoloDelegate.cpu,
  });

  Future<void> start() async {
    if (_isolate != null) return;

    // Load model bytes on the main isolate (asset channels require bindings).
    final modelData = await rootBundle.load(modelPath);
    final modelBytes = modelData.buffer.asUint8List();

    _isolate = await Isolate.spawn<_WorkerInit>(
      _workerMain,
      _WorkerInit(
        sendPort: _receivePort.sendPort,
        modelBytes: TransferableTypedData.fromList([modelBytes]),
        labels: labels,
        threads: threads,
        delegate: delegate,
      ),
      debugName: 'yolo-inference',
    );

    _receivePort.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
        return;
      }
      if (message is Map) {
        final int id = message['id'] as int;
        final pending = _pending.remove(id);
        if (pending == null) return;

        final err = message['error'];
        if (err != null) {
          pending.completeError(StateError(err.toString()));
          return;
        }

        final List<dynamic> list = (message['detections'] as List<dynamic>? ?? const []);
        final dets = list.map((e) {
          final m = e as Map;
          return Detection(
            label: m['label'] as String,
            confidence: (m['confidence'] as num).toDouble(),
            boundingBox: Rect(
              left: (m['left'] as num).toDouble(),
              top: (m['top'] as num).toDouble(),
              right: (m['right'] as num).toDouble(),
              bottom: (m['bottom'] as num).toDouble(),
            ),
          );
        }).toList();

        pending.complete(dets);
      }
    });

    // Wait until worker sends back its SendPort.
    final sw = Stopwatch()..start();
    while (_sendPort == null) {
      if (sw.elapsedMilliseconds > 3000) {
        throw TimeoutException('Timed out starting inference isolate');
      }
      await Future<void>.delayed(const Duration(milliseconds: 5));
    }
  }

  Future<List<Detection>> infer(CameraImageLike frame) async {
    final sendPort = _sendPort;
    if (sendPort == null) throw StateError('Isolate not started.');

    final id = _nextId++;
    final c = Completer<List<Detection>>();
    _pending[id] = c;

    sendPort.send({
      'id': id,
      'width': frame.width,
      'height': frame.height,
      'yRowStride': frame.yRowStride,
      'uvRowStride': frame.uvRowStride,
      'uvPixelStride': frame.uvPixelStride,
      'y': TransferableTypedData.fromList([frame.y]),
      'u': TransferableTypedData.fromList([frame.u]),
      'v': TransferableTypedData.fromList([frame.v]),
    });

    return c.future;
  }

  Future<void> dispose() async {
    for (final p in _pending.values) {
      p.completeError(StateError('Disposed'));
    }
    _pending.clear();
    _receivePort.close();
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _sendPort = null;
  }
}

/// Minimal, isolate-friendly representation of a YUV420 camera frame.
class CameraImageLike {
  final int width;
  final int height;
  final int yRowStride;
  final int uvRowStride;
  final int uvPixelStride;
  final Uint8List y;
  final Uint8List u;
  final Uint8List v;

  const CameraImageLike({
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

class _WorkerInit {
  final SendPort sendPort;
  final TransferableTypedData modelBytes;
  final List<String> labels;
  final int threads;
  final YoloDelegate delegate;

  const _WorkerInit({
    required this.sendPort,
    required this.modelBytes,
    required this.labels,
    required this.threads,
    required this.delegate,
  });
}

void _workerMain(_WorkerInit init) async {
  final options = createYoloInterpreterOptions(
    threads: init.threads,
    delegate: init.delegate,
  );
  final modelBytes = init.modelBytes.materialize().asUint8List();
  final interpreter = Interpreter.fromBuffer(modelBytes, options: options);
  final model = YoloModel.fromInterpreter(interpreter: interpreter, labels: init.labels);

  final port = ReceivePort();
  init.sendPort.send(port.sendPort);

  await for (final message in port) {
    if (message is! Map) continue;
    final int id = message['id'] as int;
    try {
      final width = message['width'] as int;
      final height = message['height'] as int;
      final yRowStride = message['yRowStride'] as int;
      final uvRowStride = message['uvRowStride'] as int;
      final uvPixelStride = message['uvPixelStride'] as int;

      final y = (message['y'] as TransferableTypedData).materialize().asUint8List();
      final u = (message['u'] as TransferableTypedData).materialize().asUint8List();
      final v = (message['v'] as TransferableTypedData).materialize().asUint8List();

      final rgb = _yuv420ToImage(
        width: width,
        height: height,
        y: y,
        u: u,
        v: v,
        yRowStride: yRowStride,
        uvRowStride: uvRowStride,
        uvPixelStride: uvPixelStride,
      );

      final dets = await model.detect(rgb);
      init.sendPort.send({
        'id': id,
        'detections': dets
            .map((d) => {
                  'label': d.label,
                  'confidence': d.confidence,
                  'left': d.boundingBox.left,
                  'top': d.boundingBox.top,
                  'right': d.boundingBox.right,
                  'bottom': d.boundingBox.bottom,
                })
            .toList(),
      });
    } catch (e) {
      init.sendPort.send({'id': id, 'error': e.toString()});
    }
  }
}

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
    final int uvRow = (row >> 1);
    final int yRowOffset = row * yRowStride;
    final int uvRowOffset = uvRow * uvRowStride;
    for (int col = 0; col < width; col++) {
      final int yIndex = yRowOffset + col;
      final int uvIndex = uvRowOffset + (col >> 1) * uvPixelStride;

      final int yVal = y[yIndex];
      final int uVal = u[uvIndex] - 128;
      final int vVal = v[uvIndex] - 128;

      final int r = (yVal + 1.402 * vVal).clamp(0, 255).toInt();
      final int g = (yVal - 0.344 * uVal - 0.714 * vVal).clamp(0, 255).toInt();
      final int b = (yVal + 1.772 * uVal).clamp(0, 255).toInt();
      out.setPixelRgba(col, row, r, g, b);
    }
  }
  return out;
}

