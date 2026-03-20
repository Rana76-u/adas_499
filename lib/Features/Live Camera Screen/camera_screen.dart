import 'package:camera/camera.dart';
import 'dart:typed_data';
import 'package:flutter/material.dart';

import '../../Core/detection_painter.dart';
import '../../Core/yolo_inference_isolate.dart';
import '../../Core/yolo_model.dart';

class LiveCameraScreen extends StatefulWidget {
  final YoloModel model;
  const LiveCameraScreen({super.key, required this.model});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  static const double _liveConfidenceThreshold = 0.70;

  CameraController? _ctrl;
  List<Detection> _detections = [];
  bool _processing = false;
  bool _ready = false;
  String? _error;
  YoloInferenceIsolate? _infer;

  // Detected at runtime — true when the camera delivers NV21 (2-plane)
  // instead of YUV420 (3-plane).
  bool _isNv21 = false;

  int _frameCount = 0;
  double _fps = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  // ── Camera init with NV21 → YUV420 fallback ───────────────────────────────
  //
  // Some Android devices (e.g. OnePlus) reject YUV420 streams at medium
  // resolution with "Function not implemented (-38)".  We try NV21 first
  // (2-plane interleaved VU) then fall back to low resolution yuv420.

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      if (mounted) setState(() => _error = 'No cameras found on this device.');
      return;
    }

    // Ordered list of (format, resolution) combinations to try.
    final attempts = [
      (ImageFormatGroup.nv21, ResolutionPreset.medium),
      (ImageFormatGroup.yuv420, ResolutionPreset.low),
      (ImageFormatGroup.nv21, ResolutionPreset.low),
    ];

    for (final (fmt, res) in attempts) {
      _ctrl?.dispose();
      _ctrl = CameraController(
        cameras.first,
        res,
        enableAudio: false,
        imageFormatGroup: fmt,
      );

      try {
        await _ctrl!.initialize();
        _isNv21 = (fmt == ImageFormatGroup.nv21);
        debugPrint(
          '✅ Camera opened — format: $fmt  resolution: $res  nv21: $_isNv21',
        );
        break; // success
      } catch (e) {
        debugPrint('⚠️  Camera init failed ($fmt / $res): $e — trying next…');
        if ((fmt, res) == attempts.last) {
          if (mounted) setState(() => _error = 'Camera error: $e');
          return;
        }
      }
    }

    final modelPath = widget.model.modelPath;
    if (modelPath == null) {
      if (mounted)
        setState(() => _error = 'Model path unavailable (model not loaded?).');
      return;
    }

    try {
      _infer = YoloInferenceIsolate(
        modelPath: modelPath,
        labels: widget.model.labels,
        delegate: YoloDelegate.nnapi,
      );
      await _infer!.start();
      await _ctrl!.startImageStream(_onFrame);
      if (mounted) setState(() => _ready = true);
    } catch (e) {
      if (mounted) setState(() => _error = 'Isolate/stream error: $e');
    }
  }

  int _frameSkip = 0;

  void _onFrame(CameraImage frame) {
    if ((++_frameSkip % 3) != 0) return;
    if (_processing) return;
    _processing = true;

    _frameCount++;
    final now = DateTime.now();
    final ms = now.difference(_lastFpsUpdate).inMilliseconds;
    if (ms >= 1000) {
      if (mounted) setState(() => _fps = _frameCount * 1000 / ms);
      _frameCount = 0;
      _lastFpsUpdate = now;
    }

    _processFrame(frame).whenComplete(() => _processing = false);
  }

  Future<void> _processFrame(CameraImage frame) async {
    try {
      final infer = _infer;
      if (infer == null) return;

      // Validate frame planes
      debugPrint('Frame planes count: ${frame.planes.length}');
      if (frame.planes.isEmpty) {
        debugPrint('No image planes available in frame');
        return;
      }

      final Uint8List y;
      final Uint8List u;
      final Uint8List v;
      final int uvRowStride;
      final int uvPixelStride;

      if (_isNv21 && frame.planes.length >= 2) {
        // ── NV21: plane[0] = Y, plane[1] = interleaved VU ──────────────────
        // Split the interleaved VU plane into separate V and U buffers so the
        // existing YUV converter in the isolate can process them as-is.
        debugPrint('Processing NV21 format with ${frame.planes.length} planes');
        y = Uint8List.fromList(frame.planes[0].bytes);
        uvRowStride = frame.planes[1].bytesPerRow;
        uvPixelStride = frame.planes[1].bytesPerPixel ?? 2;

        final vu = frame.planes[1].bytes;
        final vBuf = Uint8List((vu.length / 2).ceil());
        final uBuf = Uint8List((vu.length / 2).ceil());

        // NV21 interleave order is V, U, V, U …
        for (int i = 0, j = 0; i < vu.length - 1; i += 2, j++) {
          vBuf[j] = vu[i];
          uBuf[j] = vu[i + 1];
        }
        v = vBuf;
        u = uBuf;
      } else if (frame.planes.length == 1) {
        // ── Single plane fallback (grayscale Y only) ───────────────────────
        debugPrint('Processing single plane Y format');
        y = Uint8List.fromList(frame.planes[0].bytes);

        // Create dummy UV data (gray)
        final uvSize = (frame.width * frame.height) ~/ 4;
        u = Uint8List(uvSize);
        v = Uint8List(uvSize);
        u.fillRange(0, uvSize, 128); // Neutral UV value
        v.fillRange(0, uvSize, 128); // Neutral UV value
        uvRowStride = frame.width ~/ 2;
        uvPixelStride = 1;
      } else if (frame.planes.length >= 3) {
        // ── YUV420: three separate planes ───────────────────────────────────
        debugPrint(
          'Processing YUV420 format with ${frame.planes.length} planes',
        );
        y = Uint8List.fromList(frame.planes[0].bytes);
        u = Uint8List.fromList(frame.planes[1].bytes);
        v = Uint8List.fromList(frame.planes[2].bytes);
        uvRowStride = frame.planes[1].bytesPerRow;
        uvPixelStride = frame.planes[1].bytesPerPixel ?? 1;
      } else {
        debugPrint(
          'Unsupported frame format with ${frame.planes.length} planes',
        );
        return;
      }

      debugPrint(
        'About to call infer.infer with frame dimensions: ${frame.width}x${frame.height}',
      );

      final dets = await infer.infer(
        CameraImageLike(
          width: frame.width,
          height: frame.height,
          yRowStride: frame.planes[0].bytesPerRow,
          uvRowStride: uvRowStride,
          uvPixelStride: uvPixelStride,
          y: y,
          u: u,
          v: v,
        ),
      );

      final filtered = dets
          .where((d) => d.confidence >= _liveConfidenceThreshold)
          .toList();
      if (mounted) setState(() => _detections = filtered);
    } catch (e) {
      debugPrint('Frame error: $e');
    }
  }

  @override
  void dispose() {
    _ctrl?.stopImageStream();
    _ctrl?.dispose();
    _infer?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.redAccent),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    if (!_ready || _ctrl == null) {
      return const Center(child: CircularProgressIndicator());
    }

    final previewSize = _ctrl!.value.previewSize!; // landscape sensor size
    final logicalSize = Size(previewSize.height, previewSize.width); // portrait

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_ctrl!),
        SizedBox.expand(
          child: CustomPaint(
            painter: CameraDetectionPainter(
              detections: _detections,
              cameraLogicalSize: logicalSize,
            ),
          ),
        ),
        Positioned(
          top: 12,
          right: 12,
          child: _Hud(fps: _fps, count: _detections.length),
        ),
        if (_detections.isNotEmpty)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: _BottomStrip(detections: _detections),
          ),
      ],
    );
  }
}

// ── HUD ───────────────────────────────────────────────────────────────────────

class _Hud extends StatelessWidget {
  final double fps;
  final int count;
  const _Hud({required this.fps, required this.count});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.60),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.speed, color: Colors.greenAccent, size: 12),
              const SizedBox(width: 4),
              Text(
                '${fps.toStringAsFixed(1)} FPS',
                style: const TextStyle(
                  color: Colors.greenAccent,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 2),
          Text(
            '$count object${count == 1 ? '' : 's'}',
            style: const TextStyle(color: Colors.white60, fontSize: 11),
          ),
        ],
      ),
    );
  }
}

// ── Bottom strip ──────────────────────────────────────────────────────────────

class _BottomStrip extends StatelessWidget {
  final List<Detection> detections;
  const _BottomStrip({required this.detections});

  @override
  Widget build(BuildContext context) {
    final sorted = [...detections]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    return Container(
      height: 52,
      color: Colors.black.withValues(alpha: 0.55),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: sorted.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          final d = sorted[i];
          final color = colorForLabel(d.label);
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.20),
              border: Border.all(color: color.withValues(alpha: 0.70)),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              '${d.label}  ${(d.confidence * 100).toStringAsFixed(0)}%',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
          );
        },
      ),
    );
  }
}
