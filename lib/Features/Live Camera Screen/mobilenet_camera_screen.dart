import 'package:camera/camera.dart';
import 'dart:typed_data';
import 'package:flutter/material.dart';

import '../../Core/mobilenet_inference_isolate.dart';
import '../../Core/mobilenet_model.dart';

class MobileNetCameraScreen extends StatefulWidget {
  final MobileNetModel model;
  final int topK;

  const MobileNetCameraScreen({
    super.key,
    required this.model,
    this.topK = 3,
  });

  @override
  State<MobileNetCameraScreen> createState() => _MobileNetCameraScreenState();
}

class _MobileNetCameraScreenState extends State<MobileNetCameraScreen> {
  static const double _liveConfidenceThreshold = 0.10;

  CameraController? _ctrl;
  List<ClassificationResult> _results = [];
  bool _processing = false;
  bool _ready = false;
  String? _error;
  MobileNetInferenceIsolate? _infer;

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

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      if (mounted) setState(() => _error = 'No cameras found on this device.');
      return;
    }

    final attempts = [
      (ImageFormatGroup.nv21,   ResolutionPreset.medium),
      (ImageFormatGroup.yuv420, ResolutionPreset.low),
      (ImageFormatGroup.nv21,   ResolutionPreset.low),
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
        debugPrint('✅ Camera opened — format: $fmt  resolution: $res  nv21: $_isNv21');
        break;
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
      if (mounted) setState(() => _error = 'Model path unavailable (model not loaded?).');
      return;
    }

    try {
      _infer = MobileNetInferenceIsolate(
        modelPath: modelPath,
        labels: widget.model.labels,
        threads: 4,
        delegate: MobileNetDelegate.gpu,
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

      final Uint8List y;
      final Uint8List u;
      final Uint8List v;
      final int uvRowStride;
      final int uvPixelStride;

      if (_isNv21 && frame.planes.length == 2) {
        // ── NV21: plane[0] = Y, plane[1] = interleaved VU ──────────────────
        y             = Uint8List.fromList(frame.planes[0].bytes);
        uvRowStride   = frame.planes[1].bytesPerRow;
        uvPixelStride = frame.planes[1].bytesPerPixel ?? 2;

        final vu   = frame.planes[1].bytes;
        final vBuf = Uint8List((vu.length / 2).ceil());
        final uBuf = Uint8List((vu.length / 2).ceil());

        // NV21 order: V, U, V, U …
        for (int i = 0, j = 0; i < vu.length - 1; i += 2, j++) {
          vBuf[j] = vu[i];
          uBuf[j] = vu[i + 1];
        }
        v = vBuf;
        u = uBuf;
      } else {
        // ── YUV420: three separate planes ───────────────────────────────────
        y             = Uint8List.fromList(frame.planes[0].bytes);
        u             = Uint8List.fromList(frame.planes[1].bytes);
        v             = Uint8List.fromList(frame.planes[2].bytes);
        uvRowStride   = frame.planes[1].bytesPerRow;
        uvPixelStride = frame.planes[1].bytesPerPixel ?? 1;
      }

      final results = await infer.infer(
        CameraFrameLike(
          width:          frame.width,
          height:         frame.height,
          yRowStride:     frame.planes[0].bytesPerRow,
          uvRowStride:    uvRowStride,
          uvPixelStride:  uvPixelStride,
          y: y,
          u: u,
          v: v,
        ),
        topK: widget.topK,
      );

      final filtered = results
          .where((r) => r.confidence >= _liveConfidenceThreshold)
          .toList();

      if (mounted) setState(() => _results = filtered);
    } catch (e) {
      debugPrint('MobileNet frame error: $e');
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

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_ctrl!),
        Positioned(
          top: 12,
          right: 12,
          child: _MobileNetHud(fps: _fps),
        ),
        if (_results.isNotEmpty)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: _ClassificationStrip(results: _results),
          ),
        if (_results.isNotEmpty)
          Positioned(
            top: 12,
            left: 12,
            child: _TopLabel(result: _results.first),
          ),
      ],
    );
  }
}

// ── HUD ───────────────────────────────────────────────────────────────────────

class _MobileNetHud extends StatelessWidget {
  final double fps;
  const _MobileNetHud({required this.fps});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.60),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white12),
      ),
      child: Row(
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
    );
  }
}

// ── Top label ─────────────────────────────────────────────────────────────────

class _TopLabel extends StatelessWidget {
  final ClassificationResult result;
  const _TopLabel({required this.result});

  @override
  Widget build(BuildContext context) {
    final pct = (result.confidence * 100).toStringAsFixed(1);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.tealAccent.withValues(alpha: 0.60)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            result.label,
            style: const TextStyle(
              color: Colors.tealAccent,
              fontSize: 15,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            '$pct%',
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

// ── Bottom classification strip ───────────────────────────────────────────────

class _ClassificationStrip extends StatelessWidget {
  final List<ClassificationResult> results;
  const _ClassificationStrip({required this.results});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black.withValues(alpha: 0.55),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: results.map((r) {
          final pct = r.confidence;
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: Row(
              children: [
                SizedBox(
                  width: 180,
                  child: Text(
                    r.label,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: pct,
                      backgroundColor: Colors.white12,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        _barColor(pct),
                      ),
                      minHeight: 8,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  width: 40,
                  child: Text(
                    '${(pct * 100).toStringAsFixed(0)}%',
                    textAlign: TextAlign.right,
                    style: const TextStyle(
                      color: Colors.white60,
                      fontSize: 11,
                    ),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Color _barColor(double confidence) {
    if (confidence >= 0.70) return Colors.tealAccent;
    if (confidence >= 0.40) return Colors.amberAccent;
    return Colors.redAccent;
  }
}
