import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../../Core/detection_painter.dart';
import '../../Core/yolo_model.dart';

class LiveCameraScreen extends StatefulWidget {
  final YoloModel model;
  const LiveCameraScreen({super.key, required this.model});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  CameraController? _ctrl;
  List<Detection> _detections = [];
  bool _processing = false;
  bool _ready = false;
  String? _error;

  int _frameCount = 0;
  double _fps = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No cameras found on this device.');
        return;
      }

      _ctrl = CameraController(
        cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _ctrl!.initialize();
      await _ctrl!.startImageStream(_onFrame);
      if (mounted) setState(() => _ready = true);
    } catch (e) {
      setState(() => _error = 'Camera error: $e');
    }
  }

  void _onFrame(CameraImage frame) {
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
      final image = _yuv420ToRgb(frame);
      final dets = await widget.model.detect(image);
      if (mounted) setState(() => _detections = dets);
    } catch (e) {
      debugPrint('Frame error: $e');
    }
  }

  img.Image _yuv420ToRgb(CameraImage frame) {
    final w = frame.width;
    final h = frame.height;
    final yP = frame.planes[0];
    final uP = frame.planes[1];
    final vP = frame.planes[2];
    final out = img.Image(w, h);

    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final yIdx = row * yP.bytesPerRow + col;
        final uvIdx = (row ~/ 2) * uP.bytesPerRow + (col ~/ 2);
        final yv = yP.bytes[yIdx];
        final uv = uP.bytes[uvIdx] - 128;
        final vv = vP.bytes[uvIdx] - 128;
        out.setPixelRgba(
          col,
          row,
          (yv + 1.402 * vv).clamp(0, 255).toInt(),
          (yv - 0.344 * uv - 0.714 * vv).clamp(0, 255).toInt(),
          (yv + 1.772 * uv).clamp(0, 255).toInt(),
        );
      }
    }
    return out;
  }

  @override
  void dispose() {
    _ctrl?.stopImageStream();
    _ctrl?.dispose();
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

    // The camera sensor reports its size as (sensorWidth × sensorHeight).
    // On most phones the sensor is landscape, so in portrait mode the display
    // size is (sensorHeight wide × sensorWidth tall).
    final previewSize = _ctrl!.value.previewSize!; // landscape
    final logicalSize = Size(previewSize.height, previewSize.width); // portrait

    return Stack(
      fit: StackFit.expand,
      children: [
        // ── Live feed ───────────────────────────────────────────────────────
        CameraPreview(_ctrl!),

        // ── Detection overlay ───────────────────────────────────────────────
        // SizedBox.expand gives CustomPaint the same constraints as the Stack
        // so size == the full screen area, matching what CameraPreview sees.
        SizedBox.expand(
          child: CustomPaint(
            painter: CameraDetectionPainter(
              detections: _detections,
              cameraLogicalSize: logicalSize,
            ),
          ),
        ),

        // ── HUD ─────────────────────────────────────────────────────────────
        Positioned(
          top: 12,
          right: 12,
          child: _Hud(fps: _fps, count: _detections.length),
        ),

        // ── Bottom strip ─────────────────────────────────────────────────────
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
