import 'dart:ui' as ui;
import 'package:flutter/material.dart';

import '../../Core/detection_painter.dart';
import '../../Core/native_detection_bridge.dart';
import '../../Core/yolo_model.dart';

/// Live camera detection screen.
///
/// The camera preview is rendered via a Flutter [Texture] widget that maps
/// directly to the native SurfaceTexture registered with [FlutterEngine].
/// All inference (YUV→RGB, letterbox, TFLite, NMS) runs entirely in Kotlin.
/// Dart only receives final bounding-box data through the EventChannel.
class LiveCameraScreen extends StatefulWidget {
  final NativeDetectionBridge bridge;
  const LiveCameraScreen({super.key, required this.bridge});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen>
    with WidgetsBindingObserver {

  List<Detection> _detections = [];
  double _fps     = 0;
  int    _inferMs = 0;
  String? _error;

  /// The Flutter texture ID returned by native `startCamera`.
  /// Until it is set the screen shows a loading indicator.
  int? _textureId;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _startNativeCamera();
  }

  Future<void> _startNativeCamera() async {
    try {
      final textureId = await widget.bridge.startCamera();
      if (!mounted) return;
      setState(() => _textureId = textureId);

      widget.bridge.detectionStream.listen(
        _onFrame,
        onError: (e) {
          if (mounted) setState(() => _error = e.toString());
        },
      );
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  void _onFrame(NativeFrame frame) {
    if (!mounted) return;
    // Use a direct setState — this is already throttled by the native
    // frame-drop gate, so we never flood the UI thread.
    setState(() {
      _detections = frame.detections;
      _fps        = frame.fps;
      _inferMs    = frame.inferMs;
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      widget.bridge.stopCamera();
    } else if (state == AppLifecycleState.resumed) {
      _textureId = null;
      _startNativeCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    widget.bridge.stopCamera();
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

    if (_textureId == null) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Color(0xFF1A73E8)),
            SizedBox(height: 16),
            Text('Starting camera…',
                style: TextStyle(color: Colors.white54, fontSize: 13)),
          ],
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // ── Native camera preview via Flutter Texture widget ──────────────────
        // Texture maps directly to the SurfaceTexture registered in Kotlin.
        // Zero pixel copies — the GPU composites this straight onto the screen.
        Texture(textureId: _textureId!),

        // ── Bounding-box overlay ──────────────────────────────────────────────
        // Uses a fullscreen painter; bounding boxes are already in normalised
        // [0,1] screen coordinates, so no fit-transform is needed here.
        SizedBox.expand(
          child: CustomPaint(
            painter: _FullscreenDetectionPainter(detections: _detections),
          ),
        ),

        // ── HUD ───────────────────────────────────────────────────────────────
        Positioned(
          top: 12, right: 12,
          child: _Hud(fps: _fps, inferMs: _inferMs, count: _detections.length),
        ),

        // ── Bottom detection strip ────────────────────────────────────────────
        if (_detections.isNotEmpty)
          Positioned(
            bottom: 0, left: 0, right: 0,
            child: _BottomStrip(detections: _detections),
          ),
      ],
    );
  }
}

// ── Painter ───────────────────────────────────────────────────────────────────

class _FullscreenDetectionPainter extends CustomPainter {
  final List<Detection> detections;
  const _FullscreenDetectionPainter({required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    drawDetections(
      canvas,
      detections,
      ui.Rect.fromLTWH(0, 0, size.width, size.height),
    );
  }

  @override
  bool shouldRepaint(_FullscreenDetectionPainter old) =>
      old.detections != detections;
}

// ── HUD ───────────────────────────────────────────────────────────────────────

class _Hud extends StatelessWidget {
  final double fps;
  final int    inferMs;
  final int    count;
  const _Hud({required this.fps, required this.inferMs, required this.count});

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
          _hudRow(Icons.speed,          Colors.greenAccent, '${fps.toStringAsFixed(1)} FPS'),
          const SizedBox(height: 2),
          _hudRow(Icons.timer_outlined, Colors.amberAccent, '$inferMs ms'),
          const SizedBox(height: 2),
          Text('$count object${count == 1 ? '' : 's'}',
              style: const TextStyle(color: Colors.white60, fontSize: 11)),
        ],
      ),
    );
  }

  Widget _hudRow(IconData icon, Color color, String text) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(icon, color: color, size: 12),
      const SizedBox(width: 4),
      Text(text,
          style: TextStyle(
            color: color, fontSize: 12, fontWeight: FontWeight.bold)),
    ],
  );
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
          final d     = sorted[i];
          final color = colorForLabel(d.label);
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color:  color.withValues(alpha: 0.20),
              border: Border.all(color: color.withValues(alpha: 0.70)),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              '${d.label}  ${(d.confidence * 100).toStringAsFixed(0)}%',
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.w600),
            ),
          );
        },
      ),
    );
  }
}
