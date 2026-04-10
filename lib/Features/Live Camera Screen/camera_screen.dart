import 'dart:ui' as ui;
import 'package:flutter/material.dart';

import '../../Core/detection_painter.dart';
import '../../Core/native_detection_bridge.dart';
import '../../Core/yolo_model.dart';

/// Live camera detection screen.
///
/// The camera preview is rendered via a Flutter [Texture] widget that maps
/// directly to the native SurfaceTexture. All inference runs in Kotlin.
/// Dart only receives final bounding-box data through the EventChannel.
class LiveCameraScreen extends StatefulWidget {
  final NativeDetectionBridge bridge;
  const LiveCameraScreen({super.key, required this.bridge});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen>
    with WidgetsBindingObserver {

  List<Detection> _detections = const [];
  /// Per-track normalized center history (see notebook `track_history`, maxlen 30).
  Map<int, List<ui.Offset>> _trailNormByTrack = {};
  double _fps     = 0;
  int    _inferMs = 0;
  String? _error;
  int?    _textureId;

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

  static const int _kTrailLength = 30;

  void _onFrame(NativeFrame frame) {
    if (!mounted) return;
    // Native throttles sends to ~12/s (MIN_SEND_INTERVAL_MS = 80).
    // Each setState triggers one raster frame; this is already cheap.
    setState(() {
      _detections = frame.detections;
      _fps        = frame.fps;
      _inferMs    = frame.inferMs;
      _updateTrails(frame.detections);
    });
  }

  void _updateTrails(List<Detection> dets) {
    final seen = <int>{};
    final next = Map<int, List<ui.Offset>>.from(_trailNormByTrack);
    for (final d in dets) {
      if (!d.hasTrackId) continue;
      seen.add(d.trackId);
      final bb = d.boundingBox;
      final c = ui.Offset(bb.centerX, bb.centerY);
      final prev = next[d.trackId] ?? const <ui.Offset>[];
      final list = [...prev, c];
      while (list.length > _kTrailLength) {
        list.removeAt(0);
      }
      next[d.trackId] = list;
    }
    next.removeWhere((k, _) => !seen.contains(k));
    _trailNormByTrack = next;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      widget.bridge.stopCamera();
    } else if (state == AppLifecycleState.resumed) {
      setState(() => _textureId = null);
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
          child: Text(_error!,
              style: const TextStyle(color: Colors.redAccent),
              textAlign: TextAlign.center),
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
        // Native preview — wrap in RepaintBoundary so the texture compositing
        // layer is isolated; Flutter won't re-rasterize it when overlays change.
        RepaintBoundary(child: Texture(textureId: _textureId!)),

        // Bounding-box overlay — kept in its own RepaintBoundary so the
        // raster cache for the Texture is NOT invalidated on every detection.
        RepaintBoundary(
          child: SizedBox.expand(
            child: CustomPaint(
              painter: _FullscreenDetectionPainter(
                detections: _detections,
                trailNormByTrack: _trailNormByTrack,
              ),
            ),
          ),
        ),

        // HUD — light widget, doesn't need its own boundary
        Positioned(
          top: 12, right: 12,
          child: _Hud(fps: _fps, inferMs: _inferMs, count: _detections.length),
        ),

        if (_detections.isNotEmpty)
          Positioned(
            bottom: 0, left: 0, right: 0,
            child: RepaintBoundary(child: _BottomStrip(detections: _detections)),
          ),
      ],
    );
  }
}

// ── Painter ───────────────────────────────────────────────────────────────────

class _FullscreenDetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Map<int, List<ui.Offset>> trailNormByTrack;

  const _FullscreenDetectionPainter({
    required this.detections,
    required this.trailNormByTrack,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final rect = ui.Rect.fromLTWH(0, 0, size.width, size.height);
    drawTrailsAndPredictedPaths(canvas, rect, detections, trailNormByTrack);
    drawDetections(canvas, detections, rect, showMonocularDistance: true);
    drawVelocityArrowsAndLabels(canvas, rect, detections);
    drawRiskOverlay(canvas, rect, detections);
  }

  @override
  bool shouldRepaint(_FullscreenDetectionPainter old) =>
      !identical(old.detections, detections) ||
      !identical(old.trailNormByTrack, trailNormByTrack);
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
          _hudRow(Icons.speed,          Colors.greenAccent,
              '${fps.toStringAsFixed(1)} FPS'),
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
    // Sort is O(n log n) but n is typically < 10; no perf concern.
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
