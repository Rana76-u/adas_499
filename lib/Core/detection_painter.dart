import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'adas_visual_math.dart';
import 'yolo_model.dart';

// ── Palette ───────────────────────────────────────────────────────────────────

const List<Color> _palette = [
  Color(0xFFFF3B30), Color(0xFFFF9500), Color(0xFFFFCC00), Color(0xFF34C759),
  Color(0xFF00C7BE), Color(0xFF007AFF), Color(0xFF5856D6), Color(0xFFAF52DE),
  Color(0xFFFF2D55), Color(0xFF30B0C7), Color(0xFF32ADE6), Color(0xFFFF6482),
  Color(0xFF64D2FF), Color(0xFFBF5AF2), Color(0xFFFFD60A),
];

Color colorForLabel(String label) =>
    _palette[label.hashCode.abs() % _palette.length];

// ── Rect helpers ──────────────────────────────────────────────────────────────

ui.Rect containRect(Size src, Size dst) {
  final sa = src.width / src.height;
  final da = dst.width / dst.height;
  final double w, h;
  if (sa > da) { w = dst.width;  h = dst.width  / sa; }
  else          { h = dst.height; w = dst.height * sa; }
  return ui.Rect.fromLTWH((dst.width - w) / 2, (dst.height - h) / 2, w, h);
}

ui.Rect coverRect(Size src, Size dst) {
  final sa = src.width / src.height;
  final da = dst.width / dst.height;
  final double w, h;
  if (sa < da) { w = dst.width;  h = dst.width  / sa; }
  else          { h = dst.height; w = dst.height * sa; }
  return ui.Rect.fromLTWH((dst.width - w) / 2, (dst.height - h) / 2, w, h);
}

// ── Reusable Paint objects — allocated once, mutated per draw call ────────────
// Eliminates per-detection Paint() allocations in the hot path.
final _fillPaint   = Paint();
final _borderPaint = Paint()
  ..strokeWidth = 2.5
  ..style = PaintingStyle.stroke;
final _cornerPaint = Paint()
  ..strokeWidth = 3.5
  ..style = PaintingStyle.stroke
  ..strokeCap = StrokeCap.round;

// Notebook palette (BGR) → Flutter: trail magenta, pred yellow, velocity red
final _trailPaint = Paint()
  ..strokeWidth = 2
  ..style = PaintingStyle.stroke
  ..strokeCap = StrokeCap.round
  ..color = const Color(0xFFFF00FF);
final _predPaint = Paint()
  ..strokeWidth = 2
  ..style = PaintingStyle.stroke
  ..strokeCap = StrokeCap.round
  ..color = const Color(0xFFFFFF00);
final _velPaint = Paint()
  ..strokeWidth = 2
  ..style = PaintingStyle.stroke
  ..strokeCap = StrokeCap.round
  ..color = const Color(0xFFFF0000);

final _riskHighPaint = Paint()
  ..strokeWidth = 2.5
  ..style = PaintingStyle.stroke
  ..strokeCap = StrokeCap.round
  ..color = const Color(0xFFFF0000); // RED
final _riskMedPaint = Paint()
  ..strokeWidth = 2.5
  ..style = PaintingStyle.stroke
  ..strokeCap = StrokeCap.round
  ..color = const Color(0xFFFFA500); // ORANGE
final _riskLowPaint = Paint()
  ..strokeWidth = 2.0
  ..style = PaintingStyle.stroke
  ..strokeCap = StrokeCap.round
  ..color = const Color(0xFFFFFF00); // YELLOW

const int _kPredictionSteps = 15;

// ── Core drawing helpers ──────────────────────────────────────────────────────

/// Normalized center → overlay pixel position.
Offset _normToScreen(Offset norm, ui.Rect displayRect) => Offset(
      displayRect.left + norm.dx * displayRect.width,
      displayRect.top + norm.dy * displayRect.height,
    );

void drawDetections(
  Canvas canvas,
  List<Detection> detections,
  ui.Rect displayRect, {
  bool showMonocularDistance = false,
}) {
  for (final det in detections) {
    final color = colorForLabel(det.label);
    final bb = det.boundingBox;

    final l = displayRect.left + bb.left   * displayRect.width;
    final t = displayRect.top  + bb.top    * displayRect.height;
    final r = displayRect.left + bb.right  * displayRect.width;
    final b = displayRect.top  + bb.bottom * displayRect.height;
    final rect = ui.Rect.fromLTRB(l, t, r, b);

    _fillPaint.color   = color.withValues(alpha: 0.15);
    canvas.drawRect(rect, _fillPaint);

    _borderPaint.color = color;
    canvas.drawRect(rect, _borderPaint);

    _drawCorners(canvas, rect, color);

    final badgeAnchorY = t < 28 ? b + 2 : t;
    final idPrefix = det.hasTrackId ? 'ID:${det.trackId} · ' : '';
    var line = '$idPrefix${det.label}  ${(det.confidence * 100).toStringAsFixed(0)}%';
    if (showMonocularDistance) {
      final dist = estimateDistanceMeters(
        boxHeightNorm: bb.height,
        label: det.label,
      );
      if (dist.isFinite && dist > 0.3 && dist < 250) {
        line += '  D:${dist.toStringAsFixed(1)}m';
      }
    }
    _drawBadge(
      canvas,
      line,
      Offset(l, badgeAnchorY),
      color,
      above: t >= 28,
    );
  }
}

/// Trail (magenta) + predicted path (yellow), drawn beneath boxes — see [assets/code.ipynb].
void drawTrailsAndPredictedPaths(
  Canvas canvas,
  ui.Rect displayRect,
  List<Detection> detections,
  Map<int, List<Offset>> trailNormByTrack,
) {
  for (final e in trailNormByTrack.entries) {
    final hist = e.value;
    if (hist.length > 1) {
      for (var i = 1; i < hist.length; i++) {
        canvas.drawLine(
          _normToScreen(hist[i - 1], displayRect),
          _normToScreen(hist[i], displayRect),
          _trailPaint,
        );
      }
    }
  }

  for (final det in detections) {
    if (!det.hasTrackId) continue;
    final pts = trailNormByTrack[det.trackId];
    if (pts == null || pts.length < 2) continue;
    final pred = predictTrajectoryNorm(pts, _kPredictionSteps);
    if (pred.length < 2) continue;
    for (var i = 0; i < pred.length - 1; i++) {
      canvas.drawLine(
        _normToScreen(pred[i], displayRect),
        _normToScreen(pred[i + 1], displayRect),
        _predPaint,
      );
    }
  }
}

/// Velocity arrow (red) + direction / speed caption — notebook uses `speed > 5` px/s.
void drawVelocityArrowsAndLabels(
  Canvas canvas,
  ui.Rect displayRect,
  List<Detection> detections,
) {
  final scale = math.min(displayRect.width, displayRect.height) * 0.04;
  const speedThresholdNorm = 0.003;

  for (final det in detections) {
    if (!det.hasTrackId) continue;
    final sp = speedNormPerSec(det.vxNormPerSec, det.vyNormPerSec);
    if (sp <= speedThresholdNorm) continue;

    final bb = det.boundingBox;
    final cx = (bb.left + bb.right) / 2;
    final cy = (bb.top + bb.bottom) / 2;
    final center = Offset(
      displayRect.left + cx * displayRect.width,
      displayRect.top + cy * displayRect.height,
    );
    final end = Offset(
      center.dx + det.vxNormPerSec * scale,
      center.dy + det.vyNormPerSec * scale,
    );
    _drawArrowLine(canvas, center, end, _velPaint.color);

    final dir = directionLabelFromVelocity(det.vxNormPerSec, det.vyNormPerSec);
    final pxApprox = sp * math.min(displayRect.width, displayRect.height);
    final caption = '$dir ${pxApprox.toStringAsFixed(0)} px/s';
    _drawSubLabel(canvas, caption, Offset(
      displayRect.left + bb.left * displayRect.width,
      displayRect.top + bb.bottom * displayRect.height + 4,
    ));
  }
}

/// Notebook-style risk visualization using TTC and pixel distance.
///
/// Camera is assumed at bottom-center of [displayRect], mirroring
/// `cam_pos = (w // 2, h)` in the notebook.
void drawRiskOverlay(
  Canvas canvas,
  ui.Rect displayRect,
  List<Detection> detections, {
  double ttcThresholdSeconds = 2.0,
  double collisionDistPixels = 100,
}) {
  if (detections.isEmpty) return;

  final camNorm = const Offset(0.5, 1.0);
  final camPx = Offset(
    displayRect.left + camNorm.dx * displayRect.width,
    displayRect.top + camNorm.dy * displayRect.height,
  );

  for (final det in detections) {
    if (!det.hasTrackId) continue;

    final bb = det.boundingBox;
    final cx = (bb.left + bb.right) / 2;
    final cy = (bb.top + bb.bottom) / 2;
    final centerNorm = Offset(cx, cy);
    final centerPx = Offset(
      displayRect.left + cx * displayRect.width,
      displayRect.top + cy * displayRect.height,
    );

    final velNorm = Offset(det.vxNormPerSec, det.vyNormPerSec);
    final ttc = calcCameraTtcNorm(
      objPosNorm: centerNorm,
      velNormPerSec: velNorm,
      camPosNorm: camNorm,
    );

    final dxPx = centerPx.dx - camPx.dx;
    final dyPx = centerPx.dy - camPx.dy;
    final distPx = math.sqrt(dxPx * dxPx + dyPx * dyPx);

    String? level;
    if (ttc != null && ttc < ttcThresholdSeconds) {
      if (ttc < 0.5) {
        level = 'HIGH';
      } else if (ttc < 1.0) {
        level = 'MEDIUM';
      } else {
        level = 'LOW';
      }
    } else if (distPx < collisionDistPixels) {
      level = 'MEDIUM';
    }

    if (level != null) {
      final paint = switch (level) {
        'HIGH' => _riskHighPaint,
        'MEDIUM' => _riskMedPaint,
        _ => _riskLowPaint,
      };

      canvas.drawLine(camPx, centerPx, paint);

      final mid = Offset(
        (camPx.dx + centerPx.dx) / 2,
        (camPx.dy + centerPx.dy) / 2,
      );

      var txt = '$level RISK';
      if (ttc != null) {
        txt += ' (${ttc.toStringAsFixed(1)}s)';
      }
      _drawSubLabel(canvas, txt, mid.translate(-20, -6));
    }

    if (ttc != null) {
      final ttcColor = ttc > 2.0
          ? Colors.green
          : (ttc > 1.0 ? Colors.orange : Colors.red);
      _drawSubLabelColored(
        canvas,
        'TTC:${ttc.toStringAsFixed(1)}s',
        Offset(
          displayRect.left + bb.left * displayRect.width,
          displayRect.top + bb.bottom * displayRect.height + 18,
        ),
        ttcColor,
      );
    }
  }
}

void _drawArrowLine(Canvas canvas, Offset from, Offset to, Color color) {
  _velPaint.color = color;
  final d = to - from;
  final len = d.distance;
  if (len < 2) return;
  final u = Offset(d.dx / len, d.dy / len);
  canvas.drawLine(from, to, _velPaint);
  final headLen = math.min(10.0, len * 0.35);
  final base = to - u * headLen;
  final perp = Offset(-u.dy, u.dx) * (headLen * 0.45);
  canvas.drawLine(to, base + perp, _velPaint);
  canvas.drawLine(to, base - perp, _velPaint);
}

final _subTp = TextPainter(textDirection: TextDirection.ltr);

const _subStyle = TextStyle(
  color: Color(0xFFFF4444),
  fontSize: 10.5,
  fontWeight: FontWeight.w600,
);

void _drawSubLabel(Canvas canvas, String text, Offset anchor) {
  _subTp.text = TextSpan(text: text, style: _subStyle);
  _subTp.layout();
  _subTp.paint(canvas, anchor);
}

void _drawSubLabelColored(Canvas canvas, String text, Offset anchor, Color color) {
  _subTp.text = TextSpan(
    text: text,
    style: _subStyle.copyWith(color: color),
  );
  _subTp.layout();
  _subTp.paint(canvas, anchor);
}

void _drawCorners(Canvas canvas, ui.Rect r, Color color) {
  final len = (r.shortestSide * 0.18).clamp(9.0, 18.0);
  _cornerPaint.color = color;
  canvas
    ..drawLine(r.topLeft,     r.topLeft     + Offset(len, 0),   _cornerPaint)
    ..drawLine(r.topLeft,     r.topLeft     + Offset(0,   len), _cornerPaint)
    ..drawLine(r.topRight,    r.topRight    + Offset(-len, 0),  _cornerPaint)
    ..drawLine(r.topRight,    r.topRight    + Offset(0,   len), _cornerPaint)
    ..drawLine(r.bottomLeft,  r.bottomLeft  + Offset(len, 0),   _cornerPaint)
    ..drawLine(r.bottomLeft,  r.bottomLeft  + Offset(0,  -len), _cornerPaint)
    ..drawLine(r.bottomRight, r.bottomRight + Offset(-len, 0),  _cornerPaint)
    ..drawLine(r.bottomRight, r.bottomRight + Offset(0,  -len), _cornerPaint);
}

// Reusable TextPainter — layout() is still called per badge but the object
// itself is not re-created on every draw call.
final _tp = TextPainter(textDirection: TextDirection.ltr);

const _badgePx = 7.0;
const _badgePy = 4.0;
const _badgeStyle = TextStyle(
  color: Colors.white,
  fontSize: 11.5,
  fontWeight: FontWeight.w700,
);

final _badgePaint = Paint();

void _drawBadge(
  Canvas canvas,
  String text,
  Offset anchor,
  Color color, {
  required bool above,
}) {
  _tp.text = TextSpan(text: text, style: _badgeStyle);
  _tp.layout();

  final bw = _tp.width  + _badgePx * 2;
  final bh = _tp.height + _badgePy * 2;
  final by = above ? anchor.dy - bh : anchor.dy;
  final badgeRect = ui.Rect.fromLTWH(anchor.dx, by, bw, bh);

  _badgePaint.color = color;
  canvas.drawRRect(
    RRect.fromRectAndRadius(badgeRect, const Radius.circular(4)),
    _badgePaint,
  );
  _tp.paint(canvas, Offset(badgeRect.left + _badgePx, badgeRect.top + _badgePy));
}

// ── Still-image painter ───────────────────────────────────────────────────────

class ImageDetectionPainter extends CustomPainter {
  final ui.Image image;
  final List<Detection> detections;

  const ImageDetectionPainter({required this.image, required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    final imgSize = Size(image.width.toDouble(), image.height.toDouble());
    final fitRect = containRect(imgSize, size);
    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, imgSize.width, imgSize.height),
      fitRect,
      Paint(),
    );
    drawDetections(canvas, detections, fitRect, showMonocularDistance: true);
  }

  @override
  bool shouldRepaint(ImageDetectionPainter old) =>
      old.image != image || !identical(old.detections, detections);
}

// ── Live-camera overlay painter ───────────────────────────────────────────────

class CameraDetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Size cameraLogicalSize;

  const CameraDetectionPainter({
    required this.detections,
    required this.cameraLogicalSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    drawDetections(canvas, detections, coverRect(cameraLogicalSize, size));
  }

  @override
  bool shouldRepaint(CameraDetectionPainter old) =>
      !identical(old.detections, detections) ||
      old.cameraLogicalSize != cameraLogicalSize;
}
