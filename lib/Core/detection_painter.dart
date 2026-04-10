import 'dart:ui' as ui;
import 'package:flutter/material.dart';
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

// ── Core drawing helpers ──────────────────────────────────────────────────────

void drawDetections(
  Canvas canvas,
  List<Detection> detections,
  ui.Rect displayRect,
) {
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
    _drawBadge(
      canvas,
      '$idPrefix${det.label}  ${(det.confidence * 100).toStringAsFixed(0)}%',
      Offset(l, badgeAnchorY),
      color,
      above: t >= 28,
    );
  }
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
    drawDetections(canvas, detections, fitRect);
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
