import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'yolo_model.dart';

// ── Palette ───────────────────────────────────────────────────────────────────

const List<Color> _palette = [
  Color(0xFFFF3B30),
  Color(0xFFFF9500),
  Color(0xFFFFCC00),
  Color(0xFF34C759),
  Color(0xFF00C7BE),
  Color(0xFF007AFF),
  Color(0xFF5856D6),
  Color(0xFFAF52DE),
  Color(0xFFFF2D55),
  Color(0xFF30B0C7),
  Color(0xFF32ADE6),
  Color(0xFFFF6482),
  Color(0xFF64D2FF),
  Color(0xFFBF5AF2),
  Color(0xFFFFD60A),
];

Color colorForLabel(String label) =>
    _palette[label.hashCode.abs() % _palette.length];

// ── Rect helpers ──────────────────────────────────────────────────────────────

/// BoxFit.contain — scales [src] to fit inside [dst], centred, no cropping.
ui.Rect containRect(Size src, Size dst) {
  final sa = src.width / src.height;
  final da = dst.width / dst.height;
  double w, h;
  if (sa > da) {
    w = dst.width;
    h = dst.width / sa;
  } else {
    h = dst.height;
    w = dst.height * sa;
  }
  return ui.Rect.fromLTWH((dst.width - w) / 2, (dst.height - h) / 2, w, h);
}

/// BoxFit.cover — scales [src] to fill [dst], centred, may crop.
ui.Rect coverRect(Size src, Size dst) {
  final sa = src.width / src.height;
  final da = dst.width / dst.height;
  double w, h;
  if (sa < da) {
    w = dst.width;
    h = dst.width / sa;
  } else {
    h = dst.height;
    w = dst.height * sa;
  }
  return ui.Rect.fromLTWH((dst.width - w) / 2, (dst.height - h) / 2, w, h);
}

// ── Core drawing helpers (used by both painters) ──────────────────────────────

void drawDetections(
  Canvas canvas,
  List<Detection> detections,
  ui.Rect displayRect,
) {
  for (final det in detections) {
    final color = colorForLabel(det.label);
    final bb = det.boundingBox;

    print('bb: $bb');
    print('displayRect: $displayRect');
    print('left: ${displayRect.left}');
    print('top: ${displayRect.top}');
    print('right: ${displayRect.right}');

    print('bottom: ${displayRect.bottom}');
    print('width: ${displayRect.width}');
    print('height: ${displayRect.height}');
    print('bb.left: ${bb.left}');
    print('bb.top: ${bb.top}');
    print('bb.right: ${bb.right}');
    print('bb.bottom: ${bb.bottom}');
    print('bb.width: ${bb.width}');
    print('bb.height: ${bb.height}');

    // Map normalised [0,1] → pixel coords inside displayRect
    final l = displayRect.left + bb.left * displayRect.width;
    final t = displayRect.top + bb.top * displayRect.height;
    final r = displayRect.left + bb.right * displayRect.width;
    final b = displayRect.top + bb.bottom * displayRect.height;
    final rect = ui.Rect.fromLTRB(l, t, r, b);

    // Fill
    canvas.drawRect(rect, Paint()..color = color.withOpacity(0.15));

    // Border
    canvas.drawRect(
      rect,
      Paint()
        ..color = color
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke,
    );

    // Corner ticks
    _drawCorners(canvas, rect, color);

    // Label badge — above the box top edge, or below if near top of screen
    final badgeAnchorY = t < 28 ? b + 2 : t;
    final above = t >= 28;
    _drawBadge(
      canvas,
      '${det.label}  ${(det.confidence * 100).toStringAsFixed(0)}%',
      Offset(l, badgeAnchorY),
      color,
      above: above,
    );
  }
}

void _drawCorners(Canvas canvas, ui.Rect r, Color color) {
  final len = (r.shortestSide * 0.18).clamp(9.0, 18.0);
  final p = Paint()
    ..color = color
    ..strokeWidth = 3.5
    ..style = PaintingStyle.stroke
    ..strokeCap = StrokeCap.round;

  canvas
    // TL
    ..drawLine(r.topLeft, r.topLeft + Offset(len, 0), p)
    ..drawLine(r.topLeft, r.topLeft + Offset(0, len), p)
    // TR
    ..drawLine(r.topRight, r.topRight + Offset(-len, 0), p)
    ..drawLine(r.topRight, r.topRight + Offset(0, len), p)
    // BL
    ..drawLine(r.bottomLeft, r.bottomLeft + Offset(len, 0), p)
    ..drawLine(r.bottomLeft, r.bottomLeft + Offset(0, -len), p)
    // BR
    ..drawLine(r.bottomRight, r.bottomRight + Offset(-len, 0), p)
    ..drawLine(r.bottomRight, r.bottomRight + Offset(0, -len), p);
}

void _drawBadge(
  Canvas canvas,
  String text,
  Offset anchor,
  Color color, {
  required bool above,
}) {
  const style = TextStyle(
    color: Colors.white,
    fontSize: 11.5,
    fontWeight: FontWeight.w700,
  );
  final tp = TextPainter(
    text: TextSpan(text: text, style: style),
    textDirection: TextDirection.ltr,
  )..layout();

  const px = 7.0, py = 4.0;
  final bw = tp.width + px * 2;
  final bh = tp.height + py * 2;

  // "above" → badge sits above the anchor line; else below it
  final by = above ? anchor.dy - bh : anchor.dy;
  final badgeRect = ui.Rect.fromLTWH(anchor.dx, by, bw, bh);

  canvas.drawRRect(
    RRect.fromRectAndRadius(badgeRect, const Radius.circular(4)),
    Paint()..color = color,
  );
  tp.paint(canvas, Offset(badgeRect.left + px, badgeRect.top + py));
}

// ── Still-image painter (contain fit) ────────────────────────────────────────

/// Renders [image] letter-boxed then draws detections on top.
/// Use this for the image-picker screen.
class ImageDetectionPainter extends CustomPainter {
  final ui.Image image;
  final List<Detection> detections;

  const ImageDetectionPainter({required this.image, required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    final imgSize = Size(image.width.toDouble(), image.height.toDouble());
    final fitRect = containRect(imgSize, size);

    // Draw image
    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, imgSize.width, imgSize.height),
      fitRect,
      Paint(),
    );

    // Draw detections mapped onto the image rect
    drawDetections(canvas, detections, fitRect);
  }

  @override
  bool shouldRepaint(ImageDetectionPainter old) =>
      old.image != image || old.detections != detections;
}

// ── Live-camera overlay painter (cover fit) ───────────────────────────────────

/// Paints YOLO boxes over a live [CameraPreview].
///
/// CameraPreview fills its parent using BoxFit.cover, so we replicate that
/// transform so normalised [0,1] coordinates land on the correct pixels.
///
/// [cameraLogicalSize] is the size of the camera sensor in *display* orientation
/// (width = previewSize.height, height = previewSize.width on most phones).
class CameraDetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Size cameraLogicalSize; // sensor display-orientation size

  const CameraDetectionPainter({
    required this.detections,
    required this.cameraLogicalSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // CameraPreview uses BoxFit.cover → use coverRect
    final displayRect = coverRect(cameraLogicalSize, size);
    drawDetections(canvas, detections, displayRect);
  }

  @override
  bool shouldRepaint(CameraDetectionPainter old) =>
      old.detections != detections ||
      old.cameraLogicalSize != cameraLogicalSize;
}
