import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

import 'detection_painter.dart';
import 'yolo_model.dart';

class ImageDetectionScreen extends StatefulWidget {
  final YoloModel model;
  const ImageDetectionScreen({super.key, required this.model});

  @override
  State<ImageDetectionScreen> createState() => _ImageDetectionScreenState();
}

class _ImageDetectionScreenState extends State<ImageDetectionScreen> {
  final _picker = ImagePicker();

  ui.Image? _uiImage;
  List<Detection> _detections = [];
  bool _busy = false;
  String? _error;
  Duration? _elapsed;

  Future<void> _pick(ImageSource src) async {
    final xfile = await _picker.pickImage(source: src, imageQuality: 92);
    if (xfile == null) return;
    _runOnFile(File(xfile.path));
  }

  Future<void> _runOnFile(File file) async {
    setState(() {
      _busy = true;
      _error = null;
      _detections = [];
      _uiImage = null;
    });

    try {
      final bytes = await file.readAsBytes();

      // Decode for inference (image package — gives img.Image in RGB)
      final imgImage = img.decodeImage(bytes);
      if (imgImage == null) throw Exception('Cannot decode image');

      // Decode separately for display (dart:ui — Flutter's native renderer)
      // We decode twice because img.Image and ui.Image serve different purposes
      // and there is no zero-copy path between the two packages.
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();

      final sw = Stopwatch()..start();
      // detect() now handles letterbox + normalize + unscale internally
      final dets = await widget.model.detect(imgImage);
      sw.stop();

      if (mounted) {
        setState(() {
          _uiImage = frame.image;
          _detections = dets;
          _elapsed = sw.elapsed;
          _busy = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = '$e';
          _busy = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // ── Toolbar ────────────────────────────────────────────────────────────
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _Btn(
                icon: Icons.photo_library_rounded,
                label: 'Gallery',
                onTap: () => _pick(ImageSource.gallery),
              ),
              const SizedBox(width: 12),
              _Btn(
                icon: Icons.camera_alt_rounded,
                label: 'Camera',
                onTap: () => _pick(ImageSource.camera),
              ),
            ],
          ),
        ),

        // ── Stats ──────────────────────────────────────────────────────────────
        if (_detections.isNotEmpty || _elapsed != null)
          _StatsBar(detections: _detections, elapsed: _elapsed),

        // ── Main canvas ────────────────────────────────────────────────────────
        Expanded(
          child: _busy
              ? const _Loading()
              : _error != null
              ? _Err(msg: _error!)
              : _uiImage != null
              ? SizedBox.expand(
                  child: CustomPaint(
                    painter: ImageDetectionPainter(
                      image: _uiImage!,
                      detections: _detections,
                    ),
                  ),
                )
              : const _Empty(),
        ),

        // ── Detection chips ────────────────────────────────────────────────────
        if (_detections.isNotEmpty) _Chips(detections: _detections),
      ],
    );
  }
}

// ── Shared widgets ────────────────────────────────────────────────────────────

class _Btn extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _Btn({required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) => ElevatedButton.icon(
    onPressed: onTap,
    icon: Icon(icon, size: 17),
    label: Text(label),
    style: ElevatedButton.styleFrom(
      backgroundColor: const Color(0xFF1A73E8),
      foregroundColor: Colors.white,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 11),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
    ),
  );
}

class _StatsBar extends StatelessWidget {
  final List<Detection> detections;
  final Duration? elapsed;
  const _StatsBar({required this.detections, this.elapsed});

  @override
  Widget build(BuildContext context) {
    final unique = detections.map((d) => d.label).toSet().length;
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 0, 12, 8),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _Stat('Objects', '${detections.length}'),
          _Stat('Classes', '$unique'),
          if (elapsed != null) _Stat('Time', '${elapsed!.inMilliseconds} ms'),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  final String label;
  final String value;
  const _Stat(this.label, this.value);
  @override
  Widget build(BuildContext context) => Column(
    children: [
      Text(
        value,
        style: const TextStyle(
          color: Colors.greenAccent,
          fontWeight: FontWeight.bold,
          fontSize: 16,
        ),
      ),
      Text(label, style: const TextStyle(color: Colors.white54, fontSize: 11)),
    ],
  );
}

class _Chips extends StatelessWidget {
  final List<Detection> detections;
  const _Chips({required this.detections});

  @override
  Widget build(BuildContext context) {
    final sorted = [...detections]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    return Container(
      height: 118,
      color: const Color(0xFF0F0F23),
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        itemCount: sorted.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          final d = sorted[i];
          final color = colorForLabel(d.label);
          return Container(
            width: 96,
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.14),
              border: Border.all(color: color.withOpacity(0.55)),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(_iconFor(d.label), color: color, size: 22),
                const SizedBox(height: 4),
                Text(
                  d.label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                  textAlign: TextAlign.center,
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  '${(d.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(color: color, fontSize: 11),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  IconData _iconFor(String label) {
    const m = {
      'person': Icons.person,
      'car': Icons.directions_car,
      'truck': Icons.local_shipping,
      'bus': Icons.directions_bus,
      'bicycle': Icons.directions_bike,
      'motorcycle': Icons.two_wheeler,
      'dog': Icons.pets,
      'cat': Icons.catching_pokemon,
      'bird': Icons.flutter_dash,
      'bottle': Icons.local_drink,
      'cup': Icons.coffee,
      'laptop': Icons.laptop,
      'cell phone': Icons.phone_android,
      'tv': Icons.tv,
      'chair': Icons.chair,
      'book': Icons.book,
      'clock': Icons.access_time,
      'knife': Icons.cut,
    };
    return m[label] ?? Icons.category;
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => const Center(
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        CircularProgressIndicator(color: Color(0xFF1A73E8)),
        SizedBox(height: 14),
        Text(
          'Running YOLO…',
          style: TextStyle(color: Colors.white54, fontSize: 14),
        ),
      ],
    ),
  );
}

class _Err extends StatelessWidget {
  final String msg;
  const _Err({required this.msg});
  @override
  Widget build(BuildContext context) => Center(
    child: Padding(
      padding: const EdgeInsets.all(28),
      child: Text(
        msg,
        style: const TextStyle(color: Colors.redAccent),
        textAlign: TextAlign.center,
      ),
    ),
  );
}

class _Empty extends StatelessWidget {
  const _Empty();
  @override
  Widget build(BuildContext context) => const Center(
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.image_search_rounded, size: 72, color: Colors.white10),
        SizedBox(height: 14),
        Text(
          'Pick an image to start detecting',
          style: TextStyle(color: Colors.white30, fontSize: 14),
        ),
      ],
    ),
  );
}
