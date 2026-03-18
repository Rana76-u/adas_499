import 'dart:io';
import 'dart:ui' as ui;

import 'package:adas_499/Core/image_preprocessing.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../../Core/detection_painter.dart';
import '../../../Core/yolo_model.dart';
import '../../../Shared/btn.dart';
import '../../../Shared/detection_chips.dart';
import '../../../Shared/error.dart';
import '../../../Shared/empty.dart';
import '../../../Shared/loading.dart';
import '../../../Shared/statsbar.dart';

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
      // preprocess image for inference and display
      final imgImage = await ImagePreprocessing(image: file).decodeImageForInference();
      final uiImage = await ImagePreprocessing(image: file).decodeImageForDisplay();

      // measure time taken to detect objects in the image
      final sw = Stopwatch()..start();

      // detect objects in the image
      final dets = await widget.model.detect(imgImage);

      // stop timer
      sw.stop();

      if (mounted) {
        setState(() {
          _uiImage = uiImage;
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
              Btn(
                icon: Icons.photo_library_rounded,
                label: 'Gallery',
                onTap: () => _pick(ImageSource.gallery),
              ),
              const SizedBox(width: 12),
              Btn(
                icon: Icons.camera_alt_rounded,
                label: 'Camera',
                onTap: () => _pick(ImageSource.camera),
              ),
            ],
          ),
        ),

        // ── Stats ──────────────────────────────────────────────────────────────
        if (_detections.isNotEmpty || _elapsed != null)
          StatsBar(detections: _detections, elapsed: _elapsed),

        // ── Main canvas ────────────────────────────────────────────────────────
        Expanded(
          child: _busy
                  ? const Loading()
              : _error != null
              ? Err(msg: _error!)
              : _uiImage != null
              ? SizedBox.expand(
                  child: CustomPaint(
                    painter: ImageDetectionPainter(
                      image: _uiImage!,
                      detections: _detections,
                    ),
                  ),
                )
              : const Empty(),
        ),

        // ── Detection chips ────────────────────────────────────────────────────
        if (_detections.isNotEmpty) DetectionChips(detections: _detections),
      ],
    );
  }
}







