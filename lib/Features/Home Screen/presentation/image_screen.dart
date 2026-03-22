import 'dart:io';
import 'dart:ui' as ui;

import 'package:adas_499/Core/image_preprocessing.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../../Core/detection_painter.dart';
import '../../../Core/native_detection_bridge.dart';
import '../../../Core/yolo_model.dart';
import '../../../Shared/btn.dart';
import '../../../Shared/detection_chips.dart';
import '../../../Shared/error.dart';
import '../../../Shared/empty.dart';
import '../../../Shared/loading.dart';
import '../../../Shared/statsbar.dart';

class ImageDetectionScreen extends StatefulWidget {
  final NativeDetectionBridge bridge;
  const ImageDetectionScreen({super.key, required this.bridge});

  @override
  State<ImageDetectionScreen> createState() => _ImageDetectionScreenState();
}

class _ImageDetectionScreenState extends State<ImageDetectionScreen> {
  final _picker = ImagePicker();

  ui.Image?       _uiImage;
  List<Detection> _detections = const [];
  bool            _busy       = false;
  String?         _error;
  Duration?       _elapsed;

  Future<void> _pick(ImageSource src) async {
    final xfile = await _picker.pickImage(source: src, imageQuality: 92);
    if (xfile == null) return;
    _runOnFile(File(xfile.path));
  }

  Future<void> _runOnFile(File file) async {
    setState(() {
      _busy       = true;
      _error      = null;
      _detections = const [];
      _uiImage    = null;
    });

    try {
      // Decode for display and for native inference in parallel
      final preprocessor = ImagePreprocessing(image: file);
      final sw = Stopwatch()..start();

      final results = await Future.wait([
        preprocessor.decodeImageForDisplay(),
        widget.bridge.runOnImage(file.path),
      ]);

      sw.stop();

      final uiImage  = results[0] as ui.Image;
      final dets     = results[1] as List<Detection>;

      if (mounted) {
        setState(() {
          _uiImage    = uiImage;
          _detections = dets;
          _elapsed    = sw.elapsed;
          _busy       = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = '$e';
          _busy  = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
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
        if (_detections.isNotEmpty || _elapsed != null)
          StatsBar(detections: _detections, elapsed: _elapsed),
        Expanded(
          child: _busy
              ? const Loading()
              : _error != null
                  ? Err(msg: _error!)
                  : _uiImage != null
                      ? RepaintBoundary(
                          child: SizedBox.expand(
                            child: CustomPaint(
                              painter: ImageDetectionPainter(
                                image: _uiImage!,
                                detections: _detections,
                              ),
                            ),
                          ),
                        )
                      : const Empty(),
        ),
        if (_detections.isNotEmpty) DetectionChips(detections: _detections),
      ],
    );
  }
}
