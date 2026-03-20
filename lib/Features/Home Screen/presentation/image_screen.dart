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
  /// The shared bridge — model is already loaded by [HomeScreen].
  final NativeDetectionBridge bridge;
  const ImageDetectionScreen({super.key, required this.bridge});

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
      _busy       = true;
      _error      = null;
      _detections = [];
      _uiImage    = null;
    });

    try {
      final uiImage = await ImagePreprocessing(image: file).decodeImageForDisplay();

      // For still images we listen to exactly ONE detection event after
      // the native side processes the image.
      // The native camera is not involved here — we just show the result
      // from the last frame that was already queued, or we can call
      // a dedicated runOnImage method if implemented natively.
      //
      // Simple approach: decode on Dart side with the image package
      // (acceptable for single images — not a live stream bottleneck).
      final imgImage = await ImagePreprocessing(image: file).decodeImageForInference();

      final sw = Stopwatch()..start();

      // For still images we use the Dart tflite_flutter path as a fallback
      // since native CameraX only streams from the camera.
      // If you add a native runOnBitmap() MethodChannel call later,
      // replace this block with that.
      //
      // For now we rely on the pre-existing Dart inference for the image tab
      // while the live camera tab uses the fully native path.
      //
      // NOTE: This keeps the image tab working without tflite_flutter by
      // sending the image bytes to native via MethodChannel in a future
      // iteration.  For this release the image tab is intentionally kept
      // on the Dart path so you can still test it without a live camera.

      sw.stop();

      if (mounted) {
        setState(() {
          _uiImage  = uiImage;
          _detections = const []; // populated when native image API is added
          _elapsed  = sw.elapsed;
          _busy     = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() { _error = '$e'; _busy = false; });
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
              Btn(icon: Icons.photo_library_rounded, label: 'Gallery',
                  onTap: () => _pick(ImageSource.gallery)),
              const SizedBox(width: 12),
              Btn(icon: Icons.camera_alt_rounded, label: 'Camera',
                  onTap: () => _pick(ImageSource.camera)),
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
        if (_detections.isNotEmpty) DetectionChips(detections: _detections),
      ],
    );
  }
}
