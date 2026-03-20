import 'dart:io';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;

/// Utilities for decoding a picked image file for display and (optional)
/// Dart-side inference (used by the static image tab only).
///
/// The live camera tab no longer uses this class — all preprocessing
/// for the live stream happens natively in Kotlin.
class ImagePreprocessing {
  final File image;
  ImagePreprocessing({required this.image});

  /// Decode to [img.Image] for Dart-side inference (image tab only).
  Future<img.Image> decodeImageForInference() async {
    final bytes = await image.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception('Cannot decode image');
    return decoded;
  }

  /// Decode to [ui.Image] for Flutter canvas display.
  Future<ui.Image> decodeImageForDisplay() async {
    final bytes = await image.readAsBytes();
    final codec  = await ui.instantiateImageCodec(bytes);
    final frame  = await codec.getNextFrame();
    return frame.image;
  }
}
