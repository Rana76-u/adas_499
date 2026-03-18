import 'dart:io';
import 'package:image/image.dart' as img;
import 'dart:ui' as ui;

class ImagePreprocessing {
  final File image;
  ImagePreprocessing({required this.image});

  Future<img.Image> decodeImageForInference() async {
     final bytes = await image.readAsBytes();

      // Decode for inference (image package — gives img.Image in RGB)
      final imgImage = img.decodeImage(bytes);
      if (imgImage == null) throw Exception('Cannot decode image');
    return imgImage;
  }

  Future<ui.Image> decodeImageForDisplay() async {
    final bytes = await image.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }
}