import 'package:flutter/material.dart';

import '../Core/yolo_model.dart';

class DetectionChips extends StatelessWidget {
  final List<Detection> detections;
  const DetectionChips({super.key, required this.detections});
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
          final color = _colorForLabel(d.label);
          return Container(
            width: 96,
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.14),
              border: Border.all(color: color.withValues(alpha: 0.55)),
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

  Color _colorForLabel(String label) {
    const Map<String, Color> m = {
      'person': Colors.red,
      'car': Colors.blue,
      'truck': Colors.green,
      'bus': Colors.yellow,
      'bicycle': Colors.purple,
      'motorcycle': Colors.orange,
    };
    return m[label] ?? Colors.grey;
  }

  IconData _iconFor(String label) {
    const Map<String, IconData> m = {
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
