import 'package:flutter/material.dart';

import '../Core/yolo_model.dart';

class StatsBar extends StatelessWidget {
  final List<Detection> detections;
  final Duration? elapsed;
  const StatsBar({super.key, required this.detections, this.elapsed});

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
          Stat(label: 'Objects', value: '${detections.length}'),
          Stat(label: 'Classes', value: '$unique'),
          if (elapsed != null) Stat(label: 'Time', value: '${elapsed!.inMilliseconds} ms'),
        ],
      ),
    );
  }
}


class Stat extends StatelessWidget {
  final String label;
  final String value;
  const Stat({super.key, required this.label, required this.value});
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
