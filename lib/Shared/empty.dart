import 'package:flutter/material.dart';

class Empty extends StatelessWidget {
  const Empty({super.key});
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
