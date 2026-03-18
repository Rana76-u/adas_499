import 'package:flutter/material.dart';

class Loading extends StatelessWidget {
  const Loading({super.key});
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