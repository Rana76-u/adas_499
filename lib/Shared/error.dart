import 'package:flutter/material.dart';

class Err extends StatelessWidget {
  final String msg;
  const Err({super.key, required this.msg});
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
