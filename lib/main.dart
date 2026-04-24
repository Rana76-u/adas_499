import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'Features/Home Screen/presentation/home_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(const YoloApp());
}

class YoloApp extends StatelessWidget {
  const YoloApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0A0A1A),
        colorScheme: ColorScheme.dark(
          primary: const Color(0xFF1A73E8),
          secondary: const Color(0xFF4FC3F7),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}