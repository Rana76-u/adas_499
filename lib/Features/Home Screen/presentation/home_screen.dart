import 'package:adas_499/Features/Live%20Camera%20Screen/camera_screen.dart';
import 'package:adas_499/Features/Home%20Screen/presentation/image_screen.dart';
import 'package:adas_499/Shared/custom_appbar.dart';
import 'package:flutter/material.dart';
import 'package:adas_499/Core/native_detection_bridge.dart';
import 'package:adas_499/Core/yolo_model.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // The bridge owns all native resources (interpreter, camera, delegates).
  final NativeDetectionBridge _bridge = NativeDetectionBridge();

  bool _modelLoaded = false;
  String? _loadError;
  int _selectedTab = 0;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await _bridge.loadModel(
        modelPath:
            'assets/models/yolov8n_float32.tflite', //yolo11n_int8 //android/app/src/main/
        labels: cocoLabels, // swap to cocoLabels if using a COCO model
        delegate: 'nnapi', // 'gpu' | 'nnapi' | 'cpu'
      );
      if (mounted) setState(() => _modelLoaded = true);
    } catch (e) {
      if (mounted) {
        setState(
          () => _loadError =
              'Failed to load model.\n\n'
              'Make sure the .tflite file is listed in pubspec assets.\n\n'
              'Error: $e',
        );
      }
    }
  }

  @override
  void dispose() {
    _bridge.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppbar(modelLoaded: _modelLoaded) as PreferredSizeWidget,
      body: _buildBody(),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedTab,
        onTap: (i) => setState(() => _selectedTab = i),
        backgroundColor: const Color(0xFF0D0D1F),
        selectedItemColor: const Color(0xFF1A73E8),
        unselectedItemColor: Colors.white38,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.image_rounded),
            label: 'Image',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.videocam_rounded),
            label: 'Live Camera',
          ),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_loadError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.error_outline,
                color: Colors.redAccent,
                size: 48,
              ),
              const SizedBox(height: 16),
              Text(
                _loadError!,
                style: const TextStyle(color: Colors.redAccent),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: () {
                  setState(() => _loadError = null);
                  _loadModel();
                },
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (!_modelLoaded) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Color(0xFF1A73E8)),
            SizedBox(height: 20),
            Text(
              'Loading model on device…',
              style: TextStyle(color: Colors.white70, fontSize: 15),
            ),
            SizedBox(height: 8),
            Text(
              'Initialising GPU/NNAPI delegate',
              style: TextStyle(color: Colors.white38, fontSize: 12),
            ),
          ],
        ),
      );
    }

    return IndexedStack(
      index: _selectedTab,
      children: [
        // Image tab still works the same way
        ImageDetectionScreen(bridge: _bridge),
        // Live camera tab now uses the native pipeline
        LiveCameraScreen(bridge: _bridge),
      ],
    );
  }
}
