import 'package:adas_499/Features/Live%20Camera%20Screen/live_camera_permission_gate.dart';
import 'package:adas_499/Features/Settings%20Screen/settings_screen.dart';
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
  bool _isInferenceOn = true;
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
    final isLandscape =
        MediaQuery.of(context).orientation == Orientation.landscape;

    return Scaffold(
      appBar:
          isLandscape
              ? null
              : CustomAppbar(modelLoaded: _modelLoaded) as PreferredSizeWidget,
      body: isLandscape ? _buildLandscapeLayout() : _buildBody(),
      bottomNavigationBar: isLandscape ? null : _buildBottomNavigationBar(),
    );
  }

  Widget _buildLandscapeLayout() {
    return SafeArea(
      child: Row(
        children: [
          Expanded(child: _buildBody()),
          Container(
            width: 72,
            decoration: const BoxDecoration(
              color: Color(0xCC0D0D1F),
              border: Border(left: BorderSide(color: Colors.white12)),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildSideRailButton(
                  icon: Icons.videocam_rounded,
                  tooltip: 'Live Feed',
                  selected: _selectedTab == 0,
                  onTap: () => _onTabSelected(0),
                ),
                const SizedBox(height: 16),
                _buildSideRailButton(
                  icon: Icons.settings_rounded,
                  tooltip: 'Settings',
                  selected: _selectedTab == 1,
                  onTap: () => _onTabSelected(1),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSideRailButton({
    required IconData icon,
    required String tooltip,
    required bool selected,
    required VoidCallback onTap,
  }) {
    return Tooltip(
      message: tooltip,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(14),
        child: Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: selected ? const Color(0xFF1A73E8) : Colors.transparent,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: selected ? const Color(0xFF1A73E8) : Colors.white24,
            ),
          ),
          child: Icon(icon, color: selected ? Colors.white : Colors.white70),
        ),
      ),
    );
  }

  Widget _buildBottomNavigationBar() {
    return BottomNavigationBar(
      currentIndex: _selectedTab,
      onTap: _onTabSelected,
      backgroundColor: const Color(0xFF0D0D1F),
      selectedItemColor: const Color(0xFF1A73E8),
      unselectedItemColor: Colors.white38,
      items: const [
        BottomNavigationBarItem(
          icon: Icon(Icons.videocam_rounded),
          label: 'Live Feed',
        ),
        BottomNavigationBarItem(
          icon: Icon(Icons.settings_rounded),
          label: 'Settings',
        ),
      ],
    );
  }

  void _onTabSelected(int i) {
    setState(() {
      _selectedTab = i;
      if (i != 0) {
        // Leave Live Feed => immediately stop inference.
        _isInferenceOn = false;
      }
    });
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
        // Gate camera screen behind explicit permission handling.
        LiveCameraPermissionGate(
          bridge: _bridge,
          inferenceEnabled: _isInferenceOn,
          controlsEnabled: _modelLoaded,
          onInferenceChanged: (enabled) {
            if (!mounted) return;
            setState(() => _isInferenceOn = enabled);
          },
        ),
        // Image tab still works the same way
        SettingsScreen(),
      ],
    );
  }
}
