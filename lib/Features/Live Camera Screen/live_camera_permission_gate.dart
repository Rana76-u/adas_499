import 'dart:async';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../../Core/native_detection_bridge.dart';
import 'camera_screen.dart';

class LiveCameraPermissionGate extends StatefulWidget {
  final NativeDetectionBridge bridge;

  const LiveCameraPermissionGate({super.key, required this.bridge});

  @override
  State<LiveCameraPermissionGate> createState() =>
      _LiveCameraPermissionGateState();
}

class _LiveCameraPermissionGateState extends State<LiveCameraPermissionGate>
    with WidgetsBindingObserver {
  bool _checking = true;
  bool _requesting = false;
  bool _granted = false;
  bool _permanentlyDenied = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    unawaited(_refreshPermissionState());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      unawaited(_refreshPermissionState());
    }
  }

  Future<void> _refreshPermissionState() async {
    setState(() {
      _checking = true;
      _error = null;
    });

    try {
      final status = await Permission.camera.status;
      if (!mounted) return;

      setState(() {
        _granted = status.isGranted || status.isLimited;
        _permanentlyDenied = status.isPermanentlyDenied;
        _checking = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _checking = false;
        _error = 'Failed to check permission: $e';
      });
    }
  }

  Future<void> _requestPermission() async {
    setState(() {
      _requesting = true;
      _error = null;
    });

    try {
      final status = await Permission.camera.request();
      if (!mounted) return;

      setState(() {
        _granted = status.isGranted || status.isLimited;
        _permanentlyDenied = status.isPermanentlyDenied;
      });

      // Some Android devices apply permission changes asynchronously.
      // Re-check once more to avoid false negatives after user accepts.
      await Future<void>.delayed(const Duration(milliseconds: 250));
      await _refreshPermissionState();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Failed to request permission: $e');
    } finally {
      if (mounted) {
        setState(() => _requesting = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_checking) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Color(0xFF1A73E8)),
            SizedBox(height: 16),
            Text(
              'Checking camera permission…',
              style: TextStyle(color: Colors.white70),
            ),
          ],
        ),
      );
    }

    if (_granted) {
      return LiveCameraScreen(bridge: widget.bridge);
    }

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.no_photography_rounded,
              size: 56,
              color: Colors.orangeAccent,
            ),
            const SizedBox(height: 16),
            const Text(
              'Camera permission is required',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              _permanentlyDenied
                  ? 'Permission was permanently denied. Open settings and enable camera access.'
                  : 'Please allow camera access to use the live detection screen.',
              style: const TextStyle(color: Colors.white70),
              textAlign: TextAlign.center,
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.redAccent),
              ),
            ],
            const SizedBox(height: 20),
            if (_permanentlyDenied)
              ElevatedButton.icon(
                onPressed: () async {
                  await openAppSettings();
                  if (mounted) {
                    await _refreshPermissionState();
                  }
                },
                icon: const Icon(Icons.settings),
                label: const Text('Open Settings'),
              )
            else
              ElevatedButton.icon(
                onPressed: _requesting ? null : _requestPermission,
                icon: _requesting
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.verified_user_outlined),
                label: Text(_requesting ? 'Requesting…' : 'Grant Permission'),
              ),
            const SizedBox(height: 8),
            TextButton.icon(
              onPressed: _refreshPermissionState,
              icon: const Icon(Icons.refresh),
              label: const Text('Refresh permission status'),
            ),
          ],
        ),
      ),
    );
  }
}
