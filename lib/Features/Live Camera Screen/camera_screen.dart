import 'dart:async';
import 'dart:io' show Platform;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter_ringtone_player/flutter_ringtone_player.dart';
import 'package:geolocator/geolocator.dart';
import 'package:vibration/vibration.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../../Core/detection_painter.dart';
import '../../Core/native_detection_bridge.dart';
import '../../Core/yolo_model.dart';

/// Live camera detection screen.
///
/// The camera preview is rendered via a Flutter [Texture] widget that maps
/// directly to the native SurfaceTexture. All inference runs in Kotlin.
/// Dart only receives final bounding-box data through the EventChannel.
class LiveCameraScreen extends StatefulWidget {
  final NativeDetectionBridge bridge;
  final bool inferenceEnabled;
  final bool controlsEnabled;
  final ValueChanged<bool> onInferenceChanged;
  const LiveCameraScreen({
    super.key,
    required this.bridge,
    required this.inferenceEnabled,
    required this.controlsEnabled,
    required this.onInferenceChanged,
  });

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen>
    with WidgetsBindingObserver {
  List<Detection> _detections = const [];

  /// Per-track normalized center history (see notebook `track_history`, maxlen 30).
  Map<int, List<ui.Offset>> _trailNormByTrack = {};
  double _fps = 0;
  int _inferMs = 0;
  String? _error;
  int? _textureId;
  RiskAssessment _riskAssessment = const RiskAssessment(
    overall: RiskLevel.none,
    byTrackId: {},
  );
  RiskLevel _lastMediumOrHigherAlert = RiskLevel.none;
  Timer? _highRiskAlertTimer;
  Timer? _highRiskFlashTimer;
  bool _highRiskFlashOn = true;
  bool _highRiskToneActive = false;
  final FlutterRingtonePlayer _ringtonePlayer = FlutterRingtonePlayer();
  StreamSubscription<NativeFrame>? _detectionSub;
  StreamSubscription<Position>? _positionSub;
  bool _cameraStarting = false;
  bool _cameraStopping = false;
  double _speedKmh = 0;
  String _roadSignMessage = 'No road sign detected.';
  String _actionMessage = 'Drive carefully and obey traffic rules.';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    if (widget.inferenceEnabled) {
      _startNativeCamera();
    }
    _startSpeedTracking();
  }

  Future<void> _startNativeCamera() async {
    if (_cameraStarting || _textureId != null) return;
    try {
      _cameraStarting = true;
      final textureId = await widget.bridge.startCamera();
      if (!mounted) return;
      setState(() => _textureId = textureId);
      await _setKeepAwake(true);

      _detectionSub?.cancel();
      _detectionSub = widget.bridge.detectionStream.listen(
        _onFrame,
        onError: (e) {
          if (mounted) setState(() => _error = e.toString());
        },
      );
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      _cameraStarting = false;
    }
  }

  Future<void> _stopNativeCamera() async {
    if (_cameraStopping) return;
    _cameraStopping = true;
    try {
      await _detectionSub?.cancel();
      _detectionSub = null;
      await widget.bridge.stopCamera();
    } finally {
      await _setKeepAwake(false);
      _cameraStopping = false;
    }
    if (!mounted) return;
    setState(() {
      _textureId = null;
      _detections = const [];
      _fps = 0;
      _inferMs = 0;
      _trailNormByTrack = {};
      _riskAssessment = const RiskAssessment(
        overall: RiskLevel.none,
        byTrackId: {},
      );
    });
  }

  Future<void> _setKeepAwake(bool enabled) async {
    if (enabled) {
      await WakelockPlus.enable();
    } else {
      await WakelockPlus.disable();
    }
  }

  static const int _kTrailLength = 30;

  void _onFrame(NativeFrame frame) {
    if (!mounted) return;
    // Native throttles sends to ~12/s (MIN_SEND_INTERVAL_MS = 80).
    // Each setState triggers one raster frame; this is already cheap.
    setState(() {
      _detections = frame.detections;
      _fps = frame.fps;
      _inferMs = frame.inferMs;
      _updateTrails(frame.detections);
      _riskAssessment = assessRiskLevels(
        frame.detections,
        const ui.Rect.fromLTWH(0, 0, 1, 1),
      );
      _updateRoadSignGuidance(frame.detections);
    });
    _updateRiskAlerts(_riskAssessment.overall);
  }

  Future<void> _startSpeedTracking() async {
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      if (!mounted) return;
      setState(() {
        _actionMessage = 'Enable GPS/location service to show speed.';
      });
      return;
    }

    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      if (!mounted) return;
      setState(() {
        _actionMessage = 'Location permission needed for speed monitoring.';
      });
      return;
    }

    _positionSub?.cancel();
    _positionSub =
        Geolocator.getPositionStream(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.bestForNavigation,
            distanceFilter: 2,
          ),
        ).listen((position) {
          if (!mounted) return;
          final speedMs = position.speed.isFinite && position.speed >= 0
              ? position.speed
              : 0.0;
          setState(() {
            _speedKmh = speedMs * 3.6;
            _updateRoadSignGuidance(_detections);
          });
        });
  }

  void _updateRoadSignGuidance(List<Detection> detections) {
    final relevantDetections = detections
        .where(
          (d) => isBlueRoadSignLabel(d.label) || isRoadDamageLabel(d.label),
        )
        .toList();
    if (relevantDetections.isEmpty) {
      _roadSignMessage = 'No road sign detected.';
      _actionMessage = 'Drive carefully and obey traffic rules.';
      return;
    }

    relevantDetections.sort((a, b) => b.confidence.compareTo(a.confidence));
    final detectedLabel = relevantDetections.first.label;
    if (isRoadDamageLabel(detectedLabel)) {
      _roadSignMessage = 'Detected road damage: $detectedLabel';
      _actionMessage = _advisoryForRoadDamage(detectedLabel);
      return;
    }

    _roadSignMessage = 'Detected sign: $detectedLabel';
    _actionMessage = _advisoryForSign(detectedLabel, _speedKmh);
  }

  String _advisoryForSign(String label, double speedKmh) {
    final speedLimit = _speedLimitFromLabel(label);
    if (speedLimit != null) {
      if (speedKmh > speedLimit) {
        return 'Reduce speed now: limit is $speedLimit km/h, your speed is '
            '${speedKmh.toStringAsFixed(0)} km/h.';
      }
      return 'Maintain at or below $speedLimit km/h. Current speed is '
          '${speedKmh.toStringAsFixed(0)} km/h.';
    }

    return switch (label) {
      'Crossroads' =>
        'Approach slowly and check all directions before crossing.',
      'Hospital Ahead' => 'Reduce horn use and drive slowly near the hospital.',
      'Junction Ahead' => 'Prepare to slow down and watch for joining traffic.',
      'Mosque Ahead' => 'Drive slowly and stay alert for pedestrians nearby.',
      'No Pedestrians' => 'Keep lane discipline and do not stop unexpectedly.',
      'No Vehicle Entry' =>
        'Do not enter this road. Choose an alternate route.',
      'Pedestrians Crossing' =>
        'Slow down and be ready to stop for pedestrians.',
      'School Ahead' => 'Slow down and watch carefully for children crossing.',
      'Sharp Left Turn' => 'Reduce speed and steer carefully to the left.',
      'Sharp Right Turn' => 'Reduce speed and steer carefully to the right.',
      'Side Road On Left' =>
        'Watch for vehicles entering from the left side road.',
      'Side Road On Right' =>
        'Watch for vehicles entering from the right side road.',
      'Speed Breaker' =>
        'Slow down immediately to pass the speed breaker safely.',
      'Traffic Merges From Left' =>
        'Keep safe gap and watch for merging vehicles from left.',
      'Traffic Merges From Right' =>
        'Keep safe gap and watch for merging vehicles from right.',
      'U Turn' => 'Be prepared for vehicles making U-turns ahead.',
      _ => 'Proceed with caution and follow this road sign.',
    };
  }

  int? _speedLimitFromLabel(String label) {
    if (label == 'Speed Limit 20 km') return 20;
    if (label == 'Speed Limit 40Km') return 40;
    if (label == 'Speed Limit 80Km') return 80;
    return null;
  }

  String _advisoryForRoadDamage(String label) {
    return switch (label) {
      'pothole' =>
        'Pothole ahead: slow down, hold steering firmly, and avoid sudden braking.',
      'crack' =>
        'Road crack ahead: reduce speed and pass carefully while maintaining lane control.',
      _ => 'Road damage detected ahead. Slow down and proceed carefully.',
    };
  }

  void _updateRiskAlerts(RiskLevel currentRisk) {
    if (currentRisk == RiskLevel.high) {
      _startHighRiskLoop();
      return;
    }

    _stopHighRiskLoop();
    if (currentRisk == RiskLevel.medium &&
        _lastMediumOrHigherAlert != RiskLevel.medium) {
      // Medium risk: one short beep + light haptic pulse.
      _playMediumAlert();
    }
    _lastMediumOrHigherAlert = currentRisk;
  }

  void _startHighRiskLoop() {
    if (!_highRiskToneActive) {
      _ringtonePlayer.play(
        android: AndroidSounds.alarm,
        ios: IosSounds.alarm,
        looping: true,
        volume: 1.0,
        asAlarm: true,
      );
      _highRiskToneActive = true;
    }

    if (_highRiskAlertTimer == null) {
      _highRiskAlertTimer = Timer.periodic(const Duration(milliseconds: 350), (
        _,
      ) {
        // High risk: rapid haptic pulses while alarm tone loops.
        _vibrate(durationMs: 120, amplitude: 200);
      });
      _lastMediumOrHigherAlert = RiskLevel.high;
    }
    _highRiskFlashTimer ??= Timer.periodic(const Duration(milliseconds: 260), (
      _,
    ) {
      if (!mounted) return;
      setState(() => _highRiskFlashOn = !_highRiskFlashOn);
    });
  }

  void _stopHighRiskLoop() {
    _highRiskAlertTimer?.cancel();
    _highRiskAlertTimer = null;
    _highRiskFlashTimer?.cancel();
    _highRiskFlashTimer = null;
    if (_highRiskFlashOn == false && mounted) {
      setState(() => _highRiskFlashOn = true);
    } else {
      _highRiskFlashOn = true;
    }
    if (_highRiskToneActive) {
      _ringtonePlayer.stop();
      _highRiskToneActive = false;
    }
    Vibration.cancel();
  }

  Future<void> _playMediumAlert() async {
    _ringtonePlayer.play(
      android: AndroidSounds.notification,
      ios: IosSounds.glass,
      looping: false,
      volume: 0.85,
      asAlarm: false,
    );
    await _vibrate(durationMs: 90, amplitude: 120);
  }

  Future<void> _vibrate({
    required int durationMs,
    required int amplitude,
  }) async {
    final hasVibrator = await Vibration.hasVibrator();
    if (!hasVibrator) return;
    await Vibration.vibrate(duration: durationMs, amplitude: amplitude);
  }

  void _updateTrails(List<Detection> dets) {
    final seen = <int>{};
    final next = Map<int, List<ui.Offset>>.from(_trailNormByTrack);
    for (final d in dets) {
      if (!d.hasTrackId) continue;
      seen.add(d.trackId);
      final bb = d.boundingBox;
      final c = ui.Offset(bb.centerX, bb.centerY);
      final prev = next[d.trackId] ?? const <ui.Offset>[];
      final list = [...prev, c];
      while (list.length > _kTrailLength) {
        list.removeAt(0);
      }
      next[d.trackId] = list;
    }
    next.removeWhere((k, _) => !seen.contains(k));
    _trailNormByTrack = next;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      _stopHighRiskLoop();
      _stopNativeCamera();
    } else if (state == AppLifecycleState.resumed) {
      if (widget.inferenceEnabled) {
        _startNativeCamera();
      }
    }
  }

  @override
  void didUpdateWidget(covariant LiveCameraScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.inferenceEnabled != widget.inferenceEnabled) {
      if (widget.inferenceEnabled) {
        _startNativeCamera();
      } else {
        _stopHighRiskLoop();
        _stopNativeCamera();
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopHighRiskLoop();
    unawaited(_stopNativeCamera());
    unawaited(_positionSub?.cancel());
    _detectionSub = null;
    _positionSub = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.redAccent),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    final int cameraQuarterTurns = Platform.isAndroid ? 3 : 0;

    return Stack(
      fit: StackFit.expand,
      children: [
        Positioned.fill(
          child: _textureId != null
              // Rotate only preview texture; detections remain in native display space.
              ? RepaintBoundary(
                  child: RotatedBox(
                    quarterTurns: cameraQuarterTurns,
                    child: Texture(textureId: _textureId!),
                  ),
                )
              : const ColoredBox(color: Colors.black),
        ),

        // Bounding-box overlay — kept in its own RepaintBoundary so the
        // raster cache for the Texture is NOT invalidated on every detection.
        RepaintBoundary(
          child: SizedBox.expand(
            child: CustomPaint(
              painter: _FullscreenDetectionPainter(
                detections: _detections,
                trailNormByTrack: _trailNormByTrack,
                riskAssessment: _riskAssessment,
                highRiskFlashOn: _highRiskFlashOn,
              ),
            ),
          ),
        ),
        IgnorePointer(
          child: AnimatedOpacity(
            opacity:
                _riskAssessment.overall == RiskLevel.high && _highRiskFlashOn
                ? 1
                : 0,
            duration: const Duration(milliseconds: 120),
            curve: Curves.easeOut,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Colors.redAccent.withValues(alpha: 0.10),
                border: Border.all(
                  color: Colors.redAccent.withValues(alpha: 0.70),
                  width: 5,
                ),
              ),
              child: const SizedBox.expand(),
            ),
          ),
        ),
        if (_textureId == null)
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (widget.inferenceEnabled)
                  const CircularProgressIndicator(color: Color(0xFF1A73E8)),
                if (widget.inferenceEnabled) const SizedBox(height: 16),
                Text(
                  widget.inferenceEnabled
                      ? 'Starting camera…'
                      : 'Inference is paused',
                  style: const TextStyle(color: Colors.white54, fontSize: 13),
                ),
              ],
            ),
          ),
        // HUD — light widget, doesn't need its own boundary
        Positioned(
          left: 12,
          top: 12,
          child: _DrivingInfo(
            speedKmh: _speedKmh,
            roadSignMessage: _roadSignMessage,
            actionMessage: _actionMessage,
          ),
        ),
        Positioned(
          top: 12,
          right: 12,
          child: _Hud(
            fps: _fps,
            inferMs: _inferMs,
            count: _detections.length,
            risk: _riskAssessment.overall,
          ),
        ),
        Positioned(
          left: 12,
          bottom: 20,
          child: FilledButton.icon(
            onPressed: _controlEnabled()
                ? () {
                    final next = !widget.inferenceEnabled;
                    widget.onInferenceChanged(next);
                  }
                : null,
            icon: Icon(
              widget.inferenceEnabled ? Icons.pause_circle : Icons.play_circle,
            ),
            label: Text(
              widget.inferenceEnabled ? 'Stop Inference' : 'Start Inference',
            ),
          ),
        ),
      ],
    );
  }

  bool _controlEnabled() {
    if (!widget.controlsEnabled) return false;
    if (_cameraStarting || _cameraStopping) return false;
    if (widget.inferenceEnabled && _textureId == null) return false;
    return true;
  }
}

// ── Painter ───────────────────────────────────────────────────────────────────

class _FullscreenDetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Map<int, List<ui.Offset>> trailNormByTrack;
  final RiskAssessment riskAssessment;
  final bool highRiskFlashOn;

  const _FullscreenDetectionPainter({
    required this.detections,
    required this.trailNormByTrack,
    required this.riskAssessment,
    required this.highRiskFlashOn,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final rect = ui.Rect.fromLTWH(0, 0, size.width, size.height);
    drawTrailsAndPredictedPaths(canvas, rect, detections, trailNormByTrack);
    drawDetections(
      canvas,
      detections,
      rect,
      showMonocularDistance: true,
      riskByTrackId: riskAssessment.byTrackId,
      highRiskFlashOn: highRiskFlashOn,
    );
    drawVelocityArrowsAndLabels(canvas, rect, detections);
    drawRiskOverlay(canvas, rect, detections);
  }

  @override
  bool shouldRepaint(_FullscreenDetectionPainter old) =>
      !identical(old.detections, detections) ||
      !identical(old.trailNormByTrack, trailNormByTrack) ||
      !identical(old.riskAssessment, riskAssessment) ||
      old.highRiskFlashOn != highRiskFlashOn;
}

// ── HUD ───────────────────────────────────────────────────────────────────────

class _Hud extends StatelessWidget {
  final double fps;
  final int inferMs;
  final int count;
  final RiskLevel risk;
  const _Hud({
    required this.fps,
    required this.inferMs,
    required this.count,
    required this.risk,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.60),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          _hudRow(
            Icons.speed,
            Colors.greenAccent,
            '${fps.toStringAsFixed(1)} FPS',
          ),
          const SizedBox(height: 2),
          _hudRow(Icons.timer_outlined, Colors.blue, '$inferMs ms'),
          const SizedBox(height: 2),
          Text(
            '$count object${count == 1 ? '' : 's'}',
            style: const TextStyle(color: Colors.white60, fontSize: 11),
          ),
          const SizedBox(height: 2),
          Text(
            'Risk: ${risk.name.toUpperCase()}',
            style: TextStyle(
              color: switch (risk) {
                RiskLevel.high => Colors.redAccent,
                RiskLevel.medium => Colors.orangeAccent,
                RiskLevel.low => Colors.yellowAccent,
                RiskLevel.none => Colors.white54,
              },
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _hudRow(IconData icon, Color color, String text) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(icon, color: color, size: 12),
      const SizedBox(width: 4),
      Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    ],
  );
}

class _DrivingInfo extends StatelessWidget {
  final double speedKmh;
  final String roadSignMessage;
  final String actionMessage;

  const _DrivingInfo({
    required this.speedKmh,
    required this.roadSignMessage,
    required this.actionMessage,
  });

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 260),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.65),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.blueAccent.withValues(alpha: 0.55)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Speed: ${speedKmh.toStringAsFixed(0)} km/h',
              style: const TextStyle(
                color: Colors.lightBlueAccent,
                fontWeight: FontWeight.w700,
                fontSize: 13,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              roadSignMessage,
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            const SizedBox(height: 4),
            Text(
              actionMessage,
              style: const TextStyle(color: Colors.orangeAccent, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}
