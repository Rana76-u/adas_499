import 'package:flutter/foundation.dart';

/// Runtime-tweakable parameters used by risk and distance calculations.
class RuntimeTuning {
  final double focalLengthMm;
  final double sensorHeightMm;
  final double ttcThresholdSeconds;
  final double collisionDistPixels;

  const RuntimeTuning({
    required this.focalLengthMm,
    required this.sensorHeightMm,
    required this.ttcThresholdSeconds,
    required this.collisionDistPixels,
  });

  static const RuntimeTuning defaults = RuntimeTuning(
    focalLengthMm: 4.84,
    sensorHeightMm: 4.33,
    ttcThresholdSeconds: 2.0,
    collisionDistPixels: 100.0,
  );

  RuntimeTuning copyWith({
    double? focalLengthMm,
    double? sensorHeightMm,
    double? ttcThresholdSeconds,
    double? collisionDistPixels,
  }) {
    return RuntimeTuning(
      focalLengthMm: focalLengthMm ?? this.focalLengthMm,
      sensorHeightMm: sensorHeightMm ?? this.sensorHeightMm,
      ttcThresholdSeconds: ttcThresholdSeconds ?? this.ttcThresholdSeconds,
      collisionDistPixels: collisionDistPixels ?? this.collisionDistPixels,
    );
  }
}

final ValueNotifier<RuntimeTuning> runtimeTuningNotifier = ValueNotifier(
  RuntimeTuning.defaults,
);
