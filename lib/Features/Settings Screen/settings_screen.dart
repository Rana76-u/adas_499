import 'package:flutter/material.dart';
import '../../Core/runtime_tuning.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ValueListenableBuilder<RuntimeTuning>(
        valueListenable: runtimeTuningNotifier,
        builder: (context, tuning, _) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Runtime Tuning',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 6),
                const Text(
                  'Adjust these values for better detection and safety.',
                  style: TextStyle(color: Colors.white70),
                ),
                const SizedBox(height: 20),
                _TuningSlider(
                  label: 'TTC Threshold (seconds)',
                  value: tuning.ttcThresholdSeconds,
                  min: 0.5,
                  max: 5.0,
                  divisions: 45,
                  valueText: '${tuning.ttcThresholdSeconds.toStringAsFixed(2)} s',
                  onChanged: (v) {
                    runtimeTuningNotifier.value = tuning.copyWith(
                      ttcThresholdSeconds: v,
                    );
                  },
                ),
                _TuningSlider(
                  label: 'Collision Distance (pixels)',
                  value: tuning.collisionDistPixels,
                  min: 20,
                  max: 300,
                  divisions: 56,
                  valueText: '${tuning.collisionDistPixels.toStringAsFixed(0)} px',
                  onChanged: (v) {
                    runtimeTuningNotifier.value = tuning.copyWith(
                      collisionDistPixels: v,
                    );
                  },
                ),
                _TuningSlider(
                  label: 'Focal Length (mm)',
                  value: tuning.focalLengthMm,
                  min: 2.0,
                  max: 10.0,
                  divisions: 80,
                  valueText: '${tuning.focalLengthMm.toStringAsFixed(2)} mm',
                  onChanged: (v) {
                    runtimeTuningNotifier.value = tuning.copyWith(
                      focalLengthMm: v,
                    );
                  },
                ),
                _TuningSlider(
                  label: 'Sensor Height (mm)',
                  value: tuning.sensorHeightMm,
                  min: 2.0,
                  max: 8.0,
                  divisions: 60,
                  valueText: '${tuning.sensorHeightMm.toStringAsFixed(2)} mm',
                  onChanged: (v) {
                    runtimeTuningNotifier.value = tuning.copyWith(
                      sensorHeightMm: v,
                    );
                  },
                ),
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerRight,
                  child: OutlinedButton.icon(
                    onPressed: () {
                      runtimeTuningNotifier.value = RuntimeTuning.defaults;
                    },
                    icon: const Icon(Icons.restart_alt),
                    label: const Text('Reset Defaults'),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _TuningSlider extends StatelessWidget {
  final String label;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final String valueText;
  final ValueChanged<double> onChanged;

  const _TuningSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.valueText,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Slider(
            value: value.clamp(min, max),
            min: min,
            max: max,
            divisions: divisions,
            label: valueText,
            onChanged: onChanged,
          ),
          Align(
            alignment: Alignment.centerRight,
            child: Text(valueText, style: const TextStyle(color: Colors.white70)),
          ),
        ],
      ),
    );
  }
}
