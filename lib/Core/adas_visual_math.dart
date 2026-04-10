import 'dart:math' as math;
import 'dart:ui';

/// MiDaS-style monocular distance from the notebook (`calc_distance`):
/// \( D = \frac{H \cdot f \cdot N_p}{h_p \cdot S} \)
/// With \( h_p = h_{\mathrm{norm}} \cdot N_p \), this reduces to
/// \( D = \frac{H \cdot f}{h_{\mathrm{norm}} \cdot S} \).
double estimateDistanceMeters({
  required double boxHeightNorm,
  required String label,
  double focalLengthMm = 4.84,
  double sensorHeightMm = 4.33,
}) {
  if (boxHeightNorm <= 1e-6 || sensorHeightMm <= 1e-6) {
    return double.infinity;
  }
  final realObjectHeightM = defaultObjectHeightMeters(label);
  return (realObjectHeightM * focalLengthMm) / (boxHeightNorm * sensorHeightMm);
}

/// Rough class priors for real-world height (meters), aligned with the notebook.
double defaultObjectHeightMeters(String label) {
  final k = label.toLowerCase();
  const m = {
    'person': 1.67,
    'bicycle': 0.8,
    'car': 1.5,
    'motorcycle': 0.8,
    'bus': 3.0,
    'truck': 3.2,
    'rickshaw': 1.75,
    'other-vehicle': 1.75,
    'cng': 1.5,
    'dog': 0.5,
    'cat': 0.25,
  };
  return m[k] ?? 1.67;
}

/// ASCII direction buckets from `get_direction` in [assets/code.ipynb].
String directionLabelFromVelocity(double vx, double vy) {
  var deg = math.atan2(vy, vx) * 180 / math.pi;
  deg %= 360;
  if (deg < 0) deg += 360;

  if (deg >= 337.5 || deg < 22.5) return 'E';
  if (deg < 67.5) return 'SE';
  if (deg < 112.5) return 'S';
  if (deg < 157.5) return 'SW';
  if (deg < 202.5) return 'W';
  if (deg < 247.5) return 'NW';
  if (deg < 292.5) return 'N';
  return 'NE';
}

/// Speed in normalized coordinates per second (matches native `vx`/`vy` scale).
double speedNormPerSec(double vxNormPerSec, double vyNormPerSec) =>
    math.sqrt(vxNormPerSec * vxNormPerSec + vyNormPerSec * vyNormPerSec);

/// `predict_trajectory` from the notebook — last up to 5 centers, mean delta, extrapolate.
List<Offset> predictTrajectoryNorm(List<Offset> history, int predictionSteps) {
  if (history.length < 2 || predictionSteps <= 0) return const [];
  final take = history.length < 5 ? history.length : 5;
  final recent = history.sublist(history.length - take);
  var sx = 0.0, sy = 0.0;
  for (var i = 1; i < recent.length; i++) {
    sx += recent[i].dx - recent[i - 1].dx;
    sy += recent[i].dy - recent[i - 1].dy;
  }
  final c = recent.length - 1;
  final mx = sx / c;
  final my = sy / c;
  final last = recent.last;
  return List.generate(
    predictionSteps,
    (k) => Offset(last.dx + mx * (k + 1), last.dy + my * (k + 1)),
  );
}

/// `calc_camera_ttc` ported from the notebook, but in normalized coordinates.
///
/// [objPosNorm] and [camPosNorm] are in [0,1]x[0,1] space, [velNormPerSec] is
/// normalized velocity per second (as emitted by native).
double? calcCameraTtcNorm({
  required Offset objPosNorm,
  required Offset velNormPerSec,
  required Offset camPosNorm,
  double eps = 1e-6,
}) {
  final rx = objPosNorm.dx - camPosNorm.dx;
  final ry = objPosNorm.dy - camPosNorm.dy;
  final rvx = velNormPerSec.dx;
  final rvy = velNormPerSec.dy;

  final v2 = rvx * rvx + rvy * rvy;
  if (v2 < eps) return null;

  final ttc = -(rx * rvx + ry * rvy) / v2;
  return ttc > 0 ? ttc : null;
}
