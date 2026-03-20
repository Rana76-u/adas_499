/// Represents a single detected object.
/// [boundingBox] uses normalized [0,1] coordinates relative to the
/// original image dimensions.
class Detection {
  final String label;
  final double confidence;
  final Rect boundingBox;

  const Detection({
    required this.label,
    required this.confidence,
    required this.boundingBox,
  });

  /// Deserialise from the Map sent over the native EventChannel.
  factory Detection.fromMap(Map<Object?, Object?> map) {
    return Detection(
      label: map['label'] as String,
      confidence: (map['confidence'] as num).toDouble(),
      boundingBox: Rect(
        left:   (map['left']   as num).toDouble(),
        top:    (map['top']    as num).toDouble(),
        right:  (map['right']  as num).toDouble(),
        bottom: (map['bottom'] as num).toDouble(),
      ),
    );
  }

  @override
  String toString() =>
      'Detection(label: $label, '
      'confidence: ${confidence.toStringAsFixed(2)}, '
      'box: $boundingBox)';
}

class Rect {
  final double left;
  final double top;
  final double right;
  final double bottom;

  const Rect({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  double get width  => right  - left;
  double get height => bottom - top;
  double get centerX => (left  + right)  / 2;
  double get centerY => (top   + bottom) / 2;

  @override
  String toString() =>
      'Rect(l:${left.toStringAsFixed(3)}, t:${top.toStringAsFixed(3)}, '
      'r:${right.toStringAsFixed(3)}, b:${bottom.toStringAsFixed(3)})';
}

// ── Label lists ──────────────────────────────────────────────────────────────

const List<String> customLabels = [
  'Crossroads',
  'Hospital Ahead',
  'Junction Ahead',
  'Mosque Ahead',
  'No Pedestrians',
  'No Vehicle Entry',
  'Pedestrians Crossing',
  'School Ahead',
  'Sharp Left Turn',
  'Sharp Right Turn',
  'Side Road On Left',
  'Side Road On Right',
  'Speed Breaker',
  'Speed Limit 20 km',
  'Speed Limit 40Km',
  'Speed Limit 80Km',
  'Traffic Merges From Left',
  'Traffic Merges From Right',
  'U Turn',
  'bicycle',
  'bus',
  'car',
  'cng',
  'motorcycle',
  'other-vehicle',
  'person',
  'rickshaw',
];

const List<String> cocoLabels = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
  'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];
