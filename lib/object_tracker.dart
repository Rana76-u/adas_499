// lib/object_tracker.dart
import 'dart:math';
import 'yolo_detector.dart';

class ObjectTracker {
  final Map<int, TrackedObject> _tracks = {};
  int _nextId = 0;
  static const double maxDistance = 100.0; // pixels
  static const int maxAge = 30; // frames

  List<TrackedObject> update(List<Detection> detections, int frameNumber) {
    // Match detections to existing tracks
    final unmatched = _matchDetections(detections);

    // Update matched tracks
    for (var track in _tracks.values) {
      track.age++;
      if (track.age > maxAge) {
        _tracks.remove(track.id);
      }
    }

    // Create new tracks for unmatched detections
    for (var detection in unmatched) {
      _tracks[_nextId] = TrackedObject(
        id: _nextId,
        detection: detection,
        frameNumber: frameNumber,
      );
      _nextId++;
    }

    return _tracks.values.where((t) => t.age < 5).toList();
  }

  List<Detection> _matchDetections(List<Detection> detections) {
    List<Detection> unmatched = List.from(detections);

    for (var track in _tracks.values) {
      Detection? bestMatch;
      double bestDistance = double.infinity;

      for (var detection in unmatched) {
        final distance = _calculateDistance(
          track.detection.bbox,
          detection.bbox,
        );

        if (distance < maxDistance && distance < bestDistance) {
          bestDistance = distance;
          bestMatch = detection;
        }
      }

      if (bestMatch != null) {
        track.update(bestMatch);
        unmatched.remove(bestMatch);
      }
    }

    return unmatched;
  }

  double _calculateDistance(BoundingBox a, BoundingBox b) {
    final dx = a.centerX - b.centerX;
    final dy = a.centerY - b.centerY;
    return sqrt(dx * dx + dy * dy);
  }
}

class TrackedObject {
  final int id;
  Detection detection;
  int age = 0;
  final int frameNumber;
  final List<Point<double>> trajectory = [];

  TrackedObject({
    required this.id,
    required this.detection,
    required this.frameNumber,
  }) {
    trajectory.add(Point(detection.bbox.centerX, detection.bbox.centerY));
  }

  void update(Detection newDetection) {
    detection = newDetection;
    age = 0;
    trajectory.add(Point(detection.bbox.centerX, detection.bbox.centerY));

    // Keep last 30 positions
    if (trajectory.length > 30) {
      trajectory.removeAt(0);
    }
  }

  double getVelocity(int fps) {
    if (trajectory.length < 2) return 0;

    final last = trajectory.last;
    final prev = trajectory[trajectory.length - 2];

    final dx = last.x - prev.x;
    final dy = last.y - prev.y;
    final distance = sqrt(dx * dx + dy * dy);

    return distance * fps; // pixels per second
  }
}
