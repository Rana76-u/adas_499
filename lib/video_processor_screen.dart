// lib/video_processor_screen.dart
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'dart:io';
import 'dart:async';
import 'dart:math' as math;
import 'package:video_thumbnail/video_thumbnail.dart';
import 'package:image/image.dart' as img;
import 'yolo_detector.dart';
import 'object_tracker.dart';

class VideoProcessorScreen extends StatefulWidget {
  final File videoFile;

  const VideoProcessorScreen({super.key, required this.videoFile});

  @override
  _VideoProcessorScreenState createState() => _VideoProcessorScreenState();
}

class _VideoProcessorScreenState extends State<VideoProcessorScreen> {
  VideoPlayerController? _controller;
  YoloDetector? _detector;
  ObjectTracker? _tracker;
  final List<TrackedObject> _currentTracks = [];
  bool _isProcessing = false;
  bool _isFrameProcessing = false;
  Timer? _processingTimer;
  int _frameNumber = 0;
  bool _isModelLoaded = false;
  Size? _videoSize;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
    _initializeVideo();
  }

  Future<void> _initializeDetector() async {
    try {
      _detector = YoloDetector();
      await _detector!.loadModel();
      _tracker = ObjectTracker();
      setState(() {
        _isModelLoaded = true;
      });
      print('✅ Detector and tracker initialized');
    } catch (e) {
      print('❌ Error initializing detector: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error loading model: $e')));
      }
    }
  }

  Future<void> _initializeVideo() async {
    try {
      _controller = VideoPlayerController.file(widget.videoFile);
      await _controller!.initialize();

      // Store video dimensions
      _videoSize = _controller!.value.size;
      if (_videoSize!.width == 0 || _videoSize!.height == 0) {
        // Fallback: use aspect ratio to estimate
        final aspectRatio = _controller!.value.aspectRatio;
        _videoSize = Size(aspectRatio * 720, 720); // Assume 720p height
      }

      // Add listener to update UI when video position changes
      _controller!.addListener(_onVideoPositionChanged);

      setState(() {});
      print('✅ Video initialized: ${_videoSize!.width}x${_videoSize!.height}');
    } catch (e) {
      print('❌ Error initializing video: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error loading video: $e')));
      }
    }
  }

  void _onVideoPositionChanged() {
    if (_controller != null && _controller!.value.isPlaying && _isProcessing) {
      // Trigger frame processing periodically
      if (_frameNumber % 5 == 0) {
        // Process every 5th frame
        _processCurrentFrame();
      }
      _frameNumber++;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Vehicle Tracker'),
        actions: [
          if (!_isModelLoaded)
            const Padding(
              padding: EdgeInsets.all(16.0),
              child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: AspectRatio(
              aspectRatio: _controller!.value.aspectRatio,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  VideoPlayer(_controller!),
                  if (_isProcessing && _videoSize != null)
                    CustomPaint(
                      painter: DetectionPainter(_currentTracks, _videoSize!),
                    ),
                ],
              ),
            ),
          ),
          _buildControls(),
          _buildStats(),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          IconButton(
            icon: Icon(
              _controller!.value.isPlaying ? Icons.pause : Icons.play_arrow,
            ),
            onPressed: () {
              setState(() {
                if (_controller!.value.isPlaying) {
                  _controller!.pause();
                  _processingTimer?.cancel();
                } else {
                  _controller!.play();
                  if (_isProcessing) {
                    // Restart processing timer
                    _processingTimer = Timer.periodic(
                      const Duration(milliseconds: 100),
                      (timer) {
                        if (_controller != null &&
                            _controller!.value.isPlaying) {
                          _processCurrentFrame();
                        } else if (_controller != null &&
                            !_controller!.value.isPlaying) {
                          timer.cancel();
                        }
                      },
                    );
                  }
                }
              });
            },
          ),
          const SizedBox(width: 16),
          ElevatedButton(
            onPressed: _isModelLoaded
                ? (_isProcessing ? _stopProcessing : _processVideo)
                : null,
            style: ElevatedButton.styleFrom(
              backgroundColor: _isProcessing ? Colors.red : Colors.green,
              foregroundColor: Colors.white,
            ),
            child: Text(_isProcessing ? 'Stop Tracking' : 'Start Tracking'),
          ),
        ],
      ),
    );
  }

  void _stopProcessing() {
    setState(() {
      _isProcessing = false;
      _processingTimer?.cancel();
      _currentTracks.clear();
    });
  }

  Widget _buildStats() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Tracked Objects: ${_currentTracks.length}'),
          ..._currentTracks.map(
            (track) => Text(
              'ID ${track.id}: ${track.detection.className} '
              '(${(track.detection.confidence * 100).toStringAsFixed(1)}%) '
              'Speed: ${track.getVelocity(30).toStringAsFixed(1)} px/s',
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _processVideo() async {
    if (!_isModelLoaded || _detector == null || _tracker == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model not loaded yet. Please wait...')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _frameNumber = 0;
    });

    // Start processing frames periodically
    _processingTimer?.cancel();
    _processingTimer = Timer.periodic(const Duration(milliseconds: 100), (
      timer,
    ) {
      if (_controller != null && _controller!.value.isPlaying) {
        _processCurrentFrame();
      } else if (_controller != null && !_controller!.value.isPlaying) {
        timer.cancel();
      }
    });

    // Process first frame immediately
    _processCurrentFrame();
  }

  Future<void> _processCurrentFrame() async {
    if (_detector == null || _tracker == null || _controller == null) return;
    if (!_controller!.value.isInitialized || _videoSize == null) return;
    if (!_isProcessing) return; // Don't process if stopped
    if (_isFrameProcessing) return; // Prevent overlapping frame processing

    _isFrameProcessing = true;

    try {
      // Get current video position in milliseconds
      final positionMs = _controller!.value.position.inMilliseconds;

      // Extract frame at current position
      final thumbnailPath = await VideoThumbnail.thumbnailFile(
        video: widget.videoFile.path,
        thumbnailPath: (await Directory.systemTemp).path,
        imageFormat: ImageFormat.PNG,
        timeMs: positionMs,
        quality: 100,
        maxWidth: _videoSize!.width.toInt(),
        maxHeight: _videoSize!.height.toInt(),
      );

      if (thumbnailPath == null) {
        print('⚠️ Failed to extract frame at ${positionMs}ms');
        return;
      }

      // Load image (guard against race where file was removed)
      final file = File(thumbnailPath);
      if (!await file.exists()) {
        print('⚠️ Thumbnail file missing: $thumbnailPath');
        return;
      }

      final imageBytes = await file.readAsBytes();
      final image = img.decodeImage(imageBytes);

      if (image == null) {
        // Clean up temp file
        try {
          await File(thumbnailPath).delete();
        } catch (_) {}
        return;
      }

      // Run detection
      final detections = _detector!.detect(image);

      // Update tracker
      final tracks = _tracker!.update(detections, _frameNumber);

      // Update UI
      if (mounted && _isProcessing) {
        setState(() {
          _currentTracks.clear();
          _currentTracks.addAll(tracks);
        });
      }

      // Clean up temp file
      try {
        await file.delete();
      } catch (_) {}
    } catch (e) {
      print('❌ Error processing frame: $e');
    } finally {
      _isFrameProcessing = false;
    }
  }

  @override
  void dispose() {
    _processingTimer?.cancel();
    _controller?.removeListener(_onVideoPositionChanged);
    _controller?.dispose();
    _detector?.dispose();
    super.dispose();
  }
}

class DetectionPainter extends CustomPainter {
  final List<TrackedObject> tracks;
  final Size videoSize;

  DetectionPainter(this.tracks, this.videoSize);

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate scale to match video aspect ratio
    final scaleX = size.width / videoSize.width;
    final scaleY = size.height / videoSize.height;

    for (var track in tracks) {
      final bbox = track.detection.bbox;

      // Scale bounding box to match displayed video size
      final scaledX = bbox.x * scaleX;
      final scaledY = bbox.y * scaleY;
      final scaledW = bbox.w * scaleX;
      final scaledH = bbox.h * scaleY;

      // Draw bounding box (green)
      final boxPaint = Paint()
        ..color = Colors.green
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      canvas.drawRect(
        Rect.fromLTWH(scaledX, scaledY, scaledW, scaledH),
        boxPaint,
      );

      // Draw label background
      final labelText =
          'ID:${track.id} ${track.detection.className} '
          '${(track.detection.confidence * 100).toStringAsFixed(0)}%';

      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      // Draw label background
      final labelBgPaint = Paint()..color = Colors.black.withOpacity(0.7);
      canvas.drawRect(
        Rect.fromLTWH(
          scaledX,
          scaledY - textPainter.height - 4,
          textPainter.width + 8,
          textPainter.height + 4,
        ),
        labelBgPaint,
      );

      // Draw label text
      textPainter.paint(
        canvas,
        Offset(scaledX + 4, scaledY - textPainter.height),
      );

      // Draw trajectory trail (purple/magenta)
      if (track.trajectory.length > 1) {
        final trailPaint = Paint()
          ..color = Colors.purple.withOpacity(0.8)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0
          ..strokeCap = StrokeCap.round;

        final path = Path();
        final firstPoint = track.trajectory.first;
        path.moveTo(firstPoint.x * scaleX, firstPoint.y * scaleY);

        for (var point in track.trajectory.skip(1)) {
          path.lineTo(point.x * scaleX, point.y * scaleY);
        }
        canvas.drawPath(path, trailPaint);
      }

      // Draw velocity arrow (red) if speed is significant
      final velocity = track.getVelocity(30); // Assuming 30 fps
      if (velocity > 5 && track.trajectory.length >= 2) {
        final centerX = (scaledX + scaledW / 2);
        final centerY = (scaledY + scaledH / 2);

        if (track.trajectory.length >= 2) {
          final last = track.trajectory.last;
          final prev = track.trajectory[track.trajectory.length - 2];

          final dx = (last.x - prev.x) * scaleX;
          final dy = (last.y - prev.y) * scaleY;

          // Normalize and scale velocity vector
          final length = math.sqrt(dx * dx + dy * dy);
          if (length > 0) {
            final scale = 20.0; // Arrow length multiplier
            final endX = centerX + (dx / length) * scale;
            final endY = centerY + (dy / length) * scale;

            final velocityPaint = Paint()
              ..color = Colors.red
              ..style = PaintingStyle.stroke
              ..strokeWidth = 2.0;

            // Draw arrow
            canvas.drawLine(
              Offset(centerX, centerY),
              Offset(endX, endY),
              velocityPaint,
            );

            // Draw arrowhead (simplified)
            final angle = math.atan2(dy, dx);
            final arrowSize = 8.0;
            final arrowPath = Path();
            arrowPath.moveTo(endX, endY);
            arrowPath.lineTo(
              endX - arrowSize * math.cos(angle + 0.5),
              endY - arrowSize * math.sin(angle + 0.5),
            );
            arrowPath.moveTo(endX, endY);
            arrowPath.lineTo(
              endX - arrowSize * math.cos(angle - 0.5),
              endY - arrowSize * math.sin(angle - 0.5),
            );
            canvas.drawPath(arrowPath, velocityPaint);
          }
        }
      }

      // Draw predicted trajectory (yellow) - simplified version
      if (track.trajectory.length >= 2) {
        final last = track.trajectory.last;
        final prev = track.trajectory[track.trajectory.length - 2];

        final vx = (last.x - prev.x);
        final vy = (last.y - prev.y);

        final predPaint = Paint()
          ..color = Colors.yellow.withOpacity(0.7)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0;

        final predPath = Path();
        predPath.moveTo(last.x * scaleX, last.y * scaleY);

        // Predict next 15 frames
        for (int i = 1; i <= 15; i++) {
          final predX = (last.x + vx * i) * scaleX;
          final predY = (last.y + vy * i) * scaleY;
          predPath.lineTo(predX, predY);
        }
        canvas.drawPath(predPath, predPaint);
      }
    }
  }

  @override
  bool shouldRepaint(DetectionPainter oldDelegate) {
    return oldDelegate.tracks.length != tracks.length ||
        oldDelegate.tracks != tracks;
  }
}
