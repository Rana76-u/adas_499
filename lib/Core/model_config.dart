import 'yolo_model.dart';

/// Configuration for a TFLite model with its associated label set and delegate.
class ModelConfig {
  final String name;
  final String modelPath;
  final List<String> labels;
  final String delegate;

  const ModelConfig({
    required this.name,
    required this.modelPath,
    required this.labels,
    this.delegate = 'gpu',
  });
}

/// List of available models for selection.
/// Add new models here as they become available in assets/models/.
final List<ModelConfig> availableModels = [
  ModelConfig(
      name: 'YOLOv9t_best',
      modelPath: 'assets/models/best.tflite',
      labels: customLabels,
      delegate: 'gpu',
    ),
    ModelConfig(
      name: 'YOLOv8n',
      modelPath: 'assets/models/best_float32_kaggle.tflite',
      labels: customLabels,
      delegate: 'gpu',
    ),

    
  // Additional models can be added here when available:
  // ModelConfig(
  //   name: 'YOLO11n INT8',
  //   modelPath: 'assets/models/yolo11n_int8.tflite',
  //   labels: customLabels,
  //   delegate: 'nnapi',
  // ),
  // ModelConfig(
  //   name: 'COCO YOLO',
  //   modelPath: 'assets/models/coco_model.tflite',
  //   labels: cocoLabels,
  //   delegate: 'gpu',
  // ),
];
