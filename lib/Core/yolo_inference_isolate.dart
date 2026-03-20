// This file has been intentionally retired.
//
// YUV conversion, letterboxing, TFLite inference, and NMS now run entirely
// in native Kotlin (MainActivity.kt) via CameraX + TFLite GPU/NNAPI delegate.
//
// The Dart-side entry point is:
//   lib/Core/native_detection_bridge.dart  →  NativeDetectionBridge
//
// Do NOT import this file. It will be removed in a future cleanup.
