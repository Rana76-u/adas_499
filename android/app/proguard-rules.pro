# ── TFLite ────────────────────────────────────────────────────────────────────
# Keep all TFLite interpreter and delegate classes intact so R8 does not strip
# JNI entry points referenced only from native code.
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }
-dontwarn org.tensorflow.lite.**

# ── CameraX ───────────────────────────────────────────────────────────────────
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# ── Flutter embedding ─────────────────────────────────────────────────────────
-keep class io.flutter.** { *; }
-dontwarn io.flutter.**
