package com.example.adas_499

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : FlutterActivity() {

    companion object {
        private const val TAG = "ADAS_Native"
        private const val METHOD_CHANNEL = "com.example.adas_499/control"
        private const val EVENT_CHANNEL  = "com.example.adas_499/detections"

        // ── Model I/O ─────────────────────────────────────────────────────────
        private const val INPUT_SIZE    = 640
        private const val CONF_THRESH   = 0.50f
        private const val IOU_THRESH    = 0.45f
        private const val PAD_VALUE     = 114
    }

    // ── Channels ──────────────────────────────────────────────────────────────
    private var eventSink: EventChannel.EventSink? = null

    // ── TFLite ────────────────────────────────────────────────────────────────
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var labels: List<String> = emptyList()
    private var numClasses = 0

    // Pre-allocated buffers (created once, reused every frame)
    private var inputBuffer: ByteBuffer? = null        // float32 NHWC
    private var outputBuffer: Array<Array<FloatArray>>? = null  // [1][4+C][8400]
    private var isFloat32Model = true

    // ── Camera ────────────────────────────────────────────────────────────────
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService

    // ── State ─────────────────────────────────────────────────────────────────
    @Volatile private var isRunning   = false
    @Volatile private var frameReady  = true   // gate: drop frames while busy

    // FPS tracking
    private var frameCount  = 0
    private var fpsWindowMs = 0L
    private var currentFps  = 0.0

    // ── Scratch bitmap (reused across frames) ─────────────────────────────────
    private var scratchBitmap: Bitmap? = null

    // ─────────────────────────────────────────────────────────────────────────
    // Flutter engine setup
    // ─────────────────────────────────────────────────────────────────────────
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // ── Method channel (control commands from Dart) ───────────────────────
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, METHOD_CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "loadModel" -> {
                        val modelPath = call.argument<String>("modelPath") ?: ""
                        val labelList = call.argument<List<String>>("labels") ?: emptyList()
                        val delegatePref = call.argument<String>("delegate") ?: "gpu"
                        try {
                            loadModel(modelPath, labelList, delegatePref)
                            result.success(true)
                        } catch (e: Exception) {
                            Log.e(TAG, "loadModel failed: ${e.message}", e)
                            result.error("LOAD_FAILED", e.message, null)
                        }
                    }
                    "startCamera" -> {
                        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                            != PackageManager.PERMISSION_GRANTED) {
                            ActivityCompat.requestPermissions(this,
                                arrayOf(Manifest.permission.CAMERA), 100)
                            result.error("NO_PERMISSION", "Camera permission not granted", null)
                        } else {
                            startCamera()
                            result.success(true)
                        }
                    }
                    "stopCamera" -> {
                        stopCamera()
                        result.success(true)
                    }
                    "dispose" -> {
                        disposeAll()
                        result.success(true)
                    }
                    else -> result.notImplemented()
                }
            }

        // ── Event channel (detection stream → Dart) ───────────────────────────
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, EVENT_CHANNEL)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, sink: EventChannel.EventSink?) {
                    eventSink = sink
                }
                override fun onCancel(arguments: Any?) {
                    eventSink = null
                }
            })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Model loading
    // ─────────────────────────────────────────────────────────────────────────
    private fun loadModel(assetPath: String, labelList: List<String>, delegatePref: String) {
        // Clean up any previous interpreter
        interpreter?.close(); interpreter = null
        gpuDelegate?.close(); gpuDelegate = null
        nnApiDelegate?.close(); nnApiDelegate = null

        labels    = labelList
        numClasses = labelList.size

        // Load model bytes from Flutter assets
        val assetManager = assets
        val modelBuffer: MappedByteBuffer = try {
            // First try with the Flutter asset path directly
            assetManager.openFd(assetPath).use { afd ->
                FileInputStream(afd.fileDescriptor).channel.map(
                    FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "openFd failed for $assetPath, trying open(): ${e.message}")
            try {
                // Fallback to open() method with Flutter path
                val inputStream = assetManager.open(assetPath)
                val tempFile = File(cacheDir, "temp_model.tflite")
                tempFile.outputStream().use { output ->
                    inputStream.copyTo(output)
                }
                FileInputStream(tempFile).channel.map(
                    FileChannel.MapMode.READ_ONLY, 0, tempFile.length()
                )
            } catch (e2: Exception) {
                Log.w(TAG, "open() failed for Flutter path $assetPath, trying Android assets path")
                // Final fallback: try removing 'assets/' prefix for Android assets
                val androidPath = assetPath.removePrefix("assets/")
                Log.i(TAG, "Trying Android assets path: $androidPath")
                assetManager.openFd(androidPath).use { afd ->
                    FileInputStream(afd.fileDescriptor).channel.map(
                        FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
                    )
                }
            }
        }

        // Build options with best available delegate
        val options = Interpreter.Options().apply {
            numThreads = 4
            when (delegatePref.lowercase()) {
                "gpu" -> {
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        val gpuOptions = compatList.bestOptionsForThisDevice
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate!!)
                        Log.i(TAG, "✅ Using GPU delegate")
                    } else {
                        // GPU not supported — fall back to NNAPI
                        Log.w(TAG, "⚠️  GPU delegate not supported, falling back to NNAPI")
                        nnApiDelegate = NnApiDelegate()
                        addDelegate(nnApiDelegate!!)
                    }
                }
                "nnapi" -> {
                    nnApiDelegate = NnApiDelegate()
                    addDelegate(nnApiDelegate!!)
                    Log.i(TAG, "✅ Using NNAPI delegate")
                }
                else -> {
                    // CPU — still use 4 threads
                    Log.i(TAG, "ℹ️  Using CPU (${numThreads} threads)")
                }
            }
        }

        interpreter = Interpreter(modelBuffer, options)

        // Inspect input tensor
        val inputTensor = interpreter!!.getInputTensor(0)
        isFloat32Model = (inputTensor.dataType() == org.tensorflow.lite.DataType.FLOAT32)
        Log.i(TAG, "Model input: ${inputTensor.shape().contentToString()}  dtype=${ if (isFloat32Model) "float32" else "uint8/int8" }")

        // Pre-allocate reusable I/O buffers
        val bytesPerChannel = if (isFloat32Model) 4 else 1
        inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * bytesPerChannel)
            .apply { order(ByteOrder.nativeOrder()) }

        // Output shape: [1, 4+numClasses, 8400]
        outputBuffer = Array(1) { Array(4 + numClasses) { FloatArray(8400) } }

        Log.i(TAG, "✅ Model loaded. Classes=$numClasses  float32=$isFloat32Model")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CameraX setup
    // ─────────────────────────────────────────────────────────────────────────
    private fun startCamera() {
        isRunning = true
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindCamera(cameraProvider!!)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera(provider: ProcessCameraProvider) {
        provider.unbindAll()

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    if (isRunning && frameReady && interpreter != null) {
                        frameReady = false
                        processFrame(imageProxy)
                    } else {
                        imageProxy.close()
                    }
                }
            }

        try {
            provider.bindToLifecycle(
                this as LifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                imageAnalysis
            )
            Log.i(TAG, "✅ Camera bound")
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed: ${e.message}", e)
        }
    }

    private fun stopCamera() {
        isRunning = false
        cameraProvider?.unbindAll()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Frame processing  (runs on cameraExecutor background thread)
    // ─────────────────────────────────────────────────────────────────────────
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val startMs = System.currentTimeMillis()

            val bitmap = imageProxyToBitmap(imageProxy)
            val (letterboxed, scale, padLeft, padTop) = letterbox(bitmap)

            fillInputBuffer(letterboxed)

            val outBuf = outputBuffer!!
            // Zero previous results
            for (row in outBuf[0]) row.fill(0f)

            interpreter!!.run(inputBuffer!!, outBuf)

            val detections = decodeOutput(outBuf[0], scale, padLeft, padTop,
                bitmap.width, bitmap.height)

            val inferMs = System.currentTimeMillis() - startMs

            // FPS
            frameCount++
            fpsWindowMs += inferMs
            if (fpsWindowMs >= 1000) {
                currentFps = frameCount * 1000.0 / fpsWindowMs
                frameCount = 0
                fpsWindowMs = 0
            }

            // Send to Flutter on main thread
            val payload = buildPayload(detections, inferMs)
            runOnUiThread { eventSink?.success(payload) }

        } catch (e: Exception) {
            Log.e(TAG, "processFrame error: ${e.message}", e)
            runOnUiThread { eventSink?.error("INFER_ERROR", e.message, null) }
        } finally {
            imageProxy.close()
            frameReady = true
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // YUV → Bitmap  (native fast path via YuvImage)
    // ─────────────────────────────────────────────────────────────────────────
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yPlane  = imageProxy.planes[0]
        val uPlane  = imageProxy.planes[1]
        val vPlane  = imageProxy.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // NV21 = Y plane + interleaved VU
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        // NV21 requires VU interleaving
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21,
            imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
        val jpegBytes = out.toByteArray()
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)

        // Rotate to upright if needed (camera sensor is often landscape)
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        return if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else bitmap
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Letterbox  (matches Ultralytics letterbox with pad=114)
    // ─────────────────────────────────────────────────────────────────────────
    data class LetterboxResult(
        val bitmap: Bitmap,
        val scale: Float,
        val padLeft: Int,
        val padTop: Int
    )

    private fun letterbox(src: Bitmap): LetterboxResult {
        val scale = INPUT_SIZE.toFloat() / max(src.width, src.height)
        val newW = (src.width  * scale).toInt()
        val newH = (src.height * scale).toInt()

        val padLeft = (INPUT_SIZE - newW) / 2
        val padTop  = (INPUT_SIZE - newH) / 2

        // Create 640×640 canvas filled with grey (114)
        val canvas640 = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val androidCanvas = android.graphics.Canvas(canvas640)
        androidCanvas.drawColor(android.graphics.Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE))

        // Resize source
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        androidCanvas.drawBitmap(resized, padLeft.toFloat(), padTop.toFloat(), null)
        if (resized != src) resized.recycle()

        return LetterboxResult(canvas640, scale, padLeft, padTop)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fill TFLite input buffer from letterboxed bitmap
    // ─────────────────────────────────────────────────────────────────────────
    private fun fillInputBuffer(bitmap: Bitmap) {
        val buf = inputBuffer!!
        buf.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        if (isFloat32Model) {
            for (px in pixels) {
                buf.putFloat(((px shr 16) and 0xFF) / 255.0f)  // R
                buf.putFloat(((px shr  8) and 0xFF) / 255.0f)  // G
                buf.putFloat(( px         and 0xFF) / 255.0f)  // B
            }
        } else {
            // INT8 / UINT8 — write raw byte values
            for (px in pixels) {
                buf.put(((px shr 16) and 0xFF).toByte())  // R
                buf.put(((px shr  8) and 0xFF).toByte())  // G
                buf.put(( px         and 0xFF).toByte())  // B
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Decode YOLO output  →  List of detection maps
    //
    // Output layout: [4+numClasses, 8400]
    //   rows 0-3 : cx, cy, w, h  (normalised to [0,1] relative to 640×640)
    //   rows 4+  : class scores
    // ─────────────────────────────────────────────────────────────────────────
    private data class RawDet(
        val left: Float, val top: Float,
        val right: Float, val bottom: Float,
        val classIdx: Int, val score: Float
    )

    private fun decodeOutput(
        output: Array<FloatArray>,   // [4+C][8400]
        scale: Float,
        padLeft: Int, padTop: Int,
        origW: Int, origH: Int
    ): List<Map<String, Any>> {

        val numAnchors = output[0].size  // 8400
        val rawDets = mutableListOf<RawDet>()

        for (a in 0 until numAnchors) {
            // Find best class
            var bestScore = 0f
            var bestClass = 0
            for (c in 0 until numClasses) {
                val s = output[4 + c][a]
                if (s > bestScore) { bestScore = s; bestClass = c }
            }
            if (bestScore < CONF_THRESH) continue

            // cx/cy/w/h normalised → pixel space in 640×640
            val cx = output[0][a] * INPUT_SIZE
            val cy = output[1][a] * INPUT_SIZE
            val w  = output[2][a] * INPUT_SIZE
            val h  = output[3][a] * INPUT_SIZE

            val lbX1 = cx - w / 2f
            val lbY1 = cy - h / 2f
            val lbX2 = cx + w / 2f
            val lbY2 = cy + h / 2f

            // Remove padding + undo scale → normalised [0,1]
            val left   = (lbX1 - padLeft) / (scale * origW)
            val top    = (lbY1 - padTop)  / (scale * origH)
            val right  = (lbX2 - padLeft) / (scale * origW)
            val bottom = (lbY2 - padTop)  / (scale * origH)

            if (!isValidBox(left, top, right, bottom)) continue
            rawDets.add(RawDet(left, top, right, bottom, bestClass, bestScore))
        }

        return nms(rawDets).map { d ->
            mapOf(
                "label"      to (if (d.classIdx < labels.size) labels[d.classIdx] else "class_${d.classIdx}"),
                "confidence" to d.score.toDouble(),
                "left"       to d.left.toDouble().coerceIn(0.0, 1.0),
                "top"        to d.top.toDouble().coerceIn(0.0, 1.0),
                "right"      to d.right.toDouble().coerceIn(0.0, 1.0),
                "bottom"     to d.bottom.toDouble().coerceIn(0.0, 1.0)
            )
        }
    }

    private fun isValidBox(l: Float, t: Float, r: Float, b: Float): Boolean {
        if (!l.isFinite() || !t.isFinite() || !r.isFinite() || !b.isFinite()) return false
        val cl = l.coerceIn(0f, 1f); val ct = t.coerceIn(0f, 1f)
        val cr = r.coerceIn(0f, 1f); val cb = b.coerceIn(0f, 1f)
        val w = cr - cl; val h = cb - ct
        if (w <= 0 || h <= 0 || w < 0.03f || h < 0.03f) return false
        val area = w * h
        return area >= 0.002f && area <= 0.90f
    }

    private fun nms(dets: List<RawDet>): List<RawDet> {
        if (dets.isEmpty()) return emptyList()
        val byClass = dets.groupBy { it.classIdx }
        val results = mutableListOf<RawDet>()
        for ((_, group) in byClass) {
            val sorted = group.sortedByDescending { it.score }
            val kept = mutableListOf<RawDet>()
            for (det in sorted) {
                if (kept.none { iou(det, it) > IOU_THRESH }) kept.add(det)
            }
            results.addAll(kept)
        }
        return results
    }

    private fun iou(a: RawDet, b: RawDet): Float {
        val iL = max(a.left,  b.left);  val iT = max(a.top,    b.top)
        val iR = min(a.right, b.right); val iB = min(a.bottom, b.bottom)
        if (iR <= iL || iB <= iT) return 0f
        val inter = (iR - iL) * (iB - iT)
        val union = (a.right-a.left)*(a.bottom-a.top) + (b.right-b.left)*(b.bottom-b.top) - inter
        return if (union == 0f) 0f else inter / union
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Build event payload for Dart
    // ─────────────────────────────────────────────────────────────────────────
    private fun buildPayload(
        detections: List<Map<String, Any>>,
        inferMs: Long
    ): Map<String, Any> = mapOf(
        "detections" to detections,
        "inferMs"    to inferMs,
        "fps"        to currentFps
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────
    private fun disposeAll() {
        stopCamera()
        interpreter?.close(); interpreter = null
        gpuDelegate?.close(); gpuDelegate = null
        nnApiDelegate?.close(); nnApiDelegate = null
        cameraExecutor.shutdown()
        Log.i(TAG, "Disposed all native resources")
    }

    override fun onDestroy() {
        disposeAll()
        super.onDestroy()
    }
}
