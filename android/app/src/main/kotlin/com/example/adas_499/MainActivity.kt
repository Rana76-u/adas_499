package com.example.adas_499

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import android.view.Surface
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : FlutterActivity() {

    companion object {
        private const val TAG             = "ADAS_Native"
        private const val METHOD_CHANNEL  = "com.example.adas_499/control"
        private const val EVENT_CHANNEL   = "com.example.adas_499/detections"
        private const val PREVIEW_CHANNEL = "com.example.adas_499/preview"

        private const val INPUT_SIZE   = 640
        private const val CONF_THRESH  = 0.40f   // slightly lower → catch more objects
        private const val IOU_THRESH   = 0.45f
        private const val PAD_VALUE    = 114
        private const val JPEG_QUALITY = 85      // used nowhere now — kept for reference
    }

    // ── Channels ──────────────────────────────────────────────────────────────
    private var eventSink: EventChannel.EventSink? = null

    // ── TFLite ────────────────────────────────────────────────────────────────
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate?   = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var labels: List<String>         = emptyList()
    private var numClasses                   = 0
    private var isFloat32Model               = true

    // Pre-allocated buffers — never reallocated after loadModel()
    private var inputBuffer: ByteBuffer?              = null   // NHWC float32 or uint8
    private var outputBuffer: Array<Array<FloatArray>>? = null // [1][4+C][8400]

    // Reusable bitmaps — allocated once, reused every frame
    private var letterboxBitmap: Bitmap? = null   // 640×640 scratchpad
    private var pixelScratch:    IntArray? = null  // INPUT_SIZE*INPUT_SIZE ints

    // ── Camera ────────────────────────────────────────────────────────────────
    private var cameraProvider: ProcessCameraProvider? = null
    private var previewSurfaceProvider: Preview.SurfaceProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private var flutterSurfaceTextureEntry: io.flutter.view.TextureRegistry.SurfaceTextureEntry? = null
    private var previewUseCase: Preview? = null

    // ── State ─────────────────────────────────────────────────────────────────
    @Volatile private var isRunning  = false
    @Volatile private var frameReady = true   // simple frame-drop gate

    // FPS: exponential moving average — stable and cheap
    private var lastFrameMs: Long = 0L
    private var emaFps: Double    = 0.0
    private val EMA_ALPHA         = 0.1   // smoothing factor

    // ─────────────────────────────────────────────────────────────────────────
    // Flutter engine setup
    // ─────────────────────────────────────────────────────────────────────────
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Register a Flutter texture for the camera preview
        flutterSurfaceTextureEntry = flutterEngine.renderer.createSurfaceTexture()
        val surfaceTexture = flutterSurfaceTextureEntry!!.surfaceTexture()

        // Tell CameraX to render to this surface texture (set resolution lazily)
        previewSurfaceProvider = Preview.SurfaceProvider { request ->
            val surface = Surface(surfaceTexture)
            surfaceTexture.setDefaultBufferSize(
                request.resolution.width, request.resolution.height
            )
            request.provideSurface(surface, ContextCompat.getMainExecutor(this)) {}
        }

        // ── Method channel ────────────────────────────────────────────────────
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, METHOD_CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "loadModel" -> {
                        val modelPath    = call.argument<String>("modelPath") ?: ""
                        val labelList    = call.argument<List<String>>("labels") ?: emptyList()
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
                            ActivityCompat.requestPermissions(
                                this, arrayOf(Manifest.permission.CAMERA), 100)
                            result.error("NO_PERMISSION", "Camera permission not granted", null)
                        } else {
                            startCamera()
                            // Return the Flutter texture ID so Dart can show Texture(id)
                            result.success(flutterSurfaceTextureEntry!!.id())
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
        // Release any previous session
        interpreter?.close();   interpreter   = null
        gpuDelegate?.close();   gpuDelegate   = null
        nnApiDelegate?.close(); nnApiDelegate = null

        labels     = labelList
        numClasses = labelList.size

        // ── Load model bytes from Flutter assets ──────────────────────────────
        val modelBuffer = loadModelBuffer(assetPath)

        // ── Build interpreter options with best available delegate ────────────
        val options = Interpreter.Options().apply {
            numThreads = 4
            when (delegatePref.lowercase()) {
                "gpu" -> {
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate!!)
                        Log.i(TAG, "✅ GPU delegate active")
                    } else {
                        Log.w(TAG, "⚠️ GPU not supported → falling back to NNAPI")
                        nnApiDelegate = NnApiDelegate()
                        addDelegate(nnApiDelegate!!)
                    }
                }
                "nnapi" -> {
                    nnApiDelegate = NnApiDelegate()
                    addDelegate(nnApiDelegate!!)
                    Log.i(TAG, "✅ NNAPI delegate active")
                }
                else -> Log.i(TAG, "ℹ️ CPU path with $numThreads threads")
            }
        }

        interpreter = Interpreter(modelBuffer, options)

        // ── Inspect input tensor ──────────────────────────────────────────────
        val inputTensor = interpreter!!.getInputTensor(0)
        isFloat32Model  = (inputTensor.dataType() == DataType.FLOAT32)
        Log.i(TAG, "Input shape: ${inputTensor.shape().contentToString()}  " +
            "dtype=${if (isFloat32Model) "float32" else "int8/uint8"}")

        // ── Pre-allocate reusable I/O buffers (ONCE, never again) ─────────────
        val bytesPerElem = if (isFloat32Model) 4 else 1
        inputBuffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * bytesPerElem)
            .apply { order(ByteOrder.nativeOrder()) }

        // YOLO output: [1, 4+numClasses, 8400]
        outputBuffer = Array(1) { Array(4 + numClasses) { FloatArray(8400) } }

        // Pre-allocate scratch bitmaps so letterboxing never allocs at runtime
        letterboxBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        pixelScratch    = IntArray(INPUT_SIZE * INPUT_SIZE)

        Log.i(TAG, "✅ Model ready. classes=$numClasses  float32=$isFloat32Model")
    }

    /** Try several paths to locate the model in Android's asset system. */
    private fun loadModelBuffer(assetPath: String): java.nio.MappedByteBuffer {
        // Path 1: Flutter asset path as-is (e.g. "assets/models/foo.tflite")
        try {
            return assets.openFd(assetPath).use { afd ->
                FileInputStream(afd.fileDescriptor).channel.map(
                    FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
                )
            }
        } catch (_: Exception) { Log.w(TAG, "openFd failed for $assetPath, trying stripped path") }

        // Path 2: Strip leading "assets/" — Android merges Flutter assets into root
        val androidPath = assetPath.removePrefix("assets/")
        try {
            return assets.openFd(androidPath).use { afd ->
                FileInputStream(afd.fileDescriptor).channel.map(
                    FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
                )
            }
        } catch (_: Exception) { Log.w(TAG, "openFd failed for $androidPath, using stream copy") }

        // Path 3: Stream copy to cache (slowest but always works)
        val tmpFile = java.io.File(cacheDir, "model_tmp.tflite")
        assets.open(androidPath).use { it.copyTo(tmpFile.outputStream()) }
        return FileInputStream(tmpFile).channel.map(
            FileChannel.MapMode.READ_ONLY, 0, tmpFile.length()
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CameraX
    // ─────────────────────────────────────────────────────────────────────────
    private fun startCamera() {
        isRunning = true
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera(cameraProvider!!)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera(provider: ProcessCameraProvider) {
        provider.unbindAll()

        // ── Preview use-case → renders to the Flutter texture ─────────────────
        previewUseCase = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()
            .also { it.setSurfaceProvider(previewSurfaceProvider) }

        // ── Analysis use-case → runs inference ───────────────────────────────
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
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
                previewUseCase!!,
                imageAnalysis
            )
            Log.i(TAG, "✅ Camera bound (preview + analysis)")
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed: ${e.message}", e)
        }
    }

    private fun stopCamera() {
        isRunning = false
        cameraProvider?.unbindAll()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Frame processing  (runs on dedicated cameraExecutor thread)
    // ─────────────────────────────────────────────────────────────────────────
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val t0 = System.currentTimeMillis()

            // ── 1. YUV → RGB bitmap (zero JPEG roundtrip) ─────────────────────
            val rgbBitmap = yuvToRgbBitmap(imageProxy)

            // ── 2. Letterbox into pre-allocated 640×640 scratch bitmap ─────────
            val (scale, padLeft, padTop) = letterboxInPlace(rgbBitmap)

            // ── 3. Fill TFLite input buffer from scratch bitmap ────────────────
            fillInputBuffer(letterboxBitmap!!)

            // ── 4. Run inference ──────────────────────────────────────────────
            val out = outputBuffer!!
            for (row in out[0]) row.fill(0f)          // zero output in-place
            interpreter!!.run(inputBuffer!!, out)

            // ── 5. Decode + NMS ───────────────────────────────────────────────
            val dets = decodeOutput(
                out[0], scale, padLeft, padTop,
                rgbBitmap.width, rgbBitmap.height
            )

            val inferMs = System.currentTimeMillis() - t0

            // ── 6. EMA FPS ────────────────────────────────────────────────────
            val nowMs = System.currentTimeMillis()
            if (lastFrameMs > 0L) {
                val instantFps = 1000.0 / (nowMs - lastFrameMs).toDouble().coerceAtLeast(1.0)
                emaFps = if (emaFps == 0.0) instantFps
                         else EMA_ALPHA * instantFps + (1 - EMA_ALPHA) * emaFps
            }
            lastFrameMs = nowMs

            // ── 7. Send to Flutter on main thread ─────────────────────────────
            val payload = mapOf(
                "detections" to dets,
                "inferMs"    to inferMs,
                "fps"        to emaFps
            )
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
    // YUV_420_888 → RGB Bitmap  (direct, no JPEG roundtrip)
    //
    // Handles the common planar YUV case (I420 and NV12/NV21 variants).
    // Falls back to a single-pixel slow path only if strides are unusual.
    // ─────────────────────────────────────────────────────────────────────────
    private fun yuvToRgbBitmap(imageProxy: ImageProxy): Bitmap {
        val width  = imageProxy.width
        val height = imageProxy.height

        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride   // 1 = planar I420, 2 = semi-planar NV12/NV21

        val yData  = ByteArray(yBuf.remaining()).also { yBuf.get(it) }
        val uData  = ByteArray(uBuf.remaining()).also { uBuf.get(it) }
        val vData  = ByteArray(vBuf.remaining()).also { vBuf.get(it) }

        val pixels = IntArray(width * height)

        for (row in 0 until height) {
            val uvRow = row shr 1
            for (col in 0 until width) {
                val uvCol = col shr 1

                val yIdx  = row * yRowStride + col
                val uvIdx = uvRow * uvRowStride + uvCol * uvPixelStride

                val y = (yData[yIdx].toInt() and 0xFF)
                val u = (uData[uvIdx].toInt() and 0xFF) - 128
                val v = (vData[uvIdx].toInt() and 0xFF) - 128

                // BT.601 integer approximation
                val r = (y + 1.402f * v).toInt().coerceIn(0, 255)
                val g = (y - 0.344f * u - 0.714f * v).toInt().coerceIn(0, 255)
                val b = (y + 1.772f * u).toInt().coerceIn(0, 255)

                pixels[row * width + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

        // Rotate to upright orientation
        val rotation = imageProxy.imageInfo.rotationDegrees
        return if (rotation != 0) {
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
            bitmap.recycle()
            rotated
        } else {
            bitmap
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Letterbox — draws into the pre-allocated 640×640 bitmap in-place.
    // Returns (scale, padLeft, padTop) for bounding-box unscaling.
    // ─────────────────────────────────────────────────────────────────────────
    private data class LetterboxMeta(val scale: Float, val padLeft: Int, val padTop: Int)

    private fun letterboxInPlace(src: Bitmap): LetterboxMeta {
        val scale  = INPUT_SIZE.toFloat() / max(src.width, src.height)
        val newW   = (src.width  * scale).toInt()
        val newH   = (src.height * scale).toInt()
        val padLeft = (INPUT_SIZE - newW) / 2
        val padTop  = (INPUT_SIZE - newH) / 2

        val canvas = android.graphics.Canvas(letterboxBitmap!!)
        // Fill padding area with grey (114,114,114)
        canvas.drawColor(android.graphics.Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE))
        // Draw scaled source
        val dst = android.graphics.RectF(
            padLeft.toFloat(), padTop.toFloat(),
            (padLeft + newW).toFloat(), (padTop + newH).toFloat()
        )
        canvas.drawBitmap(src, null, dst, null)

        return LetterboxMeta(scale, padLeft, padTop)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fill TFLite input buffer from the 640×640 letterboxed bitmap.
    //
    // Uses getPixels() into a pre-allocated IntArray then byte-shifts —
    // fastest approach without JNI for this buffer size.
    // ─────────────────────────────────────────────────────────────────────────
    private fun fillInputBuffer(bitmap: Bitmap) {
        val buf     = inputBuffer!!
        val scratch = pixelScratch!!
        buf.rewind()

        bitmap.getPixels(scratch, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        if (isFloat32Model) {
            val inv255 = 1f / 255f
            for (px in scratch) {
                buf.putFloat(((px shr 16) and 0xFF) * inv255)   // R
                buf.putFloat(((px shr  8) and 0xFF) * inv255)   // G
                buf.putFloat(( px         and 0xFF) * inv255)   // B
            }
        } else {
            // INT8 quantised model — write raw bytes
            for (px in scratch) {
                buf.put(((px shr 16) and 0xFF).toByte())   // R
                buf.put(((px shr  8) and 0xFF).toByte())   // G
                buf.put(( px         and 0xFF).toByte())   // B
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Decode YOLO output + NMS
    //
    // Output tensor: [1, 4+numClasses, 8400]
    //   Rows 0–3 : cx, cy, w, h  (normalised to INPUT_SIZE space)
    //   Rows 4+  : per-class confidence scores
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

        val numAnchors = output[0].size   // 8400 for 640-input YOLO
        val rawDets    = ArrayList<RawDet>(64)

        for (a in 0 until numAnchors) {
            // Find highest-confidence class
            var bestScore = CONF_THRESH   // early-exit: only enter loop if we already pass threshold
            var bestClass = -1
            for (c in 0 until numClasses) {
                val s = output[4 + c][a]
                if (s > bestScore) { bestScore = s; bestClass = c }
            }
            if (bestClass == -1) continue

            // cx/cy/w/h in INPUT_SIZE pixel space
            val cx = output[0][a] * INPUT_SIZE
            val cy = output[1][a] * INPUT_SIZE
            val bw = output[2][a] * INPUT_SIZE
            val bh = output[3][a] * INPUT_SIZE

            val lbX1 = cx - bw / 2f
            val lbY1 = cy - bh / 2f
            val lbX2 = cx + bw / 2f
            val lbY2 = cy + bh / 2f

            // Undo letterbox → normalised [0,1] relative to original frame
            val left   = (lbX1 - padLeft) / (scale * origW)
            val top    = (lbY1 - padTop)  / (scale * origH)
            val right  = (lbX2 - padLeft) / (scale * origW)
            val bottom = (lbY2 - padTop)  / (scale * origH)

            if (!isValidBox(left, top, right, bottom)) continue
            rawDets.add(RawDet(left, top, right, bottom, bestClass, bestScore))
        }

        return nms(rawDets).map { d ->
            mapOf(
                "label"      to (labels.getOrElse(d.classIdx) { "class_${d.classIdx}" }),
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
        val w = cr - cl;  val h = cb - ct
        if (w <= 0f || h <= 0f || w < 0.01f || h < 0.01f) return false
        val area = w * h
        return area >= 0.0005f && area <= 0.95f
    }

    private fun nms(dets: ArrayList<RawDet>): List<RawDet> {
        if (dets.isEmpty()) return emptyList()
        val results = ArrayList<RawDet>(dets.size)
        val byClass = dets.groupBy { it.classIdx }
        for ((_, group) in byClass) {
            val sorted = group.sortedByDescending { it.score }
            val kept   = ArrayList<RawDet>(sorted.size)
            for (det in sorted) {
                if (kept.none { iou(det, it) > IOU_THRESH }) kept.add(det)
            }
            results.addAll(kept)
        }
        return results
    }

    private fun iou(a: RawDet, b: RawDet): Float {
        val iL = max(a.left, b.left);   val iT = max(a.top, b.top)
        val iR = min(a.right, b.right); val iB = min(a.bottom, b.bottom)
        if (iR <= iL || iB <= iT) return 0f
        val inter = (iR - iL) * (iB - iT)
        val union  = (a.right-a.left)*(a.bottom-a.top) +
                     (b.right-b.left)*(b.bottom-b.top) - inter
        return if (union <= 0f) 0f else inter / union
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────
    private fun disposeAll() {
        stopCamera()
        interpreter?.close();    interpreter    = null
        gpuDelegate?.close();    gpuDelegate    = null
        nnApiDelegate?.close();  nnApiDelegate  = null
        letterboxBitmap?.recycle(); letterboxBitmap = null
        flutterSurfaceTextureEntry?.release()
        if (::cameraExecutor.isInitialized) cameraExecutor.shutdown()
        Log.i(TAG, "All native resources disposed")
    }

    override fun onDestroy() {
        disposeAll()
        super.onDestroy()
    }
}
