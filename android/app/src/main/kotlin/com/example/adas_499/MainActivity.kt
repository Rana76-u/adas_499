package com.example.adas_499

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
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
        private const val TAG                  = "ADAS_Native"
        private const val METHOD_CHANNEL       = "com.example.adas_499/control"
        private const val EVENT_CHANNEL        = "com.example.adas_499/detections"

        private const val INPUT_SIZE           = 640
        private const val CONF_THRESH          = 0.40f
        private const val IOU_THRESH           = 0.45f
        private const val PAD_VALUE            = 114

        // Throttle event-channel sends — inference still runs every frame.
        private const val MIN_SEND_INTERVAL_MS = 80L

        // EMA smoothing factor for FPS — must be in companion to use const val
        private const val EMA_ALPHA            = 0.1
    }

    // ── Channels ──────────────────────────────────────────────────────────────
    private var eventSink: EventChannel.EventSink? = null

    // ── TFLite ────────────────────────────────────────────────────────────────
    private var interpreter: Interpreter?     = null
    private var gpuDelegate: GpuDelegate?     = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var labels: List<String>          = emptyList()
    private var numClasses                    = 0
    private var isFloat32Model                = true

    // Pre-allocated I/O buffers (never reallocated after loadModel)
    private var inputBuffer:  ByteBuffer?               = null
    private var outputBuffer: Array<Array<FloatArray>>? = null

    // Reusable bitmaps — eliminate per-frame GC pressure
    private var letterboxBitmap: Bitmap?   = null
    private var pixelScratch:    IntArray? = null
    private var rgbScratch:      Bitmap?   = null

    // ── Camera ────────────────────────────────────────────────────────────────
    private var cameraProvider: ProcessCameraProvider? = null
    private var previewSurfaceProvider: Preview.SurfaceProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private var flutterSurfaceTextureEntry: io.flutter.view.TextureRegistry.SurfaceTextureEntry? = null
    private var previewUseCase: Preview? = null

    // ── State ─────────────────────────────────────────────────────────────────
    @Volatile private var isRunning  = false
    @Volatile private var frameReady = true

    private var lastFrameMs: Long = 0L
    private var emaFps: Double    = 0.0   // mutable — not const
    private var lastSendMs: Long  = 0L

    /** IoU + class association; mirrors notebook DeepSORT-style max_age / n_init. */
    private val objectTracker = ObjectTracker(maxAge = 30, nInit = 3)

    // Pre-allocated letterbox canvas and paint objects (no per-frame allocation)
    private var lbCanvas: android.graphics.Canvas? = null
    private val lbPaint  = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
    private val lbDst    = android.graphics.RectF()
    private val padColor = android.graphics.Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE)

    // ─────────────────────────────────────────────────────────────────────────
    // Flutter engine setup
    // ─────────────────────────────────────────────────────────────────────────
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        cameraExecutor = Executors.newSingleThreadExecutor()

        flutterSurfaceTextureEntry = flutterEngine.renderer.createSurfaceTexture()
        val surfaceTexture = flutterSurfaceTextureEntry!!.surfaceTexture()

        previewSurfaceProvider = Preview.SurfaceProvider { request ->
            val surface = Surface(surfaceTexture)
            surfaceTexture.setDefaultBufferSize(
                request.resolution.width, request.resolution.height)
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
                            result.success(flutterSurfaceTextureEntry!!.id())
                        }
                    }
                    "stopCamera" -> { stopCamera(); result.success(true) }
                    "dispose"    -> { disposeAll(); result.success(true) }

                    // Still-image inference — offloaded to cameraExecutor
                    "runOnImage" -> {
                        val path = call.argument<String>("path")
                        if (path == null) {
                            result.error("BAD_ARG", "path is null", null)
                        } else {
                            cameraExecutor.execute {
                                try {
                                    val dets = runInferenceOnImagePath(path)
                                    runOnUiThread { result.success(dets) }
                                } catch (e: Exception) {
                                    Log.e(TAG, "runOnImage error: ${e.message}", e)
                                    runOnUiThread { result.error("INFER_ERROR", e.message, null) }
                                }
                            }
                        }
                    }

                    else -> result.notImplemented()
                }
            }

        // ── Event channel ─────────────────────────────────────────────────────
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, EVENT_CHANNEL)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, sink: EventChannel.EventSink?) { eventSink = sink }
                override fun onCancel(arguments: Any?) { eventSink = null }
            })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Model loading
    // ─────────────────────────────────────────────────────────────────────────
    private fun loadModel(assetPath: String, labelList: List<String>, delegatePref: String) {
        interpreter?.close();    interpreter   = null
        gpuDelegate?.close();    gpuDelegate   = null
        nnApiDelegate?.close();  nnApiDelegate = null

        labels     = labelList
        numClasses = labelList.size

        val modelBuffer = loadModelBuffer(assetPath)

        val options = Interpreter.Options().apply {
            numThreads = 2   // 2 > 4 on mobile; GPU/NNAPI ignores this anyway
            when (delegatePref.lowercase()) {
                "gpu" -> {
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        // Use plain GpuDelegate() — bestOptionsForThisDevice returns
                        // GpuDelegateFactory.Options which is not on the classpath
                        // with tensorflow-lite-gpu:2.x without the delegate-plugin AAR.
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate!!)
                        Log.i(TAG, "✅ GPU delegate active")
                    } else {
                        Log.w(TAG, "⚠️ GPU not supported → NNAPI fallback")
                        nnApiDelegate = NnApiDelegate()
                        addDelegate(nnApiDelegate!!)
                    }
                }
                "nnapi" -> {
                    nnApiDelegate = NnApiDelegate()
                    addDelegate(nnApiDelegate!!)
                    Log.i(TAG, "✅ NNAPI delegate active")
                }
                else -> Log.i(TAG, "ℹ️ CPU with 2 threads")
            }
        }

        interpreter = Interpreter(modelBuffer, options)

        val inputTensor = interpreter!!.getInputTensor(0)
        isFloat32Model  = (inputTensor.dataType() == DataType.FLOAT32)
        Log.i(TAG, "Input shape: ${inputTensor.shape().contentToString()}  " +
            "dtype=${if (isFloat32Model) "float32" else "int8/uint8"}")

        val bytesPerElem = if (isFloat32Model) 4 else 1
        inputBuffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * bytesPerElem)
            .apply { order(ByteOrder.nativeOrder()) }

        outputBuffer = Array(1) { Array(4 + numClasses) { FloatArray(8400) } }

        letterboxBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        lbCanvas        = android.graphics.Canvas(letterboxBitmap!!)
        pixelScratch    = IntArray(INPUT_SIZE * INPUT_SIZE)

        Log.i(TAG, "✅ Model ready. classes=$numClasses  float32=$isFloat32Model")
    }

    private fun loadModelBuffer(assetPath: String): java.nio.MappedByteBuffer {
        try {
            return assets.openFd(assetPath).use { afd ->
                FileInputStream(afd.fileDescriptor).channel.map(
                    FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        } catch (_: Exception) {}

        val androidPath = assetPath.removePrefix("assets/")
        try {
            return assets.openFd(androidPath).use { afd ->
                FileInputStream(afd.fileDescriptor).channel.map(
                    FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        } catch (_: Exception) {}

        val tmpFile = java.io.File(cacheDir, "model_tmp.tflite")
        assets.open(androidPath).use { it.copyTo(tmpFile.outputStream()) }
        return FileInputStream(tmpFile).channel.map(
            FileChannel.MapMode.READ_ONLY, 0, tmpFile.length())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CameraX
    // ─────────────────────────────────────────────────────────────────────────
    private fun startCamera() {
        objectTracker.reset()
        isRunning = true
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera(cameraProvider!!)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera(provider: ProcessCameraProvider) {
        provider.unbindAll()

        previewUseCase = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()
            .also { it.setSurfaceProvider(previewSurfaceProvider) }

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
                imageAnalysis)
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
    // Frame processing  (runs on dedicated cameraExecutor thread)
    // ─────────────────────────────────────────────────────────────────────────
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val t0 = System.currentTimeMillis()

            // 1. YUV → RGB (reuses scratchpad bitmap)
            yuvToRgbInPlace(imageProxy)

            // 2. Letterbox into pre-allocated 640×640 (rotation baked into draw)
            val rotation = imageProxy.imageInfo.rotationDegrees
            val (scale, padLeft, padTop) = letterboxBitmapInPlace(rgbScratch!!, rotation)

            // 3. Fill TFLite input buffer
            fillInputBuffer(letterboxBitmap!!)

            // 4. Run inference
            val out = outputBuffer!!
            for (row in out[0]) row.fill(0f)
            interpreter!!.run(inputBuffer!!, out)

            // 5. Determine original display dimensions after rotation
            val srcW = if (rotation % 180 == 0) imageProxy.width  else imageProxy.height
            val srcH = if (rotation % 180 == 0) imageProxy.height else imageProxy.width

            // 6. Decode + NMS + multi-object tracking
            val boxes   = decodeOutput(out[0], scale, padLeft, padTop, srcW, srcH)
            val tracked = objectTracker.update(boxes, emaFps.coerceAtLeast(1.0))
            val dets    = tracked.map { it.toFlutterMap() }
            val inferMs = System.currentTimeMillis() - t0

            // 7. EMA FPS
            val nowMs = System.currentTimeMillis()
            if (lastFrameMs > 0L) {
                val instantFps = 1000.0 / (nowMs - lastFrameMs).toDouble().coerceAtLeast(1.0)
                emaFps = if (emaFps == 0.0) instantFps
                         else EMA_ALPHA * instantFps + (1.0 - EMA_ALPHA) * emaFps
            }
            lastFrameMs = nowMs

            // 8. Throttled send — avoids flooding Dart event loop
            if (nowMs - lastSendMs >= MIN_SEND_INTERVAL_MS) {
                lastSendMs = nowMs
                val payload = mapOf(
                    "detections" to dets,
                    "inferMs"    to inferMs,
                    "fps"        to emaFps)
                runOnUiThread { eventSink?.success(payload) }
            }

        } catch (e: Exception) {
            Log.e(TAG, "processFrame error: ${e.message}", e)
            runOnUiThread { eventSink?.error("INFER_ERROR", e.message, null) }
        } finally {
            imageProxy.close()
            frameReady = true
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Still-image inference
    // ─────────────────────────────────────────────────────────────────────────
    private fun runInferenceOnImagePath(path: String): List<Map<String, Any>> {
        val interp = interpreter ?: error("Model not loaded")

        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val raw  = BitmapFactory.decodeFile(path, opts)
            ?: error("Cannot decode image at $path")

        val lbBmp  = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(lbBmp)

        val scale   = INPUT_SIZE.toFloat() / max(raw.width, raw.height)
        val newW    = (raw.width  * scale).toInt()
        val newH    = (raw.height * scale).toInt()
        val padLeft = (INPUT_SIZE - newW) / 2
        val padTop  = (INPUT_SIZE - newH) / 2

        canvas.drawColor(padColor)
        canvas.drawBitmap(raw, null,
            android.graphics.RectF(padLeft.toFloat(), padTop.toFloat(),
                (padLeft + newW).toFloat(), (padTop + newH).toFloat()), lbPaint)
        raw.recycle()

        val bytesPerElem = if (isFloat32Model) 4 else 1
        val iBuf = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * bytesPerElem)
            .apply { order(ByteOrder.nativeOrder()) }

        val px = IntArray(INPUT_SIZE * INPUT_SIZE)
        lbBmp.getPixels(px, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        lbBmp.recycle()

        if (isFloat32Model) {
            val inv255 = 1f / 255f
            for (p in px) {
                iBuf.putFloat(((p shr 16) and 0xFF) * inv255)
                iBuf.putFloat(((p shr  8) and 0xFF) * inv255)
                iBuf.putFloat(( p         and 0xFF) * inv255)
            }
        } else {
            for (p in px) {
                iBuf.put(((p shr 16) and 0xFF).toByte())
                iBuf.put(((p shr  8) and 0xFF).toByte())
                iBuf.put(( p         and 0xFF).toByte())
            }
        }

        val oBuf = Array(1) { Array(4 + numClasses) { FloatArray(8400) } }
        for (row in oBuf[0]) row.fill(0f)
        interp.run(iBuf, oBuf)

        // origW/origH in terms of the raw image (before scale)
        val origW = raw.width   // raw was recycled but dimensions are captured by scale
        val origH = raw.height
        // Recalculate from scale: origW = (INPUT_SIZE - 2*padLeft) / scale
        val realOrigW = ((INPUT_SIZE - 2 * padLeft) / scale).toInt().coerceAtLeast(1)
        val realOrigH = ((INPUT_SIZE - 2 * padTop)  / scale).toInt().coerceAtLeast(1)

        return decodeOutput(oBuf[0], scale, padLeft, padTop, realOrigW, realOrigH)
            .map { b -> b.toStillImageMap() }
            .also { Log.d(TAG, "runOnImage → ${it.size} detections") }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // YUV_420_888 → RGB in-place (reuses rgbScratch Bitmap)
    // ─────────────────────────────────────────────────────────────────────────
    private fun yuvToRgbInPlace(imageProxy: ImageProxy) {
        val width  = imageProxy.width
        val height = imageProxy.height

        if (rgbScratch == null ||
            rgbScratch!!.width != width ||
            rgbScratch!!.height != height) {
            rgbScratch?.recycle()
            rgbScratch = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        }

        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride    = yPlane.rowStride
        val uvRowStride   = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val yData = ByteArray(yBuf.remaining()).also { yBuf.get(it) }
        val uData = ByteArray(uBuf.remaining()).also { uBuf.get(it) }
        val vData = ByteArray(vBuf.remaining()).also { vBuf.get(it) }

        // Reuse pixelScratch if big enough (allocated for INPUT_SIZE²)
        val pixels = if (width * height <= (pixelScratch?.size ?: 0))
            pixelScratch!! else IntArray(width * height)

        for (row in 0 until height) {
            val uvRow = row shr 1
            for (col in 0 until width) {
                val uvCol = col shr 1
                val yIdx  = row * yRowStride + col
                val uvIdx = uvRow * uvRowStride + uvCol * uvPixelStride

                val y = yData[yIdx].toInt() and 0xFF
                val u = (uData[uvIdx].toInt() and 0xFF) - 128
                val v = (vData[uvIdx].toInt() and 0xFF) - 128

                val r = (y + 1.402f * v).toInt().coerceIn(0, 255)
                val g = (y - 0.344f * u - 0.714f * v).toInt().coerceIn(0, 255)
                val b = (y + 1.772f * u).toInt().coerceIn(0, 255)
                pixels[row * width + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        rgbScratch!!.setPixels(pixels, 0, width, 0, 0, width, height)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Letterbox into the pre-allocated 640×640 bitmap.
    // Bakes rotation into the canvas draw — no intermediate copy needed.
    // ─────────────────────────────────────────────────────────────────────────
    private data class LetterboxMeta(val scale: Float, val padLeft: Int, val padTop: Int)

    private fun letterboxBitmapInPlace(src: Bitmap, rotationDegrees: Int): LetterboxMeta {
        val canvas = lbCanvas!!
        val displayW = if (rotationDegrees % 180 == 0) src.width  else src.height
        val displayH = if (rotationDegrees % 180 == 0) src.height else src.width

        val scale   = INPUT_SIZE.toFloat() / max(displayW, displayH)
        val newW    = (displayW * scale).toInt()
        val newH    = (displayH * scale).toInt()
        val padLeft = (INPUT_SIZE - newW) / 2
        val padTop  = (INPUT_SIZE - newH) / 2

        canvas.drawColor(padColor)

        if (rotationDegrees != 0) {
            canvas.save()
            canvas.translate(INPUT_SIZE / 2f, INPUT_SIZE / 2f)
            canvas.rotate(rotationDegrees.toFloat())
            val dst2 = android.graphics.RectF(-newW / 2f, -newH / 2f, newW / 2f, newH / 2f)
            canvas.drawBitmap(src, null, dst2, lbPaint)
            canvas.restore()
        } else {
            lbDst.set(padLeft.toFloat(), padTop.toFloat(),
                (padLeft + newW).toFloat(), (padTop + newH).toFloat())
            canvas.drawBitmap(src, null, lbDst, lbPaint)
        }

        return LetterboxMeta(scale, padLeft, padTop)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fill TFLite input buffer from the 640×640 letterboxed bitmap.
    // inv255 is hoisted outside the loop to avoid per-pixel division.
    // ─────────────────────────────────────────────────────────────────────────
    private fun fillInputBuffer(bitmap: Bitmap) {
        val buf     = inputBuffer!!
        val scratch = pixelScratch!!
        buf.rewind()

        bitmap.getPixels(scratch, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        if (isFloat32Model) {
            val inv255 = 1f / 255f
            for (px in scratch) {
                buf.putFloat(((px shr 16) and 0xFF) * inv255)
                buf.putFloat(((px shr  8) and 0xFF) * inv255)
                buf.putFloat(( px         and 0xFF) * inv255)
            }
        } else {
            for (px in scratch) {
                buf.put(((px shr 16) and 0xFF).toByte())
                buf.put(((px shr  8) and 0xFF).toByte())
                buf.put(( px         and 0xFF).toByte())
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Decode YOLO output + per-class NMS
    // ─────────────────────────────────────────────────────────────────────────
    private data class RawDet(
        val left: Float, val top: Float,
        val right: Float, val bottom: Float,
        val classIdx: Int, val score: Float
    )

    private fun decodeOutput(
        output: Array<FloatArray>,
        scale: Float,
        padLeft: Int, padTop: Int,
        origW: Int, origH: Int
    ): List<DetectionBox> {

        val numAnchors = output[0].size   // 8400 for 640-input YOLO
        val rawDets    = ArrayList<RawDet>(64)

        // Reciprocals computed once — avoids division inside the 8400-iteration loop
        val invScaleW = 1f / (scale * origW)
        val invScaleH = 1f / (scale * origH)

        for (a in 0 until numAnchors) {
            var bestScore = CONF_THRESH
            var bestClass = -1
            for (c in 0 until numClasses) {
                val s = output[4 + c][a]
                if (s > bestScore) { bestScore = s; bestClass = c }
            }
            if (bestClass == -1) continue

            val cx = output[0][a] * INPUT_SIZE
            val cy = output[1][a] * INPUT_SIZE
            val bw = output[2][a] * INPUT_SIZE
            val bh = output[3][a] * INPUT_SIZE

            val left   = (cx - bw / 2f - padLeft) * invScaleW
            val top    = (cy - bh / 2f - padTop)  * invScaleH
            val right  = (cx + bw / 2f - padLeft) * invScaleW
            val bottom = (cy + bh / 2f - padTop)  * invScaleH

            if (!isValidBox(left, top, right, bottom)) continue
            rawDets.add(RawDet(left, top, right, bottom, bestClass, bestScore))
        }

        return nms(rawDets).map { d ->
            DetectionBox(
                left   = d.left.coerceIn(0f, 1f),
                top    = d.top.coerceIn(0f, 1f),
                right  = d.right.coerceIn(0f, 1f),
                bottom = d.bottom.coerceIn(0f, 1f),
                classIdx = d.classIdx,
                score  = d.score,
                label  = labels.getOrElse(d.classIdx) { "class_${d.classIdx}" },
            )
        }
    }

    private fun TrackedDetection.toFlutterMap(): Map<String, Any> {
        val b = box
        return mapOf(
            "label"      to b.label,
            "confidence" to b.score.toDouble(),
            "left"       to b.left.toDouble(),
            "top"        to b.top.toDouble(),
            "right"      to b.right.toDouble(),
            "bottom"     to b.bottom.toDouble(),
            "trackId"    to trackId,
            "vx"         to vxNormPerSec,
            "vy"         to vyNormPerSec,
        )
    }

    /** Single-frame inference has no temporal context. */
    private fun DetectionBox.toStillImageMap(): Map<String, Any> = mapOf(
        "label"      to label,
        "confidence" to score.toDouble(),
        "left"       to left.toDouble(),
        "top"        to top.toDouble(),
        "right"      to right.toDouble(),
        "bottom"     to bottom.toDouble(),
        "trackId"    to -1,
        "vx"         to 0.0,
        "vy"         to 0.0,
    )

    private fun isValidBox(l: Float, t: Float, r: Float, b: Float): Boolean {
        if (!l.isFinite() || !t.isFinite() || !r.isFinite() || !b.isFinite()) return false
        val cl = l.coerceIn(0f, 1f); val ct = t.coerceIn(0f, 1f)
        val cr = r.coerceIn(0f, 1f); val cb = b.coerceIn(0f, 1f)
        val w = cr - cl; val h = cb - ct
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
        objectTracker.reset()
        stopCamera()
        interpreter?.close();       interpreter     = null
        gpuDelegate?.close();       gpuDelegate     = null
        nnApiDelegate?.close();     nnApiDelegate   = null
        letterboxBitmap?.recycle(); letterboxBitmap = null
        rgbScratch?.recycle();      rgbScratch      = null
        lbCanvas = null
        flutterSurfaceTextureEntry?.release()
        if (::cameraExecutor.isInitialized) cameraExecutor.shutdown()
        Log.i(TAG, "All native resources disposed")
    }

    override fun onDestroy() {
        disposeAll()
        super.onDestroy()
    }
}
