package com.example.adas_499

import kotlin.math.max
import kotlin.math.min

/** One YOLO detection in normalized [0,1] image coordinates. */
data class DetectionBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val classIdx: Int,
    val score: Float,
    val label: String,
)

/**
 * Per-frame association result: stable [trackId] after [nInit] consecutive matches
 * (same idea as DeepSORT confirmation in assets/code.ipynb).
 */
data class TrackedDetection(
    val box: DetectionBox,
    val trackId: Int,
    val vxNormPerSec: Double,
    val vyNormPerSec: Double,
)

/**
 * Lightweight multi-object tracker: greedy IoU matching within the same class,
 * max age for missed frames, and hit counting before a public ID is assigned.
 */
class ObjectTracker(
    private val maxAge: Int = 30,
    private val nInit: Int = 3,
    private val iouMatchThreshold: Float = 0.25f,
) {
    private var nextId = 1

    private class Track(
        val id: Int,
        var left: Float,
        var top: Float,
        var right: Float,
        var bottom: Float,
        val classIdx: Int,
        var score: Float,
        val label: String,
        var hits: Int,
        var timeLost: Int,
        var cxPrev: Float?,
        var cyPrev: Float?,
        var lastVx: Double = 0.0,
        var lastVy: Double = 0.0,
    )

    private val tracks = mutableListOf<Track>()

    fun reset() {
        tracks.clear()
        nextId = 1
    }

    fun update(dets: List<DetectionBox>, fps: Double): List<TrackedDetection> {
        val f = fps.coerceIn(1.0, 120.0)

        if (dets.isEmpty()) {
            val it = tracks.iterator()
            while (it.hasNext()) {
                val t = it.next()
                t.timeLost++
                if (t.timeLost > maxAge) it.remove()
            }
            return emptyList()
        }

        val unmatchedTracks = tracks.toMutableList()
        val unmatchedDets = dets.toMutableList()
        val matches = ArrayList<Pair<Track, DetectionBox>>(min(dets.size, tracks.size))

        while (true) {
            var bestIou = iouMatchThreshold
            var bestT: Track? = null
            var bestD: DetectionBox? = null
            for (t in unmatchedTracks) {
                for (d in unmatchedDets) {
                    if (d.classIdx != t.classIdx) continue
                    val iou = iou(
                        t.left, t.top, t.right, t.bottom,
                        d.left, d.top, d.right, d.bottom,
                    )
                    if (iou > bestIou) {
                        bestIou = iou
                        bestT = t
                        bestD = d
                    }
                }
            }
            if (bestT == null) break
            matches.add(bestT!! to bestD!!)
            unmatchedTracks.remove(bestT)
            unmatchedDets.remove(bestD)
        }

        for (t in unmatchedTracks) {
            t.timeLost++
        }
        tracks.removeAll { it.timeLost > maxAge }

        for ((t, d) in matches) {
            val cx = (d.left + d.right) / 2f
            val cy = (d.top + d.bottom) / 2f
            if (t.cxPrev != null && t.cyPrev != null) {
                t.lastVx = ((cx - t.cxPrev!!) * f).toDouble()
                t.lastVy = ((cy - t.cyPrev!!) * f).toDouble()
            } else {
                t.lastVx = 0.0
                t.lastVy = 0.0
            }
            t.left = d.left
            t.top = d.top
            t.right = d.right
            t.bottom = d.bottom
            t.score = d.score
            t.hits++
            t.timeLost = 0
            t.cxPrev = cx
            t.cyPrev = cy
        }

        val brandNew = ArrayList<Pair<DetectionBox, Track>>(unmatchedDets.size)
        for (d in unmatchedDets) {
            val cx = (d.left + d.right) / 2f
            val cy = (d.top + d.bottom) / 2f
            val t = Track(
                id = nextId++,
                left = d.left,
                top = d.top,
                right = d.right,
                bottom = d.bottom,
                classIdx = d.classIdx,
                score = d.score,
                label = d.label,
                hits = 1,
                timeLost = 0,
                cxPrev = cx,
                cyPrev = cy,
            )
            tracks.add(t)
            brandNew.add(d to t)
        }

        return dets.map { d ->
            matches.find { it.second === d }?.let { (t, _) ->
                val tid = if (t.hits >= nInit) t.id else -1
                TrackedDetection(d, tid, t.lastVx, t.lastVy)
            } ?: brandNew.find { it.first === d }?.let {
                TrackedDetection(d, -1, 0.0, 0.0)
            } ?: TrackedDetection(d, -1, 0.0, 0.0)
        }
    }

    private fun iou(
        al: Float, at: Float, ar: Float, ab: Float,
        bl: Float, bt: Float, br: Float, bb: Float,
    ): Float {
        val il = max(al, bl)
        val it = max(at, bt)
        val ir = min(ar, br)
        val ib = min(ab, bb)
        if (ir <= il || ib <= it) return 0f
        val inter = (ir - il) * (ib - it)
        val aArea = (ar - al) * (ab - at)
        val bArea = (br - bl) * (bb - bt)
        val union = aArea + bArea - inter
        return if (union <= 0f) 0f else inter / union
    }
}
