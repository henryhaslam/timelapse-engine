"""Additional effects: the entropy family, temporal colour maps, time-slice and
clean-up/trail effects. All follow the Processor contract in ``processors.py``
and run on the GPU (CuPy) or CPU (NumPy) from the same code.

Temporal colour (one value per pixel per colour channel -> false colour)
    temporal_entropy   Shannon / Renyi entropy of each pixel's values over time
    change_entropy     entropy of frame-to-frame *changes* (turbulence vs drift)
    colour_entropy     entropy of brightness and colour separately (Y, Cb, Cr)
    temporal_shape     skewness or kurtosis of each pixel's values

Motion and time
    time_of_peak       hue = when each pixel was brightest / darkest / changed most
    rhythm_map         hue = dominant rhythm (period) of each pixel's flicker
    eigen_colour       the 3 strongest patterns of change (PCA) mapped to RGB
    memory_map         how long pixels stay similar at 3 time scales -> RGB

Time slice
    time_slice         each part of the image from a different moment

Clean-up and trails
    temporal_denoise   motion-aware median (for video / image sequences)
    comet_trails       brightest values that fade out like comet tails
    clipped_average    average that rejects outliers (planes, people, birds)
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .backend import HAS_CUPY, asnumpy, cp, xp_of
from .processors import (Processor, _batch_sum, _channel_max, _dilate_cross, normalise_threshold, register)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ndimage(xp):
    if xp is np:
        import scipy.ndimage as nd
    else:
        import cupyx.scipy.ndimage as nd
    return nd


def _luma(frames):
    """(..., C) -> (...) BT.709 luma (or the single channel)."""
    if frames.shape[-1] == 1:
        return frames[..., 0]
    return frames[..., 0] * 0.2126 + frames[..., 1] * 0.7152 + frames[..., 2] * 0.0722


def _robust_max(x, q: float = 99.5) -> float:
    """High percentile of an array, from a subsample (fast on 20 MP maps)."""
    xp = xp_of(x)
    flat = x.ravel()
    sample = flat[:: max(1, flat.size // 2_000_000)]
    return float(xp.percentile(sample.astype(xp.float32), q))


def _activity(std, q: float = 99.0, power: float = 0.5):
    """0-1 map of how much a pixel varied, so static areas render black (like entropy)."""
    xp = xp_of(std)
    ref = _robust_max(std, q)
    if ref <= 0:
        return xp.zeros(std.shape, xp.float32)
    return (xp.clip(std / ref, 0.0, 1.0) ** power).astype(xp.float32)


def hsv_to_rgb(h, s, v):
    """Vectorised HSV -> RGB on NumPy or CuPy. h, s, v in 0-1, same shape. Returns (..., 3) float32."""
    xp = xp_of(h)
    h6 = (h % 1.0) * 6.0
    i = xp.floor(h6)
    f = h6 - i
    i = i.astype(xp.int32) % 6
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    def pick(c0, c1, c2, c3, c4, c5):
        return xp.where(i == 0, c0, xp.where(i == 1, c1, xp.where(i == 2, c2,
                        xp.where(i == 3, c3, xp.where(i == 4, c4, c5)))))

    rgb = xp.stack([pick(v, q, p, p, t, v), pick(t, v, v, q, p, p), pick(p, p, t, v, v, q)], axis=-1)
    return xp.clip(rgb, 0.0, 1.0).astype(xp.float32)


class _Indexed(Processor):
    """Base for effects that need to know each frame's position in the sequence."""

    def begin(self, n_frames, frame_shape, indices):
        self.n_frames = int(n_frames)
        self.indices = list(indices)
        self._k = 0

    def _next_index(self) -> int:
        k = self._k
        self._k += 1
        return self.indices[k] if k < len(self.indices) else k


# ---------------------------------------------------------------------------
# Entropy family
# ---------------------------------------------------------------------------

_ENTROPY_KERNEL = None


def _entropy_kernel():
    """CUDA kernel: one histogram increment per pixel/channel. Each element owns its own
    slice of the histogram, so no atomics (and no index arrays) are needed."""
    global _ENTROPY_KERNEL
    if _ENTROPY_KERNEL is None:
        _ENTROPY_KERNEL = cp.ElementwiseKernel(
            "float32 v, int32 bins", "raw T hist",
            "int b = (int)floorf(v * bins); b = min(max(b, 0), bins - 1);"
            "hist[(long long)i * bins + b] += 1;",
            "timelapse_entropy_hist",
        )
    return _ENTROPY_KERNEL


class _EntropyBase(Processor):
    """Histogram of a per-pixel value (0-1) over time, turned into entropy.

    renyi_order (alpha): 1 = Shannon entropy. Below 1 lifts rare events (a bird, a
    passing light, a single bright cloud edge); above 1 keeps only dominant changes.
    neighbourhood: also count the values of pixels this many px around each pixel,
    for a smoother, more painterly result.
    """

    category = "Temporal colour"

    def __init__(self, hist_bins: int = 32, noise_threshold: float = 3.0, renyi_order: float = 1.0,
                 neighbourhood: int = 0, use_gpu: Optional[bool] = None):
        self.hist_bins = int(hist_bins)
        self.noise_threshold = normalise_threshold(noise_threshold)
        self.renyi_order = float(renyi_order)
        self.neighbourhood = int(neighbourhood)
        if self.renyi_order <= 0:
            raise ValueError("renyi_order must be greater than 0")
        self.gpu = HAS_CUPY if use_gpu is None else bool(use_gpu and HAS_CUPY)
        self.xp = cp if self.gpu else np
        self.reset()

    def state_bytes(self, frame_shape):
        # uint16 histogram (uint32 past 65535 frames) + previous frame + temporaries
        return int(np.prod(frame_shape) * (self.hist_bins * 2 + 4 * 4))

    def reset(self):
        self.hist = None  # flat (pixels * channels * bins)
        self.prev = None
        self.shape = None
        self.count = 0

    # -- per-effect hooks ---------------------------------------------------
    def _values(self, frame):
        """frame (H, W, C) float32 -> values (H, W, C') in 0-1 to histogram, or None to skip."""
        xp = self.xp
        if self.prev is not None and self.noise_threshold > 0:
            # Ignore tiny sensor-noise changes: keep the previous value instead.
            frame = xp.where(xp.abs(frame - self.prev) < self.noise_threshold, self.prev, frame)
        self.prev = frame.copy()
        return frame

    # -- accumulation ---------------------------------------------------------
    def _add(self, values):
        if self.gpu:
            _entropy_kernel()(values.ravel(), np.int32(self.hist_bins), self.hist)
        else:
            bins = np.clip(np.floor(values.ravel() * self.hist_bins), 0, self.hist_bins - 1).astype(np.int64)
            idx = np.arange(bins.size, dtype=np.int64) * self.hist_bins + bins
            self.hist[idx] += 1  # indices are unique within a frame, so plain += is safe

    def update(self, batch):
        xp = self.xp
        for frame in xp.asarray(batch, dtype=xp.float32):
            values = self._values(frame)
            if values is None:
                continue
            if self.hist is None:
                self.shape = values.shape
                self.hist = xp.zeros(int(np.prod(self.shape)) * self.hist_bins, dtype=xp.uint16)
            if self.hist.dtype == xp.uint16 and self.count + 1 > np.iinfo(np.uint16).max:
                self.hist = self.hist.astype(xp.uint32)
            self._add(xp.ascontiguousarray(values, dtype=xp.float32))
            self.count += 1

    def entropy(self):
        """(H, W, C') entropy normalised to 0-1 (1 = values spread evenly over all bins)."""
        xp = self.xp
        if self.hist is None:
            raise ValueError("No frames were processed.")
        h, w, c = self.shape
        bins = self.hist_bins
        hist = self.hist.reshape(h, w, c, bins)
        alpha = self.renyi_order
        shannon = abs(alpha - 1.0) < 1e-6
        k = 2 * self.neighbourhood + 1
        nd = _ndimage(xp) if k > 1 else None
        tiny = xp.float32(1e-12)
        count = xp.float32(max(self.count, 1))
        out = xp.empty((h, w, c), xp.float32)
        for ch in range(c):
            acc = xp.zeros((h, w), xp.float32)
            for b0 in range(0, bins, 4):  # a few bins at a time keeps memory bounded
                p = hist[:, :, ch, b0:b0 + 4].astype(xp.float32)
                if nd is not None:
                    p = nd.uniform_filter(p, size=(k, k, 1), mode="nearest")
                p /= count
                if shannon:
                    acc -= (p * xp.log2(xp.maximum(p, tiny))).sum(axis=2)
                else:
                    acc += xp.where(p > 0, p ** xp.float32(alpha), 0).sum(axis=2)
            if not shannon:
                acc = xp.log2(xp.maximum(acc, tiny)) / xp.float32(1.0 - alpha)
            out[..., ch] = acc
        out /= xp.float32(np.log2(bins))
        return xp.clip(out, 0.0, 1.0)

    def result(self):
        return self.entropy()


@register
class TemporalEntropyProcessor(_EntropyBase):
    name = "temporal_entropy"
    description = ("Shannon entropy of each pixel's values over time - how varied each colour channel was. "
                   "Moving clouds and water turn vivid; static scenery goes black.")


@register
class ChangeEntropyProcessor(_EntropyBase):
    name = "change_entropy"
    description = ("Entropy of frame-to-frame changes instead of values: turbulent motion (boiling clouds, "
                   "choppy water, flicker) separates from smooth drift.")

    def __init__(self, hist_bins: int = 32, noise_threshold: float = 3.0, renyi_order: float = 1.0,
                 neighbourhood: int = 0, change_range: float = 64.0, use_gpu: Optional[bool] = None):
        self.change_range = max(normalise_threshold(change_range), 1e-3)
        super().__init__(hist_bins, noise_threshold, renyi_order, neighbourhood, use_gpu)

    def _values(self, frame):
        xp = self.xp
        if self.prev is None:
            self.prev = frame.copy()
            return None
        diff = frame - self.prev
        self.prev = frame.copy()
        if self.noise_threshold > 0:
            diff = xp.where(xp.abs(diff) < self.noise_threshold, xp.float32(0), diff)
        centre = (self.hist_bins // 2 + 0.5) / self.hist_bins  # "no change" sits mid-bin, not on an edge
        return xp.clip(centre + diff / (2 * self.change_range), 0.0, 1.0)


@register
class ColourEntropyProcessor(_EntropyBase):
    name = "colour_entropy"
    description = ("Entropy of brightness and colour separately: red = colour shifts towards red/cyan, "
                   "green = brightness changes, blue = shifts towards blue/yellow. Sunsets pop.")

    def __init__(self, hist_bins: int = 32, noise_threshold: float = 3.0, renyi_order: float = 1.0,
                 neighbourhood: int = 0, chroma_gain: float = 4.0, use_gpu: Optional[bool] = None):
        self.chroma_gain = float(chroma_gain)
        super().__init__(hist_bins, noise_threshold, renyi_order, neighbourhood, use_gpu)

    def _values(self, frame):
        frame = super()._values(frame)
        xp = self.xp
        if frame.shape[-1] == 1:
            return frame
        y = _luma(frame)
        cb = (frame[..., 2] - y) / 1.8556
        cr = (frame[..., 0] - y) / 1.5748
        g = xp.float32(self.chroma_gain)
        centre = xp.float32((self.hist_bins // 2 + 0.5) / self.hist_bins)  # neutral grey sits mid-bin
        return xp.clip(xp.stack([centre + cr * g, y, centre + cb * g], axis=-1), 0.0, 1.0)


@register
class TemporalShapeProcessor(Processor):
    name = "temporal_shape"
    category = "Temporal colour"
    description = ("Shape of each pixel's brightness distribution over time, per colour channel. "
                   "Skewness: mostly-dark-with-bright-moments vs the opposite. Kurtosis: rare spikes vs steady change.")
    state_frames = 12.0

    def __init__(self, statistic: str = "skewness", gain: float = 1.5):
        if statistic not in ("skewness", "kurtosis"):
            raise ValueError("statistic must be 'skewness' or 'kurtosis'")
        self.statistic = statistic
        self.gain = float(gain)
        self.reset()

    def reset(self):
        self.x0 = None
        self.n = 0
        self.sums = None

    def update(self, batch):
        xp = xp_of(batch)
        if self.x0 is None:
            self.x0 = batch[0].copy()  # moments of (x - first frame): far better conditioned
            self.sums = [xp.zeros(batch.shape[1:], xp.float64) for _ in range(4)]
        d = batch - self.x0
        p = d.copy()
        for k in range(4):
            if k:
                p *= d
            self.sums[k] += p.sum(axis=0, dtype=xp.float32).astype(xp.float64)
        self.n += batch.shape[0]

    def result(self):
        xp = xp_of(self.sums[0])
        n = max(self.n, 1)
        e1, e2, e3, e4 = (s / n for s in self.sums)
        var = e2 - e1 * e1
        valid = var > 1e-9
        safe = xp.where(valid, var, 1.0)
        if self.statistic == "skewness":
            m3 = e3 - 3 * e1 * e2 + 2 * e1 ** 3
            v = m3 / safe ** 1.5
        else:
            m4 = e4 - 4 * e1 * e3 + 6 * e1 * e1 * e2 - 3 * e1 ** 4
            v = m4 / (safe * safe) - 3.0
        mapped = 0.5 + 0.5 * xp.tanh(v / self.gain)
        std = xp.sqrt(xp.maximum(var, 0)).astype(xp.float32)
        out = xp.where(valid, mapped, 0).astype(xp.float32) * _activity(std)
        return xp.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Motion and time
# ---------------------------------------------------------------------------

@register
class TimeOfPeakProcessor(_Indexed):
    name = "time_of_peak"
    category = "Motion and time"
    description = ("Colour = when each pixel was brightest, darkest or changed most; brightness = how much. "
                   "Shadows sweeping across buildings and sunset light become rainbow gradients.")
    state_frames = 4.0

    def __init__(self, event: str = "biggest_change", hue_span: float = 300.0):
        if event not in ("brightest", "darkest", "biggest_change"):
            raise ValueError("event must be 'brightest', 'darkest' or 'biggest_change'")
        self.event = event
        self.hue_span = float(hue_span)
        self.reset()

    def reset(self):
        self.best = None
        self.when = None
        self.prev = None
        self.n_frames, self.indices, self._k = 0, [], 0

    def update(self, batch):
        xp = xp_of(batch)
        for y in _luma(batch):
            idx = self._next_index()
            if self.event == "biggest_change":
                if self.prev is None:
                    self.prev = y.copy()
                    continue
                score = xp.abs(y - self.prev)
                self.prev = y.copy()
            else:
                score = y if self.event == "brightest" else 1.0 - y
            if self.best is None:
                self.best = score.copy()
                self.when = xp.full(score.shape, idx, xp.float32)
                continue
            better = score > self.best
            self.best = xp.where(better, score, self.best)
            self.when = xp.where(better, xp.float32(idx), self.when)

    def result(self):
        xp = xp_of(self.best)
        span = max(self.n_frames - 1, 1) if self.n_frames else max(float(self.when.max()), 1.0)
        hue = (self.when / span) * (self.hue_span / 360.0)
        if self.event == "biggest_change":
            ref = _robust_max(self.best, 99.0)
            value = xp.clip(self.best / ref, 0, 1) ** 0.5 if ref > 0 else self.best  # lift gentler changes
        else:
            value = self.best
        return hsv_to_rgb(hue, xp.ones_like(value), value.astype(xp.float32))


@register
class RhythmMapProcessor(_Indexed):
    name = "rhythm_map"
    category = "Motion and time"
    description = ("Colour = the dominant rhythm of each pixel's flicker (waves, traffic, blinking lights, drifting "
                   "clouds each get their own hue); saturation = how regular it is; brightness = how strong.")

    BANDS = 16

    def __init__(self, min_period: float = 2.0, max_period: float = 60.0):
        if not 2.0 <= min_period < max_period:
            raise ValueError("Periods must satisfy 2 <= min_period < max_period (in frames)")
        self.periods = np.geomspace(min_period, max_period, self.BANDS)
        self.reset()

    def state_bytes(self, frame_shape):
        return int(np.prod(frame_shape[:2]) * 4 * (2 * self.BANDS + 6))

    def reset(self):
        self.re = self.im = None
        self.prev = None
        self.count = 0
        self.n_frames, self.indices, self._k = 0, [], 0

    def update(self, batch):
        xp = xp_of(batch)
        omega = 2 * np.pi / self.periods
        for y in _luma(batch):
            t = self._next_index()
            if self.prev is None:
                self.prev = y.copy()
                continue
            d = y - self.prev  # first difference removes the average level and slow trends
            self.prev = y.copy()
            if self.re is None:
                self.re = xp.zeros((self.BANDS,) + d.shape, xp.float32)
                self.im = xp.zeros_like(self.re)
            for k, (c, s_) in enumerate(zip(np.cos(omega * t), np.sin(omega * t))):  # per band: small temporaries
                self.re[k] += xp.float32(c) * d
                self.im[k] += xp.float32(s_) * d
            self.count += 1

    def result(self):
        xp = xp_of(self.re)
        # undo the first-difference's frequency response: |2 sin(pi f)|
        response = xp.asarray((2 * np.sin(np.pi / self.periods)) ** 2, dtype=xp.float32)[:, None, None]
        power, im = self.re, self.im          # in place: no full-size temporaries (the state isn't reused)
        power *= power
        im *= im
        power += im
        power /= response
        self.re = self.im = None
        k = power.argmax(axis=0)
        peak = xp.take_along_axis(power, k[None], axis=0)[0]
        total = power.sum(axis=0) + xp.float32(1e-12)
        purity = (peak / total - 1.0 / self.BANDS) / (1.0 - 1.0 / self.BANDS)
        amp = xp.sqrt(peak) / max(self.count, 1)
        ref = _robust_max(amp)
        value = xp.clip(amp / ref, 0, 1) ** 0.7 if ref > 0 else amp
        hue = k.astype(xp.float32) / (self.BANDS - 1) * 0.8   # short periods red -> long periods violet
        return hsv_to_rgb(hue, xp.clip(purity * 2.5, 0, 1).astype(xp.float32), value.astype(xp.float32))


@register
class EigenColourProcessor(Processor):
    name = "eigen_colour"
    category = "Motion and time"
    two_pass = True
    description = ("Finds the three strongest patterns of change in the sequence (PCA) and maps them to red, "
                   "green and blue, so sky, water, shadows and lights separate by how they change.")
    state_frames = 6.0

    def __init__(self, samples: int = 65536, mode: str = "magnitude"):
        if mode not in ("magnitude", "signed"):
            raise ValueError("mode must be 'magnitude' or 'signed'")
        self.samples = int(samples)
        self.mode = mode
        self.reset()

    def reset(self):
        self.total = None
        self.count = 0
        self.sample_idx = None
        self.series = []
        self.basis = None
        self.mean = None
        self.comps = None
        self.t2 = 0

    def update(self, batch):
        xp = xp_of(batch)
        y = _luma(batch)                                   # (N, H, W)
        if self.sample_idx is None:
            n_pix = y.shape[1] * y.shape[2]
            rng = np.random.default_rng(0)
            self.sample_idx = xp.asarray(rng.choice(n_pix, size=min(self.samples, n_pix), replace=False))
            self.total = xp.zeros(y.shape[1:], xp.float64)
        self.total += y.sum(axis=0, dtype=xp.float32).astype(xp.float64)
        self.count += y.shape[0]
        self.series.append(y.reshape(y.shape[0], -1)[:, self.sample_idx].copy())

    def begin_second_pass(self):
        xp = xp_of(self.total)
        x = xp.concatenate(self.series, axis=0)            # (T, S) time series of sampled pixels
        self.series = []
        x -= x.mean(axis=0, keepdims=True)
        t = x.shape[0]
        # The big products run on the GPU; the small (T x T or T x 8) decompositions run in
        # NumPy, which also avoids needing CUDA's cuSOLVER library.
        if t < 2:
            basis = np.zeros((t, 3), np.float32)
        elif t <= 4096:
            gram = asnumpy(x @ x.T).astype(np.float64)    # temporal covariance (T x T)
            w, v = np.linalg.eigh(gram)
            basis = v[:, ::-1][:, :3]
        else:                                              # randomised SVD for very long sequences
            rng = np.random.default_rng(1)
            q = asnumpy(x @ xp.asarray(rng.standard_normal((x.shape[1], 8)).astype(np.float32)))
            for _ in range(2):
                q, _r = np.linalg.qr(q)
                qd = xp.asarray(q.astype(np.float32))
                q = asnumpy(x @ (x.T @ qd))
            q, _r = np.linalg.qr(q)
            b = asnumpy(xp.asarray(q.T.astype(np.float32)) @ x)
            ub, _s, _vt = np.linalg.svd(b, full_matrices=False)
            basis = q @ ub[:, :3]
        if basis.shape[1] < 3:
            basis = np.concatenate([basis, np.zeros((t, 3 - basis.shape[1]))], axis=1)
        signs = np.sign(basis.sum(axis=0))
        self.basis = xp.asarray((basis * np.where(signs == 0, 1, signs)).astype(np.float32))
        self.mean = (self.total / max(self.count, 1)).astype(xp.float32)
        self.comps = xp.zeros((3,) + self.mean.shape, xp.float32)
        self.t2 = 0

    def update_second_pass(self, batch):
        xp = xp_of(batch)
        yc = _luma(batch) - self.mean
        n = yc.shape[0]
        self.comps += xp.tensordot(self.basis[self.t2:self.t2 + n].T, yc, axes=1)
        self.t2 += n

    def result(self):
        xp = xp_of(self.comps)
        out = []
        for c in self.comps:
            ref = _robust_max(xp.abs(c))
            c = c / ref if ref > 0 else c
            out.append(xp.clip(xp.abs(c), 0, 1) if self.mode == "magnitude" else xp.clip(0.5 + 0.5 * c, 0, 1))
        return xp.stack(out, axis=-1).astype(xp.float32)


@register
class MemoryMapProcessor(Processor):
    name = "memory_map"
    category = "Motion and time"
    description = ("How long each pixel stays similar to itself, at three time scales mapped to red, green and "
                   "blue: water forgets instantly, clouds quickly, slow shadows and light remember.")
    state_frames = 14.0

    def __init__(self, base_scale: int = 1, ratio: int = 4):
        if base_scale < 1 or ratio < 2:
            raise ValueError("base_scale must be >= 1 and ratio >= 2")
        self.scales = [base_scale, base_scale * ratio, base_scale * ratio * ratio]
        self.reset()

    def reset(self):
        self.y0 = None
        self.state = None

    def update(self, batch):
        xp = xp_of(batch)
        for y in _luma(batch):
            if self.y0 is None:
                self.y0 = y.copy()
                z = lambda: xp.zeros(y.shape, xp.float64)  # noqa: E731
                self.state = [{"block": xp.zeros(y.shape, xp.float32), "n": 0, "prev": None, "count": 0,
                               "pairs": 0, "sv": z(), "svv": z(), "sprod": z()} for _ in self.scales]
            y = y - self.y0
            for s, st in zip(self.scales, self.state):
                st["block"] += y
                st["n"] += 1
                if st["n"] < s:
                    continue
                v = (st["block"] / s).astype(xp.float64)
                st["block"][...] = 0
                st["n"] = 0
                if st["prev"] is not None:
                    st["sprod"] += v * st["prev"]
                    st["pairs"] += 1
                st["sv"] += v
                st["svv"] += v * v
                st["count"] += 1
                st["prev"] = v

    def result(self):
        xp = xp_of(self.y0)
        channels, std1 = [], None
        for st in self.state:
            if st["count"] < 3 or st["pairs"] < 2:
                channels.append(xp.zeros(self.y0.shape, xp.float32))
                continue
            m = st["sv"] / st["count"]
            var = st["svv"] / st["count"] - m * m
            cov = st["sprod"] / st["pairs"] - m * m
            r = xp.where(var > 1e-10, cov / xp.where(var > 1e-10, var, 1.0), 0)
            channels.append(xp.clip(r, 0, 1).astype(xp.float32))
            if std1 is None:
                std1 = xp.sqrt(xp.maximum(var, 0)).astype(xp.float32)
        act = _activity(std1) if std1 is not None else xp.zeros(self.y0.shape, xp.float32)
        return xp.stack(channels, axis=-1) * act[..., None]


# ---------------------------------------------------------------------------
# Time slice
# ---------------------------------------------------------------------------

@register
class TimeSliceProcessor(_Indexed):
    """Each part of the image comes from a different moment of the sequence.

    pattern: left_to_right, top_to_bottom, diagonal (at ``angle``), radial (centre
    outwards), circular (a clock sweep starting at ``angle``) or brightness (bright
    parts of the first frame from late in the sequence).
    slices: 0 = smooth (every column/ring from its own moment, blended between
    neighbouring frames); N = N bands, each from one frame.
    feather: 0-1 fraction of a band over which neighbouring bands blend (through
    the frames in between).
    reverse: run time the other way. mirror: there and back (e.g. night in the middle).
    curve: <1 gives more of the image to the end of the sequence, >1 to the start.
    """

    name = "time_slice"
    category = "Time slice"
    description = ("Time-slice: each part of the image from a different moment - day to night across the frame. "
                   "Smooth or banded, along lines, rings, a clock sweep or by brightness.")
    state_frames = 4.0
    PATTERNS = ("left_to_right", "top_to_bottom", "diagonal", "radial", "circular", "brightness")

    def __init__(self, pattern: str = "left_to_right", slices: int = 0, feather: float = 0.0,
                 angle: float = 45.0, reverse: bool = False, mirror: bool = False, curve: float = 1.0):
        if pattern not in self.PATTERNS:
            raise ValueError(f"pattern must be one of {', '.join(self.PATTERNS)}")
        if slices < 0 or not 0 <= feather <= 1 or curve <= 0:
            raise ValueError("slices >= 0, 0 <= feather <= 1 and curve > 0 are required")
        self.pattern, self.slices, self.feather = pattern, int(slices), float(feather)
        self.angle, self.reverse, self.mirror, self.curve = float(angle), bool(reverse), bool(mirror), float(curve)
        self.reset()

    def reset(self):
        self.ft = None        # per-pixel time, in frame units
        self.canvas = None
        self.weight = None
        self.n_frames, self.indices, self._k = 0, [], 0

    # -- the time map ----------------------------------------------------------
    def _shape_map(self, xp, h, w, first_frame):
        if self.pattern == "brightness":
            y = _luma(first_frame)
            y = _ndimage(xp).uniform_filter(y, size=max(3, max(h, w) // 150), mode="nearest")
            lo, hi = float(xp.percentile(y[::4, ::4], 1)), float(xp.percentile(y[::4, ::4], 99))
            return xp.clip((y - lo) / max(hi - lo, 1e-6), 0, 1).astype(xp.float32)
        yy, xx = xp.meshgrid(xp.arange(h, dtype=xp.float32), xp.arange(w, dtype=xp.float32), indexing="ij")
        cx, cy = (w - 1) / 2, (h - 1) / 2
        if self.pattern == "left_to_right":
            return xx / max(w - 1, 1)
        if self.pattern == "top_to_bottom":
            return yy / max(h - 1, 1)
        if self.pattern == "diagonal":
            a = math.radians(self.angle)
            proj = xx * math.cos(a) + yy * math.sin(a)
            corners = [0, (w - 1) * math.cos(a), (h - 1) * math.sin(a), (w - 1) * math.cos(a) + (h - 1) * math.sin(a)]
            lo, hi = min(corners), max(corners)
            return (proj - lo) / max(hi - lo, 1e-6)
        if self.pattern == "radial":
            d = xp.hypot(xx - cx, yy - cy)
            return d / max(math.hypot(cx, cy), 1e-6)
        ang = (xp.arctan2(yy - cy, xx - cx) - math.radians(self.angle)) % (2 * math.pi)  # circular
        return ang / (2 * math.pi)

    def _shape_to_time(self, xp, tau, n):
        tau = xp.clip(tau, 0, 1) ** self.curve
        if self.reverse:
            tau = 1 - tau
        if self.mirror:
            tau = 1 - xp.abs(2 * tau - 1)
        if self.slices <= 0:
            return (tau * (n - 1)).astype(xp.float32)
        if self.slices == 1:
            return xp.zeros(tau.shape, xp.float32)
        g = tau * self.slices - 0.5                    # band coordinate; integers = band centres
        base = xp.floor(g)
        frac = g - base
        if self.feather > 0:
            ramp = xp.clip((frac - 0.5) / self.feather + 0.5, 0, 1)
        else:
            ramp = (frac >= 0.5).astype(xp.float32)
        band = xp.clip(base + ramp, 0, self.slices - 1)
        ft = band / (self.slices - 1) * (n - 1)
        return (ft if self.feather > 0 else xp.rint(ft)).astype(xp.float32)

    def frames_needed(self, n_frames, frame_shape):
        if self.slices <= 0 or self.feather > 0 or self.pattern == "brightness" or n_frames < 2:
            return None
        if self.slices == 1:
            return [0]
        return sorted({int(round(k / (self.slices - 1) * (n_frames - 1))) for k in range(self.slices)})

    # -- accumulation ------------------------------------------------------------
    def update(self, batch):
        xp = xp_of(batch)
        for frame in batch:
            idx = self._next_index()
            if self.ft is None:
                h, w = frame.shape[:2]
                n = self.n_frames or len(self.indices) or 1
                self.ft = self._shape_to_time(xp, self._shape_map(xp, h, w, frame), n)
                self.canvas = xp.zeros(frame.shape, xp.float32)
                self.weight = xp.zeros((h, w), xp.float32)
            wgt = xp.clip(1.0 - xp.abs(self.ft - xp.float32(idx)), 0, 1)
            self.canvas += wgt[..., None] * frame
            self.weight += wgt

    def result(self):
        xp = xp_of(self.canvas)
        return (self.canvas / xp.maximum(self.weight, 1e-6)[..., None]).astype(xp.float32)


# ---------------------------------------------------------------------------
# Clean-up and trails
# ---------------------------------------------------------------------------

@register
class TemporalDenoiseProcessor(Processor):
    name = "temporal_denoise"
    category = "Clean-up and trails"
    description = ("Motion-aware temporal median for videos and image sequences (use a blend of 3-9): noise "
                   "averages away, while anything that moved keeps the centre frame so it doesn't smear.")
    state_frames = 12.0

    def __init__(self, motion_threshold: float = 12.0):
        self.motion_threshold = normalise_threshold(motion_threshold)
        self.reset()

    def begin(self, n_frames, frame_shape, indices):
        from .backend import gpu_free_bytes

        need = 3 * n_frames * int(np.prod(frame_shape)) * 4
        if HAS_CUPY and need > 0.7 * gpu_free_bytes() and n_frames > 9:
            raise MemoryError(f"temporal_denoise holds every frame of a window in memory ({n_frames} here). "
                              f"Use it for videos/image sequences with a blend of 3-9.")

    def reset(self):
        self.frames = []

    def update(self, batch):
        self.frames.append(batch.copy())

    def result(self):
        xp = xp_of(self.frames[0])
        stack = xp.concatenate(self.frames, axis=0)
        centre = stack[stack.shape[0] // 2]
        med = xp.median(stack, axis=0)
        # Average the difference over 5x5 px: noise averages to its (small) mean, real motion stays large.
        diff = _ndimage(xp).uniform_filter(xp.abs(centre - med).mean(axis=2), size=5, mode="nearest")
        moving = _dilate_cross(diff > self.motion_threshold)
        return xp.where(moving[..., None], centre, med).astype(xp.float32)


@register
class CometTrailsProcessor(Processor):
    name = "comet_trails"
    category = "Clean-up and trails"
    description = ("Like Maximum, but older light fades out, so star and light trails taper like comets. "
                   "In video, moving lights leave flowing tails.")

    def __init__(self, trail_length: float = 20.0):
        if trail_length <= 0:
            raise ValueError("trail_length must be > 0")
        self.decay = math.exp(-1.0 / float(trail_length))
        self.reset()

    def reset(self):
        self.canvas = None

    def update(self, batch):
        xp = xp_of(batch)
        d = xp.float32(self.decay)
        for frame in batch:
            if self.canvas is None:
                self.canvas = frame.copy()
                continue
            faded = frame + (self.canvas - frame) * d  # old light drifts back to the current frame
            self.canvas = xp.maximum(frame, faded)

    def result(self):
        return self.canvas


@register
class ClippedAverageProcessor(Processor):
    name = "clipped_average"
    category = "Clean-up and trails"
    two_pass = True
    description = ("Long-exposure average that ignores outliers per pixel (planes, birds, people walking through): "
                   "cleaner than Average, far lighter on memory than Median.")
    state_frames = 10.0

    def __init__(self, sigma: float = 2.0):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = float(sigma)
        self.reset()

    def reset(self):
        self.s = self.ss = None
        self.n = 0
        self.lo = self.hi = self.mean = None
        self.inlier_sum = self.inlier_n = None

    def update(self, batch):
        xp = xp_of(batch)
        if self.s is None:
            self.s = xp.zeros(batch.shape[1:], xp.float64)
            self.ss = xp.zeros(batch.shape[1:], xp.float64)
        for frame in batch:  # one frame at a time keeps the float64 temporaries small
            f = frame.astype(xp.float64)
            self.s += f
            f *= f
            self.ss += f
        self.n += batch.shape[0]

    def begin_second_pass(self):
        xp = xp_of(self.s)
        mean = self.s / self.n
        std = xp.sqrt(xp.maximum(self.ss / self.n - mean * mean, 0))
        band = xp.maximum(self.sigma * std, 1.0 / 255)
        self.mean = mean.astype(xp.float32)
        self.lo, self.hi = (mean - band).astype(xp.float32), (mean + band).astype(xp.float32)
        self.s = self.ss = None
        self.inlier_sum = xp.zeros(self.mean.shape, xp.float64)
        self.inlier_n = xp.zeros(self.mean.shape, xp.float32)

    def update_second_pass(self, batch):
        xp = xp_of(batch)
        keep = (batch >= self.lo) & (batch <= self.hi)
        self.inlier_sum += _batch_sum(xp.where(keep, batch, 0))
        self.inlier_n += keep.sum(axis=0, dtype=xp.float32)

    def result(self):
        xp = xp_of(self.mean if self.mean is not None else self.s)
        if self.mean is None:  # second pass never ran
            return (self.s / self.n).astype(xp.float32)
        out = xp.where(self.inlier_n > 0, self.inlier_sum / xp.maximum(self.inlier_n, 1), self.mean)
        return xp.clip(out, 0, 1).astype(xp.float32)
