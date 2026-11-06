"""Stacking / blending processors.

Every processor follows the same small contract, so the same code is used for
single-image stacks *and* rolling-window video blends:

    proc.reset()                 # clear state
    proc.update(batch)           # batch: float32 (N, H, W, C), 0-1; call repeatedly
    proc.result()                # -> float32 (H, W, C) or (H, W), 0-1

Two-pass processors (``two_pass = True``) are fed every frame a second time:

    proc.begin_second_pass(); proc.update_second_pass(batch) ...; proc.result()

Processors with ``gpu = True`` are written against the NumPy API and work
unchanged on CuPy arrays: the engine hands them GPU batches when CuPy is
available, and their state stays on the GPU. ``result()`` may return a CuPy
array; use ``backend.asnumpy`` (the engine does this for you).
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import cv2
import numpy as np

from .backend import HAS_CUPY, asnumpy, cp, xp_of


def normalise_threshold(value: float) -> float:
    """Accept thresholds on the 0-255 scale (e.g. 22) or the 0-1 scale (e.g. 0.086)."""
    value = float(value)
    return value / 255.0 if value > 1.0 else value


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    """float 0-1 (H, W, C) -> uint8 grayscale (H, W)."""
    u8 = (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if u8.ndim == 2:
        return u8
    if u8.shape[2] == 1:
        return u8[:, :, 0]
    return cv2.cvtColor(np.ascontiguousarray(u8[:, :, :3]), cv2.COLOR_RGB2GRAY)


def _hsv_to_rgb_float(hue_0_179, value_0_1) -> np.ndarray:
    """Build a fully saturated colour image from an OpenCV hue map and a brightness map."""
    hue = np.clip(asnumpy(hue_0_179), 0, 179).astype(np.uint8)
    val = (np.clip(asnumpy(value_0_1), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    sat = np.full_like(hue, 255)
    rgb = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0


def _channel_max(x):
    """Max over the last (colour) axis. Element-wise per channel because CuPy's
    reduction over a tiny innermost axis is ~15x slower than this."""
    xp = xp_of(x)
    out = x[..., 0]
    for c in range(1, x.shape[-1]):
        out = xp.maximum(out, x[..., c])
    return out


def _channel_mean(x):
    """Mean over the last (colour) axis (see _channel_max for why it's written this way)."""
    out = x[..., 0].copy()
    for c in range(1, x.shape[-1]):
        out += x[..., c]
    out *= 1.0 / x.shape[-1]
    return out


def _normalise_peak(img, floor: float = 1e-6):
    """Scale so the brightest pixel is 1. Below ``floor`` the image is treated as
    all-zero, so rounding noise in a static scene isn't stretched to full white."""
    xp = xp_of(img)
    peak = float(img.max())
    if peak < floor:
        return xp.zeros(img.shape, xp.float32)
    return (img / peak).astype(xp.float32)


def _batch_sum(batch):
    """Sum a batch over frames: float32 within the batch, float64 across batches
    (consumer GPUs are very slow at float64 maths, but fine at float64 adds)."""
    xp = xp_of(batch)
    return batch.sum(axis=0, dtype=xp.float32).astype(xp.float64)


class Processor:
    """Base class; see module docstring for the contract."""

    name = "base"
    two_pass = False
    gpu = True
    description = ""
    state_frames = 2.0  # memory held between updates, in units of one float32 frame

    def state_bytes(self, frame_shape) -> int:
        """Device memory the processor's running state needs (used to size chunks)."""
        return int(self.state_frames * np.prod(frame_shape) * 4)

    def stored_bytes(self, n_frames: int, frame_shape) -> int:
        """Host RAM for processors that must keep every frame (median, mode)."""
        return 0

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, batch) -> None:
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def begin_second_pass(self) -> None:
        pass

    def update_second_pass(self, batch) -> None:
        pass

    # Optional sliding-window support: processors that can *remove* a frame
    # (running sums) set ``sliding = True`` and implement the three methods
    # below on raw (integer) frames. Rolling-window renders then cost
    # ``step`` frames per output instead of ``blend`` frames.
    sliding = False

    def window_add(self, raw_frame) -> None:
        raise NotImplementedError

    def window_remove(self, raw_frame) -> None:
        raise NotImplementedError

    def window_result(self, scale: float):
        raise NotImplementedError


class _RunningSums:
    """Exact running sums of raw frames (and optionally their squares).

    Integer frames are summed in int64, so adding and later subtracting a frame
    leaves no rounding error behind however many windows slide past.
    """

    def __init__(self, squares: bool):
        self.squares = squares
        self.clear()

    def clear(self):
        self.total = self.total_sq = None
        self.count = 0

    def _acc_dtype(self, xp, frame):
        return xp.int64 if np.issubdtype(frame.dtype, np.integer) else xp.float64

    def add(self, frame, sign: int = 1):
        xp = xp_of(frame)
        if self.total is None:
            self.total = xp.zeros(frame.shape, self._acc_dtype(xp, frame))
            if self.squares:
                self.total_sq = xp.zeros(frame.shape, self._acc_dtype(xp, frame))
        f = frame.astype(self.total.dtype)
        if sign > 0:
            self.total += f
        else:
            self.total -= f
        if self.squares:
            f *= f
            if sign > 0:
                self.total_sq += f
            else:
                self.total_sq -= f
        self.count += sign


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

class AverageProcessor(Processor):
    name = "average"
    description = "Mean of all frames (long-exposure look)."
    state_frames = 4.0
    sliding = True

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = None
        self.count = 0
        self.window = _RunningSums(squares=False)

    def window_add(self, raw_frame):
        self.window.add(raw_frame, +1)

    def window_remove(self, raw_frame):
        self.window.add(raw_frame, -1)

    def window_result(self, scale):
        w = self.window
        return (w.total * (scale / w.count)).astype(np.float32)

    def update(self, batch):
        s = _batch_sum(batch)
        self.total = s if self.total is None else self.total + s
        self.count += batch.shape[0]

    def result(self):
        return (self.total / self.count).astype(np.float32)


class MaximumProcessor(Processor):
    name = "maximum"
    description = "Brightest value per pixel (light trails, star trails)."

    def __init__(self):
        self.reset()

    def reset(self):
        self.image = None

    def update(self, batch):
        m = batch.max(axis=0)
        self.image = m if self.image is None else xp_of(m).maximum(self.image, m)

    def result(self):
        return self.image


class MinimumProcessor(Processor):
    name = "minimum"
    description = "Darkest value per pixel."

    def __init__(self):
        self.reset()

    def reset(self):
        self.image = None

    def update(self, batch):
        m = batch.min(axis=0)
        self.image = m if self.image is None else xp_of(m).minimum(self.image, m)

    def result(self):
        return self.image


class MedianProcessor(Processor):
    name = "median"
    gpu = False
    description = "Median per pixel (removes transient objects). Holds every frame in RAM as 16-bit."

    def stored_bytes(self, n_frames, frame_shape):
        return int(n_frames * np.prod(frame_shape) * 2 * 2)  # frames + np.median's working copy

    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = []

    def update(self, batch):
        # Store as uint16 (half the memory of float32, no visible precision loss).
        self.frames.append((np.clip(batch, 0, 1) * 65535.0 + 0.5).astype(np.uint16))

    def result(self):
        stack = np.concatenate(self.frames, axis=0)
        return (np.median(stack, axis=0) / 65535.0).astype(np.float32)


class ModeProcessor(Processor):
    name = "mode"
    gpu = False
    description = "Most common 8-bit value per pixel. Holds every frame in RAM as 8-bit."

    def stored_bytes(self, n_frames, frame_shape):
        return int(n_frames * np.prod(frame_shape) * 4)  # frames + scipy's sort/count temporaries

    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = []

    def update(self, batch):
        self.frames.append((np.clip(batch, 0, 1) * 255.0 + 0.5).astype(np.uint8))

    def result(self):
        from scipy import stats

        stack = np.concatenate(self.frames, axis=0)
        mode, _ = stats.mode(stack, axis=0, keepdims=False)
        return mode.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

class TemporalContrastProcessor(Processor):
    name = "temporal_contrast"
    description = "(max - min) / (max + min) per pixel: how much each pixel changed relative to its brightness."
    state_frames = 4.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.hi = None
        self.lo = None

    def update(self, batch):
        xp = xp_of(batch)
        hi, lo = batch.max(axis=0), batch.min(axis=0)
        self.hi = hi if self.hi is None else xp.maximum(self.hi, hi)
        self.lo = lo if self.lo is None else xp.minimum(self.lo, lo)

    def result(self):
        xp = xp_of(self.hi)
        tce = (self.hi - self.lo) / (self.hi + self.lo + xp.float32(1e-6))
        return xp.clip(tce, 0.0, 1.0).astype(xp.float32)


class TemporalStdProcessor(Processor):
    name = "temporal_std"
    description = "Standard deviation per pixel, normalised so the most variable pixel is white."
    state_frames = 8.0
    sliding = True

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = None
        self.total_sq = None
        self.count = 0
        self.window = _RunningSums(squares=True)

    def window_add(self, raw_frame):
        self.window.add(raw_frame, +1)

    def window_remove(self, raw_frame):
        self.window.add(raw_frame, -1)

    def window_result(self, scale):
        w = self.window
        xp = xp_of(w.total)
        n = w.count
        if w.total.dtype == xp.int64:
            # n*Σx² - (Σx)² is exact in integers, so a static pixel gets exactly 0.
            var = (n * w.total_sq - w.total * w.total) * (scale * scale / (n * n))
        else:
            mean = w.total / n
            var = w.total_sq / n - mean * mean
        return _normalise_peak(xp.sqrt(xp.maximum(var, 0.0)))

    def update(self, batch):
        # float64 here: E[x²] - E[x]² cancels badly in float32 (a static pixel would get noise)
        b = batch.astype(xp_of(batch).float64)
        s, sq = b.sum(axis=0), (b * b).sum(axis=0)
        if self.total is None:
            self.total, self.total_sq = s, sq
        else:
            self.total += s
            self.total_sq += sq
        self.count += batch.shape[0]

    def result(self):
        xp = xp_of(self.total)
        mean = self.total / self.count
        return _normalise_peak(xp.sqrt(xp.maximum(self.total_sq / self.count - mean * mean, 0.0)))


class TemporalGradientProcessor(Processor):
    name = "temporal_gradient"
    description = "Sum of frame-to-frame changes, normalised and auto-brightened."
    state_frames = 3.0

    def __init__(self, target_mean: float = 100.0 / 255.0):
        self.target_mean = target_mean
        self.reset()

    def reset(self):
        self.prev = None
        self.accum = None

    def update(self, batch):
        xp = xp_of(batch)
        if self.accum is None:
            self.accum = xp.zeros(batch.shape[1:], xp.float32)
        else:
            self.accum += xp.abs(batch[0] - self.prev)
        if batch.shape[0] > 1:
            self.accum += xp.abs(xp.diff(batch, axis=0)).sum(axis=0)
        self.prev = batch[-1].copy()

    def result(self):
        xp = xp_of(self.accum)
        out = self.accum / (float(self.accum.max()) + 1e-6)
        mean = float(out.mean())
        if mean < self.target_mean:
            out *= self.target_mean / (mean + 1e-6)
        return xp.clip(out, 0.0, 1.0).astype(xp.float32)


class TemporalHueMotionProcessor(Processor):
    name = "temporal_hue_motion"
    description = "Colours movement by when it happened (hue = time), brightness = amount of motion."

    def __init__(self, hue_period: int = 360):
        self.hue_period = hue_period
        self.reset()

    def reset(self):
        self.prev = None
        self.hue_accum = None
        self.motion_accum = None
        self.frame_index = 0

    def update(self, batch):
        xp = xp_of(batch)
        n = batch.shape[0]
        diffs = []
        if self.prev is None:
            self.hue_accum = xp.zeros(batch.shape[1:3], xp.float32)
            self.motion_accum = xp.zeros(batch.shape[1:3], xp.float32)
            first = self.frame_index + 1  # the very first frame has nothing to compare with
        else:
            diffs.append(_channel_mean(xp.abs(batch[0] - self.prev))[None])
            first = self.frame_index
        if n > 1:
            diffs.append(_channel_mean(xp.abs(xp.diff(batch, axis=0))))
        if diffs:
            d = xp.concatenate(diffs) if len(diffs) > 1 else diffs[0]
            idx = xp.arange(first, self.frame_index + n)
            hue = (idx % self.hue_period).astype(xp.float32) * xp.float32(179.0 / self.hue_period)
            self.hue_accum += xp.tensordot(hue, d, axes=1)
            self.motion_accum += d.sum(axis=0)
        self.prev = batch[-1].copy()
        self.frame_index += n

    def result(self):
        xp = xp_of(self.hue_accum)
        hue = xp.nan_to_num(self.hue_accum / (self.motion_accum + xp.float32(1e-6)))
        value = self.motion_accum / (float(self.motion_accum.max()) + 1e-6)
        return _hsv_to_rgb_float(hue, value)


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


class TemporalEntropyProcessor(Processor):
    name = "temporal_entropy"
    description = "Shannon entropy of each pixel's values over time (GPU accelerated when CuPy is available)."

    def __init__(self, hist_bins: int = 32, noise_threshold: float = 3.0, use_gpu: Optional[bool] = None):
        self.hist_bins = int(hist_bins)
        self.noise_threshold = normalise_threshold(noise_threshold)
        self.gpu = HAS_CUPY if use_gpu is None else bool(use_gpu and HAS_CUPY)
        self.xp = cp if self.gpu else np
        self.reset()

    def state_bytes(self, frame_shape):
        # uint16 histogram (uint32 past 65535 frames) + previous frame + temporaries
        return int(np.prod(frame_shape) * (self.hist_bins * 2 + 3 * 4))

    def reset(self):
        self.hist = None  # flat (pixels * bins), uint16 -> upgraded to uint32 if needed
        self.prev = None
        self.shape = None
        self.count = 0

    def _add_frame(self, frame):
        if self.gpu:
            _entropy_kernel()(frame.ravel(), np.int32(self.hist_bins), self.hist)
        else:
            bins = np.clip(np.floor(frame.ravel() * self.hist_bins), 0, self.hist_bins - 1).astype(np.int64)
            idx = np.arange(bins.size, dtype=np.int64) * self.hist_bins + bins
            self.hist[idx] += 1  # indices are unique within a frame, so plain += is safe

    def update(self, batch):
        xp = self.xp
        if self.hist is None:
            self.shape = batch.shape[1:]
            self.hist = xp.zeros(int(np.prod(self.shape)) * self.hist_bins, dtype=xp.uint16)
        if self.hist.dtype == xp.uint16 and self.count + batch.shape[0] > np.iinfo(np.uint16).max:
            self.hist = self.hist.astype(xp.uint32)

        data = xp.asarray(batch, dtype=xp.float32)
        for frame in data:
            if self.prev is not None and self.noise_threshold > 0:
                # Ignore tiny sensor-noise changes: keep the previous value instead.
                frame = xp.where(xp.abs(frame - self.prev) < self.noise_threshold, self.prev, frame)
            self.prev = frame.copy()
            self._add_frame(xp.ascontiguousarray(frame))
            self.count += 1

    def result(self):
        xp = self.xp
        pixels = int(np.prod(self.shape))
        hist = self.hist.reshape(pixels, self.hist_bins)
        entropy = xp.empty(pixels, xp.float32)
        chunk = 4_000_000
        for start in range(0, pixels, chunk):
            p = hist[start:start + chunk].astype(xp.float32) / xp.float32(self.count)
            entropy[start:start + chunk] = -(p * xp.log2(xp.maximum(p, xp.float32(1e-12)))).sum(axis=1)
        entropy /= xp.float32(np.log2(self.hist_bins))
        return xp.clip(entropy.reshape(self.shape), 0.0, 1.0)


class OpticalFlowProcessor(Processor):
    name = "optical_flow_accum"
    gpu = False
    description = "Accumulated dense optical flow: hue = average direction, brightness = total movement."

    def __init__(self, pyr_scale=0.5, levels=8, winsize=4, iterations=5, poly_n=7, poly_sigma=1.2):
        self.flow_params = dict(pyr_scale=pyr_scale, levels=levels, winsize=winsize,
                                iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=0)
        self.reset()

    def reset(self):
        self.prev = None
        self.mag = None
        self.sum_x = None
        self.sum_y = None

    def update(self, batch):
        for frame in batch:
            gray = _to_gray_u8(frame)
            if self.prev is None:
                self.mag = np.zeros(gray.shape, np.float32)
                self.sum_x = np.zeros(gray.shape, np.float32)
                self.sum_y = np.zeros(gray.shape, np.float32)
            else:
                flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, **self.flow_params)
                fx, fy = flow[..., 0], flow[..., 1]
                self.mag += np.sqrt(fx * fx + fy * fy)
                self.sum_x += fx
                self.sum_y += fy
            self.prev = gray

    def result(self):
        # Direction of the summed flow vector (a proper circular mean of the angles).
        angle = np.arctan2(self.sum_y, self.sum_x) % (2 * np.pi)
        hue = angle * (180.0 / np.pi / 2.0)
        value = self.mag / (self.mag.max() + 1e-6)
        return _hsv_to_rgb_float(hue, value)


def _dilate_cross(mask):
    """3x3 cross dilation over the last two axes (pixels outside the image count as False)."""
    out = mask.copy()
    out[..., 1:, :] |= mask[..., :-1, :]
    out[..., :-1, :] |= mask[..., 1:, :]
    out[..., :, 1:] |= mask[..., :, :-1]
    out[..., :, :-1] |= mask[..., :, 1:]
    return out


def _erode_cross(mask):
    """3x3 cross erosion over the last two axes (pixels outside the image count as True, like OpenCV)."""
    out = mask.copy()
    out[..., 1:, :] &= mask[..., :-1, :]
    out[..., :-1, :] &= mask[..., 1:, :]
    out[..., :, 1:] &= mask[..., :, :-1]
    out[..., :, :-1] &= mask[..., :, 1:]
    return out


def binary_close(mask, iterations: int):
    """Morphological closing with a 3x3 cross; same result as OpenCV's MORPH_CLOSE,
    but works on NumPy or CuPy and on a whole (N, H, W) stack at once."""
    for _ in range(iterations):
        mask = _dilate_cross(mask)
    for _ in range(iterations):
        mask = _erode_cross(mask)
    return mask


class MotionOverlayProcessor(Processor):
    """Pass 1 builds an average background; pass 2 pastes moving objects onto it."""

    name = "motion_overlay"
    two_pass = True
    description = "Moving objects pasted onto a clean average background."
    state_frames = 5.0

    def __init__(self, motion_threshold: float = 22, intensity_priority: bool = True, close_iterations: int = 2):
        self.motion_threshold = normalise_threshold(motion_threshold)
        self.intensity_priority = intensity_priority
        self.close_iterations = close_iterations
        self.reset()

    def reset(self):
        self.total = None
        self.count = 0
        self.background = None
        self.canvas = None
        self.max_deviation = None

    def update(self, batch):
        s = _batch_sum(batch)
        self.total = s if self.total is None else self.total + s
        self.count += batch.shape[0]

    def begin_second_pass(self):
        xp = xp_of(self.total)
        self.background = (self.total / self.count).astype(xp.float32)
        self.total = None
        self.canvas = self.background.copy()
        self.max_deviation = xp.zeros(self.background.shape[:2], xp.float32)

    def update_second_pass(self, batch):
        """Vectorised over the batch. Equivalent to looping frame by frame:
        with intensity priority a pixel takes the frame with the largest
        deviation seen so far (earliest wins ties); otherwise the latest moving frame."""
        xp = xp_of(batch)
        deviation = _channel_max(xp.abs(batch - self.background))  # (N, H, W)
        mask = deviation > self.motion_threshold
        if self.close_iterations:
            mask = binary_close(mask, self.close_iterations)

        if self.intensity_priority:
            masked = xp.where(mask, deviation, xp.float32(-1.0))
            pick = masked.argmax(axis=0)
            best = xp.take_along_axis(masked, pick[None], axis=0)[0]
            accept = best > self.max_deviation
            self.max_deviation = xp.where(accept, best, self.max_deviation)
        else:
            pick = batch.shape[0] - 1 - mask[::-1].argmax(axis=0)
            accept = mask.any(axis=0)

        chosen = xp.take_along_axis(batch, pick[None, :, :, None], axis=0)[0]
        self.canvas = xp.where(accept[:, :, None], chosen, self.canvas)

    def result(self):
        xp = xp_of(self.canvas if self.canvas is not None else self.total)
        if self.canvas is None:  # second pass never ran
            return (self.total / self.count).astype(xp.float32)
        return xp.clip(self.canvas, 0.0, 1.0)


PROCESSORS: Dict[str, Type[Processor]] = {
    cls.name: cls
    for cls in (
        AverageProcessor, MaximumProcessor, MinimumProcessor, MedianProcessor, ModeProcessor,
        TemporalContrastProcessor, TemporalStdProcessor, TemporalGradientProcessor,
        TemporalHueMotionProcessor, TemporalEntropyProcessor, OpticalFlowProcessor,
        MotionOverlayProcessor,
    )
}


def create_processor(name: str, **params) -> Processor:
    """Build a processor by name, e.g. ``create_processor("temporal_entropy", hist_bins=32)``."""
    try:
        cls = PROCESSORS[name]
    except KeyError:
        raise ValueError(f"Unknown process type {name!r}. Choose from: {', '.join(PROCESSORS)}") from None
    try:
        return cls(**params)
    except TypeError as exc:
        raise TypeError(f"Bad parameters for {name!r}: {exc}") from None


def list_processors() -> Dict[str, str]:
    """{name: description} for every processor (useful for a GUI dropdown)."""
    return {name: cls.description for name, cls in PROCESSORS.items()}
