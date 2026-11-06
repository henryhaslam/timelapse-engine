"""Stacking / blending processors.

Every processor follows the same small contract, so the same code is used for
single-image stacks *and* rolling-window video blends:

    proc.reset()                 # clear state
    proc.update(batch)           # batch: float32 (N, H, W, C), 0-1; call repeatedly
    proc.result()                # -> float32 (H, W, C) or (H, W), 0-1

Two-pass processors (``two_pass = True``) are fed every frame a second time:

    proc.begin_second_pass(); proc.update_second_pass(batch) ...; proc.result()

``reset()`` must leave the processor ready for a fresh run, and ``update``
does all of the accumulation (there is no separate "init with first batch"
step, so no batch is ever counted twice).
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import cv2
import numpy as np

from .progress import logger

try:  # CuPy is optional; GPU processors fall back to NumPy without it.
    import cupy as cp

    cp.zeros(1)  # fail early if CUDA isn't usable
    HAS_CUPY = True
except Exception:  # pragma: no cover - depends on machine
    cp = None
    HAS_CUPY = False


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


def _hsv_to_rgb_float(hue_0_179: np.ndarray, value_0_1: np.ndarray) -> np.ndarray:
    """Build a fully saturated colour image from an OpenCV hue map and a brightness map."""
    hue = np.clip(hue_0_179, 0, 179).astype(np.uint8)
    val = (np.clip(value_0_1, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    sat = np.full_like(hue, 255)
    rgb = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0


class Processor:
    """Base class; see module docstring for the contract."""

    name = "base"
    two_pass = False
    description = ""

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, batch: np.ndarray) -> None:
        raise NotImplementedError

    def result(self) -> np.ndarray:
        raise NotImplementedError

    def begin_second_pass(self) -> None:
        pass

    def update_second_pass(self, batch: np.ndarray) -> None:
        pass


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

class AverageProcessor(Processor):
    name = "average"
    description = "Mean of all frames (long-exposure look)."

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = None
        self.count = 0

    def update(self, batch):
        s = batch.sum(axis=0, dtype=np.float64)
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
        self.image = m if self.image is None else np.maximum(self.image, m)

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
        self.image = m if self.image is None else np.minimum(self.image, m)

    def result(self):
        return self.image


class MedianProcessor(Processor):
    name = "median"
    description = "Median per pixel (removes transient objects). Holds every frame in RAM as 16-bit."

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
    description = "Most common 8-bit value per pixel. Holds every frame in RAM as 8-bit."

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

    def __init__(self):
        self.reset()

    def reset(self):
        self.hi = None
        self.lo = None

    def update(self, batch):
        hi, lo = batch.max(axis=0), batch.min(axis=0)
        self.hi = hi if self.hi is None else np.maximum(self.hi, hi)
        self.lo = lo if self.lo is None else np.minimum(self.lo, lo)

    def result(self):
        tce = (self.hi - self.lo) / (self.hi + self.lo + 1e-6)
        return np.clip(tce, 0.0, 1.0).astype(np.float32)


class TemporalStdProcessor(Processor):
    name = "temporal_std"
    description = "Standard deviation per pixel, normalised so the most variable pixel is white."

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = None
        self.total_sq = None
        self.count = 0

    def update(self, batch):
        b = batch.astype(np.float64)
        s, sq = b.sum(axis=0), np.square(b).sum(axis=0)
        if self.total is None:
            self.total, self.total_sq = s, sq
        else:
            self.total += s
            self.total_sq += sq
        self.count += batch.shape[0]

    def result(self):
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.total_sq / self.count - mean * mean, 0.0))
        peak = std.max()
        return (std / peak if peak > 0 else std).astype(np.float32)


class TemporalGradientProcessor(Processor):
    name = "temporal_gradient"
    description = "Sum of frame-to-frame changes, normalised and auto-brightened."

    def __init__(self, target_mean: float = 100.0 / 255.0):
        self.target_mean = target_mean
        self.reset()

    def reset(self):
        self.prev = None
        self.accum = None

    def update(self, batch):
        for frame in batch:
            if self.prev is None:
                self.accum = np.zeros_like(frame)
            else:
                self.accum += np.abs(frame - self.prev)
            self.prev = frame

    def result(self):
        out = self.accum / (self.accum.max() + 1e-6)
        mean = out.mean()
        if mean < self.target_mean:
            out = out * (self.target_mean / (mean + 1e-6))
        return np.clip(out, 0.0, 1.0).astype(np.float32)


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
        for frame in batch:
            if self.prev is None:
                h, w = frame.shape[:2]
                self.hue_accum = np.zeros((h, w), np.float32)
                self.motion_accum = np.zeros((h, w), np.float32)
            else:
                diff = np.abs(frame - self.prev).mean(axis=2)
                hue = (self.frame_index % self.hue_period) * (179.0 / self.hue_period)
                self.hue_accum += diff * hue
                self.motion_accum += diff
            self.prev = frame
            self.frame_index += 1

    def result(self):
        hue = np.nan_to_num(self.hue_accum / (self.motion_accum + 1e-6))
        value = self.motion_accum / (self.motion_accum.max() + 1e-6)
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
        self.use_gpu = HAS_CUPY if use_gpu is None else (use_gpu and HAS_CUPY)
        self.xp = cp if self.use_gpu else np
        self.reset()

    def reset(self):
        self.hist = None  # flat (pixels * bins), uint16 -> upgraded to uint32 if needed
        self.prev = None
        self.shape = None
        self.count = 0
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()

    def _add_frame(self, frame):
        xp = self.xp
        if self.use_gpu:
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
            self.prev = frame
            self._add_frame(xp.ascontiguousarray(frame))
            self.count += 1

    def result(self):
        xp = self.xp
        pixels = int(np.prod(self.shape))
        hist = self.hist.reshape(pixels, self.hist_bins)
        entropy = np.empty(pixels, np.float32)
        chunk = 4_000_000
        for start in range(0, pixels, chunk):
            p = hist[start:start + chunk].astype(xp.float32) / np.float32(self.count)
            e = -(p * xp.log2(xp.maximum(p, np.float32(1e-12)))).sum(axis=1)
            entropy[start:start + chunk] = cp.asnumpy(e) if self.use_gpu else e
        entropy /= np.log2(self.hist_bins)
        return np.clip(entropy.reshape(self.shape), 0.0, 1.0)


class OpticalFlowProcessor(Processor):
    name = "optical_flow_accum"
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


class MotionOverlayProcessor(Processor):
    """Pass 1 builds an average background; pass 2 pastes moving objects onto it."""

    name = "motion_overlay"
    two_pass = True
    description = "Moving objects pasted onto a clean average background."

    def __init__(self, motion_threshold: float = 22, intensity_priority: bool = True, close_iterations: int = 2):
        self.motion_threshold = normalise_threshold(motion_threshold)
        self.intensity_priority = intensity_priority
        self.close_iterations = close_iterations
        self._kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        self.reset()

    def reset(self):
        self.total = None
        self.count = 0
        self.background = None
        self.canvas = None
        self.max_deviation = None

    def update(self, batch):
        s = batch.sum(axis=0, dtype=np.float64)
        self.total = s if self.total is None else self.total + s
        self.count += batch.shape[0]

    def begin_second_pass(self):
        self.background = (self.total / self.count).astype(np.float32)
        self.total = None
        self.canvas = self.background.copy()
        self.max_deviation = np.zeros(self.background.shape[:2], np.float32)

    def update_second_pass(self, batch):
        for frame in batch:
            deviation = np.abs(frame - self.background).max(axis=2)
            mask = (deviation > self.motion_threshold).astype(np.uint8)
            if self.close_iterations:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=self.close_iterations)
            mask = mask.astype(bool)
            if self.intensity_priority:
                mask &= deviation > self.max_deviation
                self.max_deviation[mask] = deviation[mask]
            self.canvas[mask] = frame[mask]

    def result(self):
        if self.canvas is None:  # second pass never ran
            return (self.total / self.count).astype(np.float32)
        return np.clip(self.canvas, 0.0, 1.0)


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
