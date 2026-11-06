"""Per-frame adjustments applied before any effect: deflicker.

Flicker in timelapse comes from tiny exposure/aperture differences between
frames. Deflicker measures each frame's brightness (on small, fast-decoded
copies), smooths that brightness curve over time, and scales every frame to
the smooth curve. The scaling is done in linear light, so it behaves like an
exposure change rather than a contrast change.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backend import xp_of
from .progress import CancelToken, ProgressCallback, check_cancel, report

GAMMA = 2.2          # approximate transfer curve of sRGB / Adobe RGB / ProPhoto-encoded frames
MEASURE_SIDE = 320   # brightness is measured on frames this size (longest side)


def measure_brightness(source, progress: Optional[ProgressCallback] = None,
                       cancel: Optional[CancelToken] = None, batch: int = 32) -> np.ndarray:
    """Mean linear-light luminance of every frame of ``source`` (fast: small decodes)."""
    n = len(source)
    levels = np.empty(n, np.float64)
    report(progress, 0, n, "measuring brightness")
    for start in range(0, n, batch):
        check_cancel(cancel)
        idx = list(range(start, min(start + batch, n)))
        for k, frame in zip(idx, source.read_small(idx, MEASURE_SIDE)):
            f = frame.astype(np.float32)
            if np.issubdtype(frame.dtype, np.integer):
                f /= np.iinfo(frame.dtype).max
            lin = np.clip(f, 0, 1) ** GAMMA
            y = lin[..., 0] if lin.shape[-1] == 1 else lin[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            levels[k] = float(y.mean())
        report(progress, idx[-1] + 1, n, "measuring brightness")
    return levels


def deflicker_gains(levels: np.ndarray, window: int = 15, strength: float = 1.0,
                    max_gain: float = 2.0) -> np.ndarray:
    """Gain per frame that moves its brightness onto a smoothed brightness curve.

    window: smoothing width in frames (larger = smoother; removes slower flicker
    but follows real changes like a sunset less closely). strength: 0-1.
    Smoothing is done on log brightness, i.e. in exposure stops.
    """
    from scipy.ndimage import gaussian_filter1d

    levels = np.maximum(np.asarray(levels, np.float64), 1e-6)
    log = np.log(levels)
    target = gaussian_filter1d(log, sigma=max(float(window), 2.0) / 4, mode="mirror")
    gains = np.exp((target - log) * float(strength))
    return np.clip(gains, 1 / max_gain, max_gain)


def apply_gain(frames, gains):
    """Scale float 0-1 frames (N, H, W, C) by per-frame gains in linear light (NumPy or CuPy)."""
    xp = xp_of(frames)
    g = xp.asarray(np.asarray(gains, np.float32)).reshape(-1, *([1] * (frames.ndim - 1)))
    lin = xp.clip(frames, 0, 1) ** xp.float32(GAMMA)
    return xp.clip(lin * g, 0, 1) ** xp.float32(1 / GAMMA)
