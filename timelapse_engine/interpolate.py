"""Motion-compensated frame interpolation for smoother (or slow-motion) timelapse video.

Optical flow is estimated on small copies of two consecutive output frames
(OpenCV Farneback), scaled up, and used to warp both full-resolution frames
towards each in-between moment on the GPU, then blended. Works best for
clouds, water and smooth camera-free motion; fast-moving small objects can
show warping artefacts.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from . import colour
from .backend import asnumpy, xp_of

FLOW_SIDE = 960  # optical flow is computed at this size (longest side)


def _ndimage(xp):
    if xp is np:
        import scipy.ndimage as nd
    else:
        import cupyx.scipy.ndimage as nd
    return nd


def _gray_small(frame, side: int):
    h, w = frame.shape[:2]
    s = min(1.0, side / max(h, w))
    small = colour.resize_area(frame.astype(xp_of(frame).float32), max(1, round(h * s)), max(1, round(w * s)))
    y = small[..., 0] * 0.2126 + small[..., 1] * 0.7152 + small[..., 2] * 0.0722 if small.shape[2] >= 3 \
        else small[..., 0]
    return (np.clip(asnumpy(y), 0, 1) * 255 + 0.5).astype(np.uint8), s


class FrameInterpolator:
    """``in_between(a, b)`` returns ``factor - 1`` frames between float frames a and b."""

    def __init__(self, factor: int = 2, flow_side: int = FLOW_SIDE):
        if factor < 1:
            raise ValueError("factor must be >= 1")
        self.factor = int(factor)
        self.flow_side = flow_side

    def _flow(self, a, b):
        ga, s = _gray_small(a, self.flow_side)
        gb, _ = _gray_small(b, self.flow_side)
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None, pyr_scale=0.5, levels=5, winsize=25, iterations=3,
                                            poly_n=7, poly_sigma=1.5, flags=0)
        h, w = a.shape[:2]
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / s   # pixels at full size
        return flow

    def in_between(self, a, b) -> List:
        if self.factor == 1:
            return []
        xp = xp_of(a)
        nd = _ndimage(xp)
        a = a.astype(xp.float32)
        b = b.astype(xp.float32)
        if a.ndim == 2:
            a, b = a[..., None], b[..., None]
        flow = xp.asarray(self._flow(a, b))
        fx, fy = flow[..., 0], flow[..., 1]
        h, w = a.shape[:2]
        yy, xx = xp.meshgrid(xp.arange(h, dtype=xp.float32), xp.arange(w, dtype=xp.float32), indexing="ij")
        out = []
        for j in range(1, self.factor):
            t = j / self.factor
            frame = xp.empty_like(a)
            coords_a = xp.stack([yy - t * fy, xx - t * fx])
            coords_b = xp.stack([yy + (1 - t) * fy, xx + (1 - t) * fx])
            for c in range(a.shape[2]):
                wa = nd.map_coordinates(a[..., c], coords_a, order=1, mode="nearest")
                wb = nd.map_coordinates(b[..., c], coords_b, order=1, mode="nearest")
                frame[..., c] = (1 - t) * wa + t * wb
            out.append(frame)
        return out
