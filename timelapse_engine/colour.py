"""RGB -> Y'CbCr packing and downscaling for video output, on the GPU or CPU.

Doing this before FFmpeg means FFmpeg only encodes: no 20 MP colour
conversion on the CPU, and a third less data through the pipe (half for
4:2:0, and previews are downscaled before they are sent at all).

Conversion uses the BT.709 matrix with limited ("TV") range, and the FFmpeg
outputs are tagged to match, which is what editors and players expect for
HD/UHD video.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from .backend import xp_of

KR, KB = 0.2126, 0.0722  # BT.709

# pix_fmt -> (chroma subsampling (x, y), bits, layout)
FORMATS = {
    "yuv422p10le": ((2, 1), 10, "planar"),
    "yuv420p10le": ((2, 2), 10, "planar"),
    "p010le": ((2, 2), 10, "semiplanar_msb"),
    "yuv420p": ((2, 2), 8, "planar"),
    "yuv444p16le": ((1, 1), 16, "planar"),
}


def output_size(height: int, width: int, target: Optional[Tuple[Optional[int], Optional[int]]]) -> Tuple[int, int]:
    """Even (h, w) for a (target_h, target_w) spec where one side may be None (keep aspect).
    Never upscales."""
    if target is None:
        return height - height % 2, width - width % 2
    th, tw = target
    if th is not None and th < height:
        h, w = th, width * th / height
    elif tw is not None and tw < width:
        h, w = height * tw / width, tw
    else:
        h, w = height, width
    return int(round(h / 2)) * 2, int(round(w / 2)) * 2


@lru_cache(maxsize=16)
def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix that averages input pixels over each output pixel's footprint."""
    scale = n_in / n_out
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        start, end = i * scale, (i + 1) * scale
        for j in range(int(np.floor(start)), min(int(np.ceil(end)), n_in)):
            m[i, j] = (min(end, j + 1) - max(start, j)) / scale
    return m


def resize_area(img, h: int, w: int):
    """Area-average downscale of (H, W, C) float32 using two matrix multiplies (fast on GPU)."""
    xp = xp_of(img)
    H, W, C = img.shape
    if (H, W) == (h, w):
        return img
    ry = xp.asarray(_area_weights(H, h))
    rx = xp.asarray(_area_weights(W, w))
    tmp = (ry @ img.reshape(H, W * C)).reshape(h, W, C)        # rows
    out = rx @ tmp.transpose(1, 0, 2).reshape(W, h * C)         # columns
    return out.reshape(w, h, C).transpose(1, 0, 2)


def _subsample(plane, sx: int, sy: int):
    if sx == 2:
        plane = (plane[:, 0::2] + plane[:, 1::2]) * 0.5
    if sy == 2:
        plane = (plane[0::2, :] + plane[1::2, :]) * 0.5
    return plane


def pack(frame, pix_fmt: str, target: Optional[Tuple[Optional[int], Optional[int]]] = None):
    """Convert an RGB frame (uint16 or float 0-1, (H, W, 3)) into the raw bytes FFmpeg expects
    for ``pix_fmt``. Returns (flat array on the same device, (height, width))."""
    xp = xp_of(frame)
    (sx, sy), bits, layout = FORMATS[pix_fmt]
    if frame.ndim == 2 or frame.shape[2] == 1:
        frame = xp.repeat(frame.reshape(frame.shape[0], frame.shape[1], 1), 3, axis=2)
    h, w = output_size(frame.shape[0], frame.shape[1], target)

    rgb = frame.astype(xp.float32)
    if frame.dtype == xp.uint16:
        rgb *= xp.float32(1 / 65535)
    elif frame.dtype == xp.uint8:
        rgb *= xp.float32(1 / 255)
    if target is not None:
        rgb = resize_area(rgb, h, w)
    else:
        rgb = rgb[:h, :w]
    xp.clip(rgb, 0.0, 1.0, out=rgb)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = KR * r + (1 - KR - KB) * g + KB * b
    cb = _subsample((b - y) * (0.5 / (1 - KB)), sx, sy)
    cr = _subsample((r - y) * (0.5 / (1 - KR)), sx, sy)

    if bits == 16:  # full range, used for 4:4:4 16-bit
        q = 65535.0
        planes = [y * q, (cb + 0.5) * q, (cr + 0.5) * q]
        top = 65535
    else:
        step = 2 ** (bits - 8)
        planes = [(16 + 219 * y) * step, (128 + 224 * cb) * step, (128 + 224 * cr) * step]
        top = (1 << bits) - 1
    dtype = xp.uint8 if bits == 8 else xp.uint16
    planes = [xp.clip(xp.rint(p), 0, top).astype(dtype) for p in planes]

    if layout == "semiplanar_msb":  # P010: Y plane, then interleaved UV, values in the high bits
        shift = dtype(16 - bits)
        uv = xp.stack([planes[1], planes[2]], axis=-1)
        return xp.concatenate([(planes[0] << shift).ravel(), (uv << shift).ravel()]), (h, w)
    return xp.concatenate([p.ravel() for p in planes]), (h, w)
