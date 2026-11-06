"""CPU / GPU array backend helpers.

Processors are written once against the NumPy API and run on either NumPy
(CPU) or CuPy (GPU) arrays; ``xp_of(array)`` returns the right module.
"""

from __future__ import annotations

import numpy as np

try:  # CuPy is optional.
    import cupy as cp

    cp.zeros(1)  # fail early if CUDA isn't usable
    HAS_CUPY = True
except Exception:  # pragma: no cover - depends on machine
    cp = None
    HAS_CUPY = False

# Target size of one float32 chunk handed to a processor (CPU RAM or VRAM).
CHUNK_BYTES = 1_500_000_000


def xp_of(array):
    """Return numpy or cupy depending on where ``array`` lives."""
    if HAS_CUPY and isinstance(array, cp.ndarray):
        return cp
    return np


def asnumpy(array) -> np.ndarray:
    if HAS_CUPY and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def scale_for(dtype) -> float:
    """Multiplier that maps a dtype's full range to 0-1."""
    dtype = np.dtype(dtype)
    return 1.0 / np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0


def to_float(raw, scale: float):
    """Raw integer (or float) frames -> float32 0-1, on whichever device ``raw`` is on."""
    xp = xp_of(raw)
    out = raw.astype(xp.float32)
    if scale != 1.0:
        out *= xp.float32(scale)
    return out


def to_uint16(img):
    """float 0-1 -> uint16, on whichever device ``img`` is on."""
    xp = xp_of(img)
    if img.dtype == xp.uint16:
        return img
    if img.dtype == xp.uint8:
        return img.astype(xp.uint16) * xp.uint16(257)
    out = xp.clip(img, 0.0, 1.0)
    out *= xp.float32(65535.0)
    out += xp.float32(0.5)
    return out.astype(xp.uint16)


def frames_per_chunk(frame_shape, chunk_bytes: int = CHUNK_BYTES) -> int:
    """How many float32 frames fit in one processing chunk."""
    return max(1, int(chunk_bytes // (int(np.prod(frame_shape)) * 4)))


# Target size of one raw (uint8/uint16) read batch. Three of these are kept in
# host RAM (pinned when feeding the GPU), so 16-bit frames get half the batch
# size of 8-bit frames instead of double the memory.
RAW_BATCH_BYTES = 1_000_000_000


def auto_batch_size(frame_shape, dtype, target_bytes: int = RAW_BATCH_BYTES) -> int:
    frame_bytes = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize
    return int(min(32, max(2, target_bytes // frame_bytes)))


def gpu_free_bytes() -> int:
    """Free VRAM, counting blocks CuPy has cached but isn't using."""
    if not HAS_CUPY:
        return 0
    pool = cp.get_default_memory_pool()
    free, _ = cp.cuda.runtime.memGetInfo()
    return int(free + pool.free_bytes())


def host_free_bytes() -> int:
    import psutil

    return int(psutil.virtual_memory().available)


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def empty_host(shape, dtype, pinned: bool) -> np.ndarray:
    """Host buffer; page-locked when feeding the GPU (faster, async-capable uploads)."""
    if pinned and HAS_CUPY:
        import cupyx

        return cupyx.empty_pinned(shape, dtype=dtype)
    return np.empty(shape, dtype=dtype)


def free_gpu_memory() -> None:
    if HAS_CUPY:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
