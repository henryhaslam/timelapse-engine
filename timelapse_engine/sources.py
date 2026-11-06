"""Frame sources: a uniform way to read frames from an image folder or a video file.

Sources hand out frames in their *native* dtype (uint8 / uint16) as RGB
(H, W, C). The engine keeps them as integers until they are on the GPU (4x
less data to move than float32) and converts to float 0-1 there.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import files
from .progress import logger


class FrameSource(ABC):
    """Base class. Subclasses implement ``__len__`` and ``read_raw``; overriding
    ``read_into`` lets a source decode straight into the engine's buffers."""

    name: str = "source"
    _probe: Optional[Tuple[tuple, np.dtype]] = None

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def read_raw(self, indices: Sequence[int]) -> List[np.ndarray]:
        """Frames in native dtype, RGB, (H, W, C)."""

    def read_into(self, indices: Sequence[int], out: np.ndarray) -> None:
        """Fill ``out[k]`` with frame ``indices[k]``."""
        for k, frame in enumerate(self.read_raw(indices)):
            out[k] = files.as_hwc(frame)

    def read_small(self, indices: Sequence[int], max_side: int) -> List[np.ndarray]:
        """Downscaled frames (native dtype) with the longest side ``max_side``."""
        import cv2

        out = []
        for frame in self.read_raw(indices):
            frame = files.as_hwc(frame)
            h, w = frame.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                frame = files.as_hwc(cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))),
                                                interpolation=cv2.INTER_AREA))
            out.append(frame)
        return out

    def read_many(self, indices: Sequence[int]) -> List[np.ndarray]:
        """Frames as float32 0-1 (convenience; the engine uses the raw methods)."""
        return [files.to_float(f) for f in self.read_raw(indices)]

    def read(self, index: int) -> np.ndarray:
        return self.read_many([index])[0]

    def probe(self) -> Tuple[tuple, np.dtype]:
        """(frame shape (H, W, C), dtype) taken from the first frame."""
        if self._probe is None:
            first = files.as_hwc(self.read_raw([0])[0])
            self._probe = (first.shape, first.dtype)
        return self._probe

    def capture_time(self, index: int) -> Optional[datetime]:
        return None

    def latest_capture_time(self) -> Optional[datetime]:
        return None

    def xmp(self, index: int) -> Optional[bytes]:
        return None

    def frame_info(self, index: int) -> Optional[files.FrameInfo]:
        """Capture details (time, exposure, camera, lens) of a frame, if known."""
        return None

    def frame_path(self, index: int) -> Optional[str]:
        """The file a frame came from (None for video frames)."""
        return None

    @property
    def fps(self) -> Optional[float]:
        return None

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ImageSequence(FrameSource):
    """A folder (or explicit list) of JPEG/TIFF frames.

    Parameters
    ----------
    folder: folder to scan for .jpg/.jpeg/.tif/.tiff files.
    paths: explicit list of files instead of ``folder``.
    sort: "exif" (capture time, falling back to filename) or "name" (natural filename order).
    validate: check every frame has the same size/bit depth before processing.
    workers: threads used to decode frames in parallel.
    """

    def __init__(self, folder=None, paths: Optional[Sequence[str]] = None, sort: str = "exif",
                 validate: bool = True, workers: int = 16, exiftool_path: Optional[str] = None):
        if (folder is None) == (paths is None):
            raise ValueError("Pass exactly one of 'folder' or 'paths'.")
        if paths is None:
            paths = files.list_images(folder)
            self.name = files.shoot_name(folder)
        else:
            paths = [str(p) for p in paths]
            self.name = files.shoot_name(Path(paths[0]).parent) if paths else "sequence"
        if not paths:
            raise FileNotFoundError(f"No JPEG/TIFF images found in {folder}")

        self.workers = workers
        self._pool: Optional[ThreadPoolExecutor] = None

        self._exiftool_path = exiftool_path
        self._info: Optional[Dict[str, files.FrameInfo]] = None
        if sort == "exif":
            self._info = files.read_frame_info(paths, exiftool_path, workers)
            self._times = {p: fi.time for p, fi in self._info.items()}
            self.paths = files.sort_by_capture_time(paths, self._times)
            missing = sum(t is None for t in self._times.values())
            if missing:
                logger.warning("%d of %d frames have no EXIF capture time; they are ordered by filename at the end.",
                               missing, len(paths))
        elif sort == "name":
            self._times = None
            self.paths = sorted(paths, key=files.natural_sort_key)
        else:
            raise ValueError(f"sort must be 'exif' or 'name', not {sort!r}")

        logger.info("%d frames found in %s (sorted by %s).", len(self.paths), self.name, sort)
        if validate:
            files.validate_sequence(self.paths, workers)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="decode")
        return self._pool

    def read_raw(self, indices: Sequence[int]) -> List[np.ndarray]:
        return list(self.pool.map(files.load_raw, [self.paths[i] for i in indices]))

    def read_small(self, indices: Sequence[int], max_side: int) -> List[np.ndarray]:
        return list(self.pool.map(lambda i: files.load_small(self.paths[i], max_side), indices))

    def read_into(self, indices: Sequence[int], out: np.ndarray) -> None:
        jobs = [self.pool.submit(files.decode_into, self.paths[i], out[k]) for k, i in enumerate(indices)]
        for job in jobs:
            job.result()

    def _ensure_info(self) -> Dict[str, files.FrameInfo]:
        if self._info is None:
            self._info = files.read_frame_info(self.paths, self._exiftool_path, self.workers)
        return self._info

    def _ensure_times(self):
        if self._times is None:
            self._times = {p: fi.time for p, fi in self._ensure_info().items()}
        return self._times

    def frame_info(self, index: int) -> Optional[files.FrameInfo]:
        return self._ensure_info().get(self.paths[index])

    def frame_path(self, index: int) -> Optional[str]:
        return self.paths[index]

    def capture_time(self, index: int) -> Optional[datetime]:
        return self._ensure_times().get(self.paths[index])

    def latest_capture_time(self) -> Optional[datetime]:
        times = [t for t in self._ensure_times().values() if t is not None]
        return max(times) if times else None

    def xmp(self, index: int) -> Optional[bytes]:
        return files.read_xmp(self.paths[index])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


class SubsetSource(FrameSource):
    """A selection of frames from another source (a frame range, or a spread of
    frames), optionally downscaled so the longest side is ``max_side`` - used
    for frame ranges and for fast low-resolution previews."""

    def __init__(self, source: FrameSource, indices: Sequence[int], max_side: Optional[int] = None):
        self.source = source
        self.indices = list(indices)
        if not self.indices:
            raise ValueError("SubsetSource needs at least one frame.")
        self.max_side = max_side
        self.name = source.name

    def __len__(self) -> int:
        return len(self.indices)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        frame = files.as_hwc(frame)
        h, w = frame.shape[:2]
        if not self.max_side or max(h, w) <= self.max_side:
            return frame
        import cv2

        s = self.max_side / max(h, w)
        small = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        return files.as_hwc(small)

    def probe(self):
        if not self.max_side:
            return self.source.probe()  # same frames, no need to decode one again
        return super().probe()

    def read_raw(self, indices: Sequence[int]) -> List[np.ndarray]:
        mapped = [self.indices[i] for i in indices]
        if self.max_side:
            return self.source.read_small(mapped, self.max_side)
        return [files.as_hwc(f) for f in self.source.read_raw(mapped)]

    def read_small(self, indices: Sequence[int], max_side: int) -> List[np.ndarray]:
        side = min(max_side, self.max_side) if self.max_side else max_side
        return self.source.read_small([self.indices[i] for i in indices], side)

    def read_into(self, indices: Sequence[int], out: np.ndarray) -> None:
        if self.max_side is None or self.probe()[0] == self.source.probe()[0]:
            self.source.read_into([self.indices[i] for i in indices], out)  # no resize: fast path
        else:
            super().read_into(indices, out)

    def capture_time(self, index: int) -> Optional[datetime]:
        return self.source.capture_time(self.indices[index])

    def latest_capture_time(self) -> Optional[datetime]:
        times = [t for t in (self.source.capture_time(i) for i in self.indices) if t is not None]
        return max(times) if times else None

    def xmp(self, index: int) -> Optional[bytes]:
        return self.source.xmp(self.indices[index])

    def frame_info(self, index: int) -> Optional[files.FrameInfo]:
        return self.source.frame_info(self.indices[index])

    def frame_path(self, index: int) -> Optional[str]:
        return self.source.frame_path(self.indices[index])

    @property
    def fps(self) -> Optional[float]:
        return self.source.fps


class VideoSource(FrameSource):
    """A video file (MP4, MOV, ...) read through imageio's FFmpeg plugin.

    Frames are decoded in order by a single reader; asking for increasing
    indices (which every engine function does) never seeks backwards.
    """

    def __init__(self, path):
        import imageio.v2 as imageio

        self.path = str(path)
        self.name = Path(self.path).stem
        self._reader = imageio.get_reader(self.path, format="ffmpeg")
        self._meta = self._reader.get_meta_data()
        self._count = self._reader.count_frames()
        self._lock = threading.Lock()
        logger.info("%d frames found in %s.", self._count, self.name)

    def __len__(self) -> int:
        return self._count

    @property
    def fps(self) -> Optional[float]:
        return self._meta.get("fps")

    def read_raw(self, indices: Sequence[int]) -> List[np.ndarray]:
        with self._lock:
            return [files.as_hwc(np.asarray(self._reader.get_data(i))) for i in indices]

    def close(self) -> None:
        self._reader.close()
