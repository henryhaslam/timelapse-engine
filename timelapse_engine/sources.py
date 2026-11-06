"""Frame sources: a uniform way to read frames from an image folder or a video file.

Every source returns float32 0-1 frames with shape (H, W, C), so processors
never need to know where the pixels came from.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import files
from .progress import logger


class FrameSource(ABC):
    """Base class. Subclasses implement ``__len__`` and ``read_many``."""

    name: str = "source"

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def read_many(self, indices: Sequence[int]) -> List[np.ndarray]:
        """Read several frames (float32 0-1, H x W x C)."""

    def read(self, index: int) -> np.ndarray:
        return self.read_many([index])[0]

    def capture_time(self, index: int) -> Optional[datetime]:
        return None

    def latest_capture_time(self) -> Optional[datetime]:
        return None

    def xmp(self, index: int) -> Optional[bytes]:
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
                 validate: bool = True, workers: int = 8, exiftool_path: Optional[str] = None):
        if (folder is None) == (paths is None):
            raise ValueError("Pass exactly one of 'folder' or 'paths'.")
        if paths is None:
            paths = files.list_images(folder)
            self.name = Path(folder).name
        else:
            paths = [str(p) for p in paths]
            self.name = Path(paths[0]).parent.name if paths else "sequence"
        if not paths:
            raise FileNotFoundError(f"No JPEG/TIFF images found in {folder}")

        self.workers = workers
        self._pool: Optional[ThreadPoolExecutor] = None

        if sort == "exif":
            self._times = files.read_capture_times(paths, exiftool_path, workers)
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

    def read_many(self, indices: Sequence[int]) -> List[np.ndarray]:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
        return list(self._pool.map(files.load_image, [self.paths[i] for i in indices]))

    def _ensure_times(self):
        if self._times is None:
            self._times = files.read_capture_times(self.paths, workers=self.workers)
        return self._times

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

    def read_many(self, indices: Sequence[int]) -> List[np.ndarray]:
        with self._lock:
            return [files.to_float(self._reader.get_data(i)) for i in indices]

    def close(self) -> None:
        self._reader.close()
