"""Output sinks for windowed renders: numbered still frames and FFmpeg video.

Sinks receive uint16 RGB frames (H, W, 3) - or float 0-1 - as NumPy or CuPy
arrays, and write them on background threads so encoding overlaps with processing::

    with FFmpegVideoSink(...) as sink:
        sink.write(frame)
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from . import colour, files
from .backend import asnumpy
from .progress import logger


class Sink:
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def outputs(self) -> List[Path]:
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ImageSequenceSink(Sink):
    """Writes ``<name>_0001.tiff``, ``<name>_0002.tiff``, ... using several encoder threads."""

    def __init__(self, export_dir, export_name: str, ext: str = "tiff", overwrite: str = "increment",
                 workers: int = 4):
        self.export_dir = Path(export_dir)
        self.export_name = export_name
        self.ext = ext.lstrip(".").lower()
        self.overwrite = overwrite
        self.index = 0
        self._written: List[Path] = []
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="save")
        self._pending: Deque[Future] = deque()
        self._max_pending = workers * 2

    def write(self, frame):
        self.index += 1
        path = files.resolve_output_path(
            self.export_dir / f"{self.export_name}_{self.index:04d}.{self.ext}", self.overwrite)
        self._written.append(path)
        self._pending.append(self._pool.submit(files.save_image, path, asnumpy(frame)))
        while len(self._pending) >= self._max_pending:  # bound memory; surface errors early
            self._pending.popleft().result()

    def close(self):
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._pool.shutdown(wait=True)

    @property
    def outputs(self):
        return self._written


@dataclass
class VideoOutput:
    """One encoded video file.

    pix_fmt: the raw format the engine sends (it also picks the encoder's input format).
    size: optional (height, width) downscale target; either may be None to keep aspect.
    """

    path: Path
    codec_args: Sequence[str]
    pix_fmt: str
    size: Optional[Tuple[Optional[int], Optional[int]]] = None


# Encoder presets. Frames are converted to each preset's pix_fmt on the GPU.
def prores_master(path) -> VideoOutput:
    """ProRes 422 HQ (CPU encoder; ~8-10 fps at 20 MP)."""
    return VideoOutput(Path(path), ["-c:v", "prores_ks", "-profile:v", "3"], "yuv422p10le")


def hevc_lossless_master(path) -> VideoOutput:
    return VideoOutput(Path(path), ["-c:v", "libx265", "-x265-params", "lossless=1", "-preset", "fast"],
                       "yuv420p10le")


def hevc_nvenc_master(path, cq: int = 16, preset: str = "p5") -> VideoOutput:
    """10-bit HEVC on the NVIDIA GPU encoder: several times faster than ProRes at 20 MP and
    leaves the CPU free for decoding. cq 16 is visually lossless; lower = higher quality."""
    return VideoOutput(Path(path), ["-c:v", "hevc_nvenc", "-preset", preset, "-rc", "vbr", "-cq", str(cq),
                                    "-b:v", "0", "-tag:v", "hvc1"], "p010le")


def h264_preview(path, height: Optional[int] = None, width: Optional[int] = None, crf: int = 18,
                 preset: str = "veryfast") -> VideoOutput:
    """8-bit H.264 preview, downscaled on the GPU to ``height`` or ``width`` (aspect kept)."""
    size = (height, width) if (height or width) else None
    return VideoOutput(Path(path), ["-c:v", "libx264", "-crf", str(crf), "-preset", preset], "yuv420p", size)


_STOP = object()
_PIPE_CHUNK = 4 << 20  # Windows pipes are ~5x slower when one write() is a whole 100+ MB frame
_BT709 = ["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


class _Encoder:
    """One FFmpeg process fed by its own writer thread."""

    def __init__(self, ffmpeg: str, output: VideoOutput, fps: float, queue_size: int):
        self.ffmpeg, self.output, self.fps = ffmpeg, output, fps
        self.queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def command(self, width: int, height: int) -> List[str]:
        o = self.output
        return [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", o.pix_fmt, "-s", f"{width}x{height}", "-r", str(self.fps),
                *_BT709, "-i", "-",
                *o.codec_args, "-pix_fmt", o.pix_fmt, *_BT709, "-r", str(self.fps), str(o.path)]

    def start(self, width: int, height: int):
        self.output.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command(width, height)
        logger.debug("FFmpeg: %s", " ".join(cmd))
        # Above-normal priority: encoders (ProRes especially) get the CPU first and
        # frame decoding uses what's left, instead of the two fighting for cores.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0,
                                     creationflags=getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0))
        self.thread = threading.Thread(target=self._run, name=f"ffmpeg-{self.output.path.name}", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                data = self.queue.get()
                if data is _STOP:
                    return
                view = memoryview(data).cast("B")
                for start in range(0, len(view), _PIPE_CHUNK):
                    self.proc.stdin.write(view[start:start + _PIPE_CHUNK])
        except BaseException as exc:  # e.g. BrokenPipeError when FFmpeg dies
            self.error = exc
            while self.queue.get() is not _STOP:  # drain so the producer never blocks
                pass

    def put(self, data: np.ndarray):
        if self.error is not None:
            raise RuntimeError(f"FFmpeg writer for {self.output.path.name} failed: {self.error}") from self.error
        self.queue.put(data)

    def close(self) -> Optional[str]:
        if self.proc is None:
            return None
        self.queue.put(_STOP)
        self.thread.join()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait()
        self.proc = None
        if code != 0 or self.error is not None:
            return f"{self.output.path.name}: exit code {code} {self.error or ''}"
        return None


class FFmpegVideoSink(Sink):
    """Encodes frames to one or more video files.

    Each output gets its frames already converted (BT.709 Y'CbCr, right bit depth
    and chroma layout) and downscaled - on the GPU when the frame is a CuPy array -
    so FFmpeg only has to encode. Every output has its own FFmpeg process and
    writer thread, so encoders run in parallel with each other and with the GPU.
    """

    def __init__(self, outputs: Sequence[VideoOutput], fps: float = 24, ffmpeg_path: str = "ffmpeg",
                 overwrite: str = "increment", queue_size: int = 3):
        if not outputs:
            raise ValueError("FFmpegVideoSink needs at least one VideoOutput.")
        exe = shutil.which(ffmpeg_path) or (ffmpeg_path if Path(ffmpeg_path).is_file() else None)
        if exe is None:
            raise FileNotFoundError(f"FFmpeg not found: {ffmpeg_path}")
        self.video_outputs = [
            VideoOutput(files.resolve_output_path(o.path, overwrite), o.codec_args, o.pix_fmt, o.size)
            for o in outputs
        ]
        self._encoders = [_Encoder(exe, o, fps, queue_size) for o in self.video_outputs]

    def write(self, frame):
        for enc in self._encoders:
            data, (h, w) = colour.pack(frame, enc.output.pix_fmt, enc.output.size)
            if enc.proc is None:
                enc.start(w, h)
            enc.put(asnumpy(data))

    def close(self):
        errors = [e for e in (enc.close() for enc in self._encoders) if e]
        if errors:
            raise RuntimeError("FFmpeg failed: " + "; ".join(errors))

    @property
    def outputs(self):
        return [o.path for o in self.video_outputs]
