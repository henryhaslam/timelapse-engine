"""Output sinks for windowed renders: numbered still frames and FFmpeg video.

Sinks receive uint16 RGB frames (H, W, 3) - or float 0-1 - as NumPy or CuPy
arrays, and write them on background threads so encoding overlaps with processing::

    with FFmpegVideoSink(...) as sink:
        sink.write(frame)
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from . import colour, files
from .backend import asnumpy
from .progress import logger


class Sink:
    wants_metadata = False  # set True to receive per-frame OutputMetadata in write()

    def write(self, frame: np.ndarray, meta=None) -> None:
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
    """Writes ``<name>_0001.tiff``, ``<name>_0002.tiff``, ... using several encoder threads.
    Each frame gets its own metadata (which frames were blended, their capture times...)."""

    wants_metadata = True

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

    def write(self, frame, meta=None):
        self.index += 1
        path = files.resolve_output_path(
            self.export_dir / f"{self.export_name}_{self.index:04d}.{self.ext}", self.overwrite)
        self._written.append(path)
        self._pending.append(self._pool.submit(files.save_image, path, asnumpy(frame), metadata=meta))
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


# Encoder presets. Frames are converted to each preset's pix_fmt (and size) on the GPU.
Size = Optional[Tuple[Optional[int], Optional[int]]]


def prores_master(path, size: Size = None) -> VideoOutput:
    """ProRes 422 HQ (CPU encoder; ~8-10 fps at 20 MP)."""
    return VideoOutput(Path(path), ["-c:v", "prores_ks", "-profile:v", "3"], "yuv422p10le", size)


def hevc_lossless_master(path, size: Size = None) -> VideoOutput:
    return VideoOutput(Path(path), ["-c:v", "libx265", "-x265-params", "lossless=1", "-preset", "fast"],
                       "yuv420p10le", size)


def hevc_nvenc_master(path, cq: int = 16, preset: str = "p5", size: Size = None) -> VideoOutput:
    """10-bit HEVC on the NVIDIA GPU encoder: several times faster than ProRes at 20 MP and
    leaves the CPU free for decoding. cq 16 is visually lossless; lower = higher quality."""
    return VideoOutput(Path(path), ["-c:v", "hevc_nvenc", "-preset", preset, "-rc", "vbr", "-cq", str(cq),
                                    "-b:v", "0", "-tag:v", "hvc1"], "p010le", size)


def h264_preview(path, height: Optional[int] = None, width: Optional[int] = None, crf: int = 18,
                 preset: str = "veryfast") -> VideoOutput:
    """8-bit H.264, downscaled on the GPU to ``height`` or ``width`` (aspect kept; neither = full size)."""
    size = (height, width) if (height or width) else None
    return VideoOutput(Path(path), ["-c:v", "libx264", "-crf", str(crf), "-preset", preset], "yuv420p", size)


# ---------------------------------------------------------------------------
# Video files by size and format (what the app's tick boxes choose)
# ---------------------------------------------------------------------------

VIDEO_SIZES = {          # key: label. Sizes: see video_target (never larger than the frames)
    "full": "Full resolution",
    "4k": "4K",          # long side 3840 (3840x2882 for the G9 II's 5776x4336 frames)
    "1080p": "1080p",    # short side 1080 (1440x1080 for a 4:3 landscape frame)
}
VIDEO_FORMATS = {        # key: (label, file extension, explanation)
    "prores": ("ProRes HQ .mov", ".mov",
               "Editing master: 10-bit 4:2:2, large files. Encoded on the CPU, so the slowest at full resolution."),
    "hevc_nvenc": ("HEVC .mov (GPU)", ".mov",
                   "Encoded on the NVIDIA GPU: several times faster than ProRes and much smaller, visually "
                   "lossless. Needs a recent NVIDIA driver."),
    "h264": ("H.264 .mp4", ".mp4",
             "Plays anywhere (phones, browsers, Windows). 8-bit and small: good for 1080p previews and sharing."),
    "hevc_lossless": ("HEVC lossless .mov", ".mov", "Archive copy with no loss at all. Very slow and very large."),
}


def video_target(size: str, frame_shape: Optional[Tuple[int, ...]] = None) -> Size:
    """(height, width) target for a size key; one side None keeps the aspect. Portrait frames
    get the same treatment turned sideways (1080p = 1080 wide)."""
    if size not in VIDEO_SIZES:
        raise ValueError(f"Unknown video size {size!r}: use one of {', '.join(VIDEO_SIZES)}")
    portrait = frame_shape is not None and frame_shape[0] > frame_shape[1]
    if size == "full":
        return None
    if size == "4k":
        return (3840, None) if portrait else (None, 3840)
    return (None, 1080) if portrait else (1080, None)


def video_file_name(name: str, size: str, fmt: str) -> str:
    """``<name>_MASTER.mov`` for a full-size .mov, otherwise ``<name>_full|4K|1080p.<ext>``
    (the names the app has always used for the master and the previews)."""
    ext = VIDEO_FORMATS[fmt][1]
    label = {"full": "MASTER" if ext == ".mov" else "full", "4k": "4K", "1080p": "1080p"}[size]
    return f"{name}_{label}{ext}"


def video_output(path, size: str, fmt: str, frame_shape: Optional[Tuple[int, ...]] = None) -> VideoOutput:
    """One video file: ``fmt`` (see VIDEO_FORMATS) at ``size`` (see VIDEO_SIZES)."""
    target = video_target(size, frame_shape)
    if fmt == "prores":
        return prores_master(path, target)
    if fmt == "hevc_nvenc":
        return hevc_nvenc_master(path, size=target)
    if fmt == "hevc_lossless":
        return hevc_lossless_master(path, target)
    if fmt == "h264":
        out = h264_preview(path)
        out.size = target
        return out
    raise ValueError(f"Unknown video format {fmt!r}: use one of {', '.join(VIDEO_FORMATS)}")


def parse_videos(videos) -> List[Tuple[str, str]]:
    """[(size, format)] from a list of (size, format) pairs or {"size": ..., "format": ...} dicts."""
    out = []
    for v in videos or []:
        size, fmt = (v.get("size"), v.get("format")) if isinstance(v, dict) else tuple(v)
        if size not in VIDEO_SIZES or fmt not in VIDEO_FORMATS:
            raise ValueError(f"Unknown video output {size!r}/{fmt!r}: sizes {', '.join(VIDEO_SIZES)}, "
                             f"formats {', '.join(VIDEO_FORMATS)}")
        out.append((size, fmt))
    return out


def legacy_videos(master: Optional[str], preview_1080p: bool = False, preview_4k: bool = False) -> List[Tuple[str, str]]:
    """The old master / preview options as (size, format) pairs."""
    videos = [("full", master)] if master else []
    if preview_4k:
        videos.append(("4k", "h264"))
    if preview_1080p:
        videos.append(("1080p", "h264"))
    return videos


_STOP = object()
_PIPE_CHUNK = 4 << 20  # Windows pipes are ~5x slower when one write() is a whole 100+ MB frame
_BT709 = ["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
# FFmpeg is a console program: from the desktop app (no console) it would open a window of its own
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PRIORITY = getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0)
# FFmpeg lines that only repeat that something above went wrong
_NOISE = ("Task finished", "Terminating thread", "Error sending frames", "Could not open encoder before EOF",
          "Nothing was written", "Conversion failed")
_FRAME_BYTES = {"p010le": 3.0, "yuv420p10le": 3.0, "yuv422p10le": 4.0, "yuv420p": 1.5}   # per pixel


def ffmpeg_message(lines: Sequence[str], code: Optional[int] = None, codec_args: Sequence[str] = ()) -> str:
    """FFmpeg's own explanation from its error output, without the ``[hevc_nvenc @ 0x...]``
    prefixes and the generic lines that follow every failure."""
    useful: List[str] = []
    for line in lines:
        text = re.sub(r"^(\[[^\]]*\]\s*)+", "", line.strip())
        if text and text not in useful and not any(text.startswith(n) for n in _NOISE):
            useful.append(text)
    if len(useful) > 1:   # "maybe incorrect parameters such as bit_rate..." is a guess once the cause is known
        useful = [u for u in useful if not u.startswith("Error while opening encoder")] or useful
    message = " ".join(u if u.endswith((".", ":")) else u + "." for u in useful[:3])
    if not message:
        message = f"FFmpeg exited with code {code}." if code is not None else "FFmpeg failed."
    if any("nvenc" in str(a) for a in codec_args):
        message += (" The GPU encoder (NVENC) can't start: update the NVIDIA driver, or choose the ProRes "
                    "master for now.")
    return message


@lru_cache(maxsize=32)
def _encoder_problem(ffmpeg: str, codec_args: Tuple[str, ...], pix_fmt: str) -> Optional[str]:
    """Encode two tiny frames the way a render does (raw frames on stdin); None if that works,
    else FFmpeg's explanation. Cached, so each encoder is checked once per session."""
    per_pixel = _FRAME_BYTES.get(pix_fmt)
    if per_pixel is None:
        return None                      # unknown layout: let the render itself report problems
    w = h = 256
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt,
           "-s", f"{w}x{h}", "-r", "25", "-i", "-", *codec_args, "-pix_fmt", pix_fmt, "-f", "null", "-"]
    blank = bytes(int(w * h * per_pixel) * 2)
    try:
        proc = subprocess.run(cmd, input=blank, capture_output=True, timeout=60, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"FFmpeg couldn't be run: {exc}"
    if proc.returncode == 0:
        return None
    return ffmpeg_message(proc.stderr.decode("utf-8", "replace").splitlines(), proc.returncode, codec_args)


class _Encoder:
    """One FFmpeg process fed by its own writer thread."""

    def __init__(self, ffmpeg: str, output: VideoOutput, fps: float, queue_size: int, metadata_args=()):
        self.ffmpeg, self.output, self.fps = ffmpeg, output, fps
        self.metadata_args = list(metadata_args)
        self.queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self.stderr: Deque[str] = deque(maxlen=40)      # FFmpeg's last error lines
        self._stderr_thread: Optional[threading.Thread] = None

    def command(self, width: int, height: int) -> List[str]:
        o = self.output
        return [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", o.pix_fmt, "-s", f"{width}x{height}", "-r", str(self.fps),
                *_BT709, "-i", "-",
                *o.codec_args, "-pix_fmt", o.pix_fmt, *_BT709, *self.metadata_args,
                "-r", str(self.fps), str(o.path)]

    def start(self, width: int, height: int):
        self.output.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command(width, height)
        logger.debug("FFmpeg: %s", " ".join(cmd))
        # Above-normal priority: encoders (ProRes especially) get the CPU first and
        # frame decoding uses what's left, instead of the two fighting for cores.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                     creationflags=_PRIORITY | _NO_WINDOW)
        self.thread = threading.Thread(target=self._run, name=f"ffmpeg-{self.output.path.name}", daemon=True)
        self.thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, name=f"ffmpeg-err-{self.output.path.name}",
                                               daemon=True)
        self._stderr_thread.start()

    def _read_stderr(self):
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.stderr.append(line)
                logger.debug("FFmpeg %s: %s", self.output.path.name, line)

    def failure(self, code: Optional[int] = None) -> str:
        """Why this encoder failed, in FFmpeg's words when it said anything."""
        if self.proc is not None and code is None:
            try:
                code = self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        if self.stderr:
            return ffmpeg_message(list(self.stderr), code, self.output.codec_args)
        return f"exit code {code}" + (f" ({self.error})" if self.error else "")

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
            raise RuntimeError(f"FFmpeg failed: {self.output.path.name}: {self.failure()}") from self.error
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
        failed = code != 0 or self.error is not None
        message = f"{self.output.path.name}: {self.failure(code)}" if failed else None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        self.proc = None
        return message


class FFmpegVideoSink(Sink):
    """Encodes frames to one or more video files.

    Each output gets its frames already converted (BT.709 Y'CbCr, right bit depth
    and chroma layout) and downscaled - on the GPU when the frame is a CuPy array -
    so FFmpeg only has to encode. Every output has its own FFmpeg process and
    writer thread, so encoders run in parallel with each other and with the GPU.
    """

    def __init__(self, outputs: Sequence[VideoOutput], fps: float = 24, ffmpeg_path: str = "ffmpeg",
                 overwrite: str = "increment", queue_size: int = 3, metadata=None):
        if not outputs:
            raise ValueError("FFmpegVideoSink needs at least one VideoOutput.")
        exe = shutil.which(ffmpeg_path) or (ffmpeg_path if Path(ffmpeg_path).is_file() else None)
        if exe is None:
            raise FileNotFoundError(f"FFmpeg not found: {ffmpeg_path}")
        # Try each encoder on two tiny frames first: a job that can't encode stops before any frame is
        # read, and no empty or half-written video files are left behind.
        for o in outputs:
            problem = _encoder_problem(exe, tuple(o.codec_args), o.pix_fmt)
            if problem:
                raise RuntimeError(f"Can't encode {Path(o.path).name}: {problem}")
        self.video_outputs = [
            VideoOutput(files.resolve_output_path(o.path, overwrite), o.codec_args, o.pix_fmt, o.size)
            for o in outputs
        ]
        from .metadata import ffmpeg_args

        self._encoders = [_Encoder(exe, o, fps, queue_size, ffmpeg_args(metadata)) for o in self.video_outputs]

    def write(self, frame, meta=None):
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
