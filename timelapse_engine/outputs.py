"""Output sinks for windowed renders: numbered still frames and FFmpeg video.

A sink is used as a context manager and receives float32 0-1 frames:

    with FFmpegVideoSink(...) as sink:
        sink.write(frame)
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import files
from .progress import logger


class Sink:
    def open(self, frame_shape) -> None:
        pass

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
    """Writes ``<name>_0001.tiff``, ``<name>_0002.tiff``, ..."""

    def __init__(self, export_dir, export_name: str, ext: str = "tiff", overwrite: str = "increment"):
        self.export_dir = Path(export_dir)
        self.export_name = export_name
        self.ext = ext.lstrip(".").lower()
        self.overwrite = overwrite
        self.index = 0
        self._written: List[Path] = []

    def write(self, frame):
        self.index += 1
        path = files.resolve_output_path(
            self.export_dir / f"{self.export_name}_{self.index:04d}.{self.ext}", self.overwrite)
        self._written.append(files.save_image(path, frame))

    @property
    def outputs(self):
        return self._written


@dataclass
class VideoOutput:
    """One encoded file produced from the shared frame stream."""

    path: Path
    codec_args: Sequence[str]
    scale: Optional[str] = None  # FFmpeg scale expression, e.g. "-2:1080"


# Encoder presets. Even dimensions are enforced for every preset.
def prores_master(path) -> VideoOutput:
    return VideoOutput(Path(path), ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"])


def hevc_lossless_master(path) -> VideoOutput:
    return VideoOutput(Path(path), ["-c:v", "libx265", "-x265-params", "lossless=1", "-preset", "fast",
                                    "-pix_fmt", "yuv420p10le"])


def h264_preview(path, scale: Optional[str] = None, crf: int = 18, preset: str = "ultrafast") -> VideoOutput:
    return VideoOutput(Path(path), ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                                    "-pix_fmt", "yuv420p"], scale)


class FFmpegVideoSink(Sink):
    """Pipes 16-bit RGB frames into ONE FFmpeg process that writes every requested output.

    The frame is sent through the pipe once and FFmpeg's ``split`` filter feeds
    each encoder, instead of piping the same full-resolution frame to several
    processes.
    """

    def __init__(self, outputs: Sequence[VideoOutput], fps: float = 24, ffmpeg_path: str = "ffmpeg",
                 overwrite: str = "increment"):
        if not outputs:
            raise ValueError("FFmpegVideoSink needs at least one VideoOutput.")
        exe = shutil.which(ffmpeg_path) or (ffmpeg_path if Path(ffmpeg_path).is_file() else None)
        if exe is None:
            raise FileNotFoundError(f"FFmpeg not found: {ffmpeg_path}")
        self.ffmpeg = exe
        self.fps = fps
        self.video_outputs = [
            VideoOutput(files.resolve_output_path(o.path, overwrite), o.codec_args, o.scale) for o in outputs
        ]
        self._proc: Optional[subprocess.Popen] = None
        self._shape = None

    def _command(self, width: int, height: int) -> List[str]:
        n = len(self.video_outputs)
        labels = [f"[s{i}]" for i in range(n)]
        chains = [f"[0:v]split={n}{''.join(labels)}"]
        for i, out in enumerate(self.video_outputs):
            scale = f"scale={out.scale}:flags=bicubic" if out.scale else "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0"
            chains.append(f"{labels[i]}{scale}[o{i}]")

        cmd = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{width}x{height}",
               "-r", str(self.fps), "-i", "-",
               "-filter_complex", ";".join(chains)]
        for i, out in enumerate(self.video_outputs):
            cmd += ["-map", f"[o{i}]", *out.codec_args, "-r", str(self.fps), str(out.path)]
        return cmd

    def open(self, frame_shape):
        height, width = frame_shape[:2]
        for out in self.video_outputs:
            out.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._command(width, height)
        logger.debug("FFmpeg: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._shape = frame_shape

    def write(self, frame):
        if self._proc is None:
            self.open(frame.shape)
        if frame.ndim == 2 or frame.shape[2] == 1:
            frame = np.repeat(frame.reshape(frame.shape[0], frame.shape[1], 1), 3, axis=2)
        try:
            self._proc.stdin.write(files.to_uint16(frame).tobytes())
        except BrokenPipeError:
            self._proc.wait()
            raise RuntimeError(f"FFmpeg exited early with code {self._proc.returncode}") from None

    def close(self):
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self._proc.wait()
        self._proc = None
        if code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {code}")

    @property
    def outputs(self):
        return [o.path for o in self.video_outputs]
