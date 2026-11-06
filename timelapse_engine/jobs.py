"""Jobs: a complete, saveable description of one render, independent of any GUI.

A :class:`Job` holds everything needed to run a still stack, a rolling-window
video or an image sequence. Jobs round-trip through JSON, so a list of them is
a reusable preset / batch queue (what the notebook's config cells used to be)::

    jobs = [Job(kind="still", source=r"E:\\TL\\shoot", process="maximum", export_dir=r"E:\\TL")]
    save_queue("night.json", jobs)
    for job in load_queue("night.json"):
        run_job(job)
"""

from __future__ import annotations

import inspect
import json
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import backend, engine, files
from .processors import PROCESSORS, create_processor
from .progress import CancelToken, ProgressCallback
from .sources import FrameSource, ImageSequence, SubsetSource, VideoSource

KINDS = {"still": "Single image", "video": "Video", "frames": "Image sequence"}
MASTERS = {"prores": "ProRes 422 HQ (CPU)", "hevc_nvenc": "HEVC 10-bit (NVIDIA GPU)",
           "hevc_lossless": "HEVC lossless (slow)", None: "None"}
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".mxf")


# ---------------------------------------------------------------------------
# Processor parameters (for building forms)
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    name: str
    label: str
    default: object
    kind: type
    minimum: float = 0
    maximum: float = 1
    step: float = 1
    help: str = ""


# label, min, max, step, help
_PARAM_HINTS = {
    "hist_bins": ("Histogram bins", 4, 256, 4, "More bins show finer changes but use more GPU memory."),
    "noise_threshold": ("Noise threshold (0-255)", 0, 64, 1, "Changes smaller than this are treated as sensor noise."),
    "motion_threshold": ("Motion threshold (0-255)", 1, 255, 1,
                         "How far a pixel must differ from the background to count as moving."),
    "intensity_priority": ("Strongest motion wins", 0, 1, 1,
                           "On: each pixel keeps the frame where it moved most. Off: the newest moving frame wins."),
    "close_iterations": ("Fill holes", 0, 10, 1, "Smooths ragged motion masks (morphological closing)."),
    "hue_period": ("Hue cycle (frames)", 2, 100000, 10, "Frames per full trip around the colour wheel."),
    "target_mean": ("Auto-brightness target", 0.0, 1.0, 0.05, "Dark results are brightened to this average level."),
    "pyr_scale": ("Pyramid scale", 0.1, 0.9, 0.05, "Optical flow image pyramid scale."),
    "levels": ("Pyramid levels", 1, 12, 1, "Optical flow pyramid levels."),
    "winsize": ("Window size", 3, 64, 1, "Optical flow averaging window."),
    "iterations": ("Iterations", 1, 20, 1, "Optical flow iterations per level."),
    "poly_n": ("Poly neighbourhood", 5, 9, 2, "Optical flow polynomial expansion size (5 or 7)."),
    "poly_sigma": ("Poly sigma", 0.5, 2.0, 0.1, "Optical flow polynomial smoothing."),
}
_HIDDEN_PARAMS = {"use_gpu"}


def processor_params(name: str) -> List[ParamSpec]:
    """Editable constructor parameters of a processor, with labels and sensible ranges."""
    specs = []
    for p in inspect.signature(PROCESSORS[name].__init__).parameters.values():
        if p.name in ("self",) or p.name in _HIDDEN_PARAMS or p.default is inspect.Parameter.empty:
            continue
        kind = type(p.default) if p.default is not None else float
        label, lo, hi, step, help_text = _PARAM_HINTS.get(p.name, (p.name.replace("_", " ").capitalize(),
                                                                  0, 1000, 1, ""))
        specs.append(ParamSpec(p.name, label, p.default, kind, lo, hi, step, help_text))
    return specs


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@dataclass
class Job:
    kind: str = "still"                 # "still" | "video" | "frames"
    source: str = ""                    # image folder or video file
    process: str = "average"
    params: Dict[str, object] = field(default_factory=dict)
    export_dir: str = ""
    export_name: str = ""               # default: derived from the source
    sort: str = "exif"                  # image folders: "exif" or "name"
    first_frame: int = 0                # frame range, 0-based, end exclusive (None = to the end)
    end_frame: Optional[int] = None
    ext: str = "tiff"                   # stills / image sequences: "tiff" or "jpg"
    blend: int = 8                      # video / image sequence
    step: int = 1
    fps: float = 24.0
    master: Optional[str] = "prores"
    preview_1080p: bool = False
    preview_4k: bool = False
    device: str = "auto"
    overwrite: str = "increment"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def validate(self) -> List[str]:
        """Problems that would stop the job from running (empty list = OK)."""
        errors = []
        if self.kind not in KINDS:
            errors.append(f"Unknown job type {self.kind!r}.")
        src = Path(self.source) if self.source else None
        if src is None or not src.exists():
            errors.append("Choose an image folder or video file to process.")
        if not self.export_dir:
            errors.append("Choose an output folder.")
        if self.process not in PROCESSORS:
            errors.append(f"Unknown effect {self.process!r}.")
        else:
            try:
                create_processor(self.process, **self.params)
            except Exception as exc:
                errors.append(str(exc))
        if self.kind in ("video", "frames") and (self.blend < 1 or self.step < 1):
            errors.append("Blend and step must be at least 1.")
        if self.kind == "video" and not (self.master or self.preview_1080p or self.preview_4k):
            errors.append("Pick at least one video output (master or preview).")
        if self.end_frame is not None and self.end_frame <= self.first_frame:
            errors.append("The frame range is empty.")
        return errors

    def default_name(self) -> str:
        return self.export_name or default_export(self.source)[1]

    def describe(self) -> str:
        name = self.default_name() or "?"
        rng = "" if (self.first_frame == 0 and self.end_frame is None) else \
            f" [{self.first_frame + 1}-{self.end_frame or 'end'}]"
        if self.kind == "still":
            what = f"{self.process} -> {self.ext.upper()}"
        elif self.kind == "video":
            outs = [o for o, on in ((self.master, bool(self.master)), ("1080p", self.preview_1080p),
                                    ("4K", self.preview_4k)) if on]
            what = f"{self.process} blend {self.blend} step {self.step} -> {' + '.join(outs)}"
        else:
            what = f"{self.process} blend {self.blend} step {self.step} -> {self.ext.upper()} frames"
        return f"{KINDS.get(self.kind, self.kind)}: {name}{rng}, {what}"


def default_export(source: str) -> Tuple[str, str]:
    """Sensible output folder and name for a source.

    A Lightroom-style ``...\\Shoot\\export`` folder renders to ``...\\Shoot`` named
    ``Shoot``; a video renders next to itself, named after the file.
    """
    if not source:
        return "", ""
    p = Path(source)
    return str(p.parent), files.shoot_name(p)


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


class SourceCache:
    """Keeps sources open between jobs and previews so a folder is only scanned,
    EXIF-sorted and validated once."""

    def __init__(self):
        self._sources: Dict[Tuple[str, str], FrameSource] = {}
        self._lock = threading.Lock()  # GUI previews and the job queue share one cache

    def open(self, path: str, sort: str = "exif") -> FrameSource:
        key = (str(Path(path).resolve()), "video" if is_video(path) else sort)
        with self._lock:
            if key not in self._sources:
                self._sources[key] = VideoSource(path) if is_video(path) else ImageSequence(path, sort=sort)
            return self._sources[key]

    def forget(self, path: str) -> None:
        """Drop a cached source so the next open re-scans the folder."""
        resolved = str(Path(path).resolve())
        with self._lock:
            for key in [k for k in self._sources if k[0] == resolved]:
                self._sources.pop(key).close()

    def close(self) -> None:
        with self._lock:
            for src in self._sources.values():
                src.close()
            self._sources.clear()


def job_source(job: Job, sources: SourceCache) -> FrameSource:
    src = sources.open(job.source, job.sort)
    end = len(src) if job.end_frame is None else min(job.end_frame, len(src))
    if job.first_frame == 0 and end == len(src):
        return src
    return SubsetSource(src, range(job.first_frame, end))


def run_job(job: Job, sources: Optional[SourceCache] = None, progress: Optional[ProgressCallback] = None,
            cancel: Optional[CancelToken] = None) -> List[Path]:
    """Run one job and return the files it wrote."""
    errors = job.validate()
    if errors:
        raise ValueError(" ".join(errors))
    own_cache = sources is None
    sources = sources or SourceCache()
    try:
        src = job_source(job, sources)
        name = job.default_name()
        common = dict(progress=progress, cancel=cancel, device=job.device, **job.params)
        if job.kind == "still":
            return [engine.stack_to_file(src, job.process, job.export_dir, name, ext=job.ext,
                                         overwrite=job.overwrite, **common)]
        if job.kind == "video":
            return engine.render_video(src, job.process, job.export_dir, name, blend=job.blend, step=job.step,
                                       fps=job.fps, master=job.master, preview_1080p=job.preview_1080p,
                                       preview_4k=job.preview_4k, overwrite=job.overwrite, **common)
        return engine.render_frames(src, job.process, Path(job.export_dir) / f"{name}_{job.process}_frames", name,
                                    blend=job.blend, step=job.step, ext=job.ext, overwrite=job.overwrite, **common)
    finally:
        if own_cache:
            sources.close()


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

def preview_indices(job: Job, n_frames: int, at_frame: int, still_samples: int = 24) -> List[int]:
    """Frames used for a quick preview.

    Video / image sequence: the rolling window that starts at ``at_frame``.
    Single image: ``still_samples`` frames spread evenly over the frame range.
    """
    first = job.first_frame
    end = n_frames if job.end_frame is None else min(job.end_frame, n_frames)
    if job.kind == "still":
        count = min(still_samples, end - first)
        return sorted(set(np.linspace(first, end - 1, count).round().astype(int).tolist()))
    start = min(max(at_frame, first), end - 1)
    return list(range(start, min(start + job.blend, end)))


def render_preview(job: Job, sources: SourceCache, at_frame: int, max_side: int = 1280,
                   still_samples: int = 24, cancel: Optional[CancelToken] = None) -> np.ndarray:
    """Run the job's effect on a few downscaled frames; returns float 0-1 (H, W, C)."""
    src = sources.open(job.source, job.sort)
    indices = preview_indices(job, len(src), at_frame, still_samples)
    proxy = SubsetSource(src, indices, max_side=max_side)
    processor = create_processor(job.process, **job.params)
    if len(indices) == 1 and job.kind != "still":
        return backend.to_float(proxy.read_raw([0])[0], backend.scale_for(proxy.probe()[1]))
    return engine.stack(proxy, processor, cancel=cancel, device=job.device)


# ---------------------------------------------------------------------------
# Queue files
# ---------------------------------------------------------------------------

def save_queue(path, jobs: List[Job]) -> None:
    Path(path).write_text(json.dumps({"version": 1, "jobs": [j.to_dict() for j in jobs]}, indent=2),
                          encoding="utf-8")


def load_queue(path) -> List[Job]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Job.from_dict(d) for d in (data["jobs"] if isinstance(data, dict) else data)]
