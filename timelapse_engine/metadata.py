"""Metadata for rendered outputs: what was made, from which frames, with which camera.

Every still, image-sequence frame and video gets:

* **Title** - e.g. "Maximum stack - 270 frames"
* **Caption / description** - process, frame count and range, exposure time,
  interval, time span, total exposure, camera and lens, effect settings
* **Camera and lens EXIF** - make, model, lens, focal length, aperture, ISO and
  exposure copied from the source frames (with exiftool: all EXIF/IPTC/XMP,
  including Lightroom's already-applied develop settings)
* **ICC colour profile** - copied from the source (e.g. ProPhoto RGB), so the
  output's colours display the same as the input's
* **EXIF UserComment** - the same facts as JSON, for scripts
* **Capture time** - stills: last frame + 1 minute (sorts after the sequence);
  sequence frames: the last frame of each window

With exiftool available (it is found automatically) everything above is
written; without it, the essentials are written natively by Pillow/tifffile.
"""

from __future__ import annotations

import atexit
import json
import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import files
from .progress import logger

PROCESS_LABELS = {
    "average": "Average", "maximum": "Maximum", "minimum": "Minimum", "median": "Median", "mode": "Mode",
    "temporal_contrast": "Temporal contrast", "temporal_std": "Temporal variation",
    "temporal_gradient": "Temporal gradient", "temporal_hue_motion": "Motion coloured by time",
    "temporal_entropy": "Temporal entropy", "optical_flow_accum": "Optical flow", "motion_overlay": "Motion overlay",
    "change_entropy": "Change entropy", "colour_entropy": "Colour entropy", "temporal_shape": "Temporal shape",
    "time_of_peak": "Time of peak", "rhythm_map": "Rhythm map", "eigen_colour": "Eigen-colour",
    "memory_map": "Memory map", "time_slice": "Time slice", "temporal_denoise": "Temporal denoise",
    "comet_trails": "Comet trails", "clipped_average": "Clipped average",
}


def software_name() -> str:
    from . import __version__

    return f"timelapse-engine {__version__}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_exposure(seconds: float) -> str:
    if seconds >= 0.95:
        return f"{seconds:.3g} s"
    return f"1/{round(1 / seconds)} s"


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min {s} s" if s else f"{m} min"
    return f"{s} s"


def fmt_interval(seconds: float) -> str:
    return f"{seconds:.3g} s" if seconds < 60 else fmt_duration(seconds)


def _range_text(values: List[float], fmt) -> str:
    lo, hi = min(values), max(values)
    return fmt(lo) if abs(hi - lo) <= 1e-9 * max(abs(hi), 1) else f"{fmt(lo)} to {fmt(hi)}"


def _ascii(text: str) -> str:
    """EXIF ImageDescription and TIFF tags are ASCII; keep the summary readable in plain ASCII."""
    return text.encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------------------
# Summary of the frames that went into an output
# ---------------------------------------------------------------------------

@dataclass
class OutputMetadata:
    title: str
    description: str
    facts: dict                                  # machine-readable, written as JSON to EXIF UserComment
    capture_time: Optional[datetime] = None      # DateTimeOriginal for stills / sequence frames
    start_time: Optional[datetime] = None        # first frame (video creation time)
    reference: Optional[str] = None              # source frame to copy camera/lens/colour metadata from
    camera: files.FrameInfo = field(default_factory=files.FrameInfo)
    icc: Optional[bytes] = None
    xmp: Optional[bytes] = None

    @property
    def comment_json(self) -> str:
        return json.dumps(self.facts, separators=(",", ":"), default=str)


def _locate(source, index: int):
    """Follow SubsetSource wrappers back to the original sequence."""
    while hasattr(source, "indices") and hasattr(source, "source"):
        index = source.indices[index]
        source = source.source
    return source, index


@lru_cache(maxsize=64)
def _icc_and_xmp(path: str):
    return files.read_icc_profile(path), files.read_xmp(path)


def summarise(source, indices: Sequence[int], process: str, params: Optional[dict] = None, kind: str = "stack",
              blend: Optional[int] = None, step: Optional[int] = None, fps: Optional[float] = None,
              outputs: Optional[int] = None) -> OutputMetadata:
    """Describe an output made from ``source`` frames ``indices``.

    kind: "stack" (one still from all frames), "window" (one frame of an image
    sequence) or "video" (a rolling-window video).
    """
    params = dict(params or {})
    indices = list(indices)
    root, first_root = _locate(source, indices[0])
    root_indices = [_locate(source, i)[1] for i in indices]
    infos: List[files.FrameInfo] = [root.frame_info(i) or files.FrameInfo() for i in root_indices]
    paths = [root.frame_path(i) for i in root_indices]
    n, total = len(indices), len(root)
    label = PROCESS_LABELS.get(process, process.replace("_", " ").capitalize())
    source_name = root.name

    times = [fi.time for fi in infos if fi.time is not None]
    exposures = [fi.exposure for fi in infos if fi.exposure]
    isos = [fi.iso for fi in infos if fi.iso]
    interval = None
    if len(times) >= 2:
        diffs = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if b > a]
        interval = statistics.median(diffs) if diffs else None
    elif getattr(root, "fps", None):
        interval = 1.0 / root.fps

    # Reference frame for camera/lens/colour: the last frame (as the Lightroom XMP has always been taken from)
    ref_pos = max((k for k, p in enumerate(paths) if p), default=None)
    reference = paths[ref_pos] if ref_pos is not None else None
    camera = infos[ref_pos] if ref_pos is not None else files.FrameInfo()

    # --- human-readable caption -------------------------------------------
    if kind == "stack":
        title = f"{label} stack - {n} frames"
        parts = [f"{label} stack of {n} frames"]
    elif kind == "window":
        title = f"{label} blend - {n} frames"
        parts = [f"{label} blend of {n} frames"]
    else:
        title = f"{label} timelapse - blend {blend}, step {step}"
        parts = [f"{label} rolling blend of {blend} frames per output frame, step {step}"
                 + (f", {outputs} frames at {fps:g} fps" if outputs and fps else "")]
    first_no, last_no = root_indices[0] + 1, root_indices[-1] + 1
    parts.append(f"source frames {first_no}-{last_no} of {total} from {source_name}")

    shot = []
    if exposures:
        shot.append(f"{_range_text(exposures, fmt_exposure)} exposures")
    if interval:
        shot.append(f"every {fmt_interval(interval)}")
    if shot:
        parts.append(" ".join(shot))
    if times:
        start, end = times[0], times[-1]
        span = f"{start:%Y-%m-%d %H:%M:%S} to " + (f"{end:%H:%M:%S}" if end.date() == start.date()
                                                     else f"{end:%Y-%m-%d %H:%M:%S}")
        parts.append(f"{span} ({fmt_duration((end - start).total_seconds())})")
    if len(exposures) > 1:
        parts.append(f"total exposure {fmt_duration(sum(exposures)) if sum(exposures) >= 60 else fmt_exposure(sum(exposures))}")

    cam = " ".join(x for x in (camera.make, camera.model) if x)
    if camera.make and camera.model.lower().startswith(camera.make.split()[0].lower()):
        cam = camera.model  # "NIKON CORPORATION NIKON Z 6" -> "NIKON Z 6"
    optics = []
    if camera.focal_length:
        optics.append(f"{camera.focal_length:g} mm")
    if camera.fnumber:
        optics.append(f"f/{camera.fnumber:g}")
    if isos:
        optics.append("ISO " + _range_text(isos, lambda v: f"{v:g}"))
    gear = ", ".join(x for x in (cam, camera.lens, " ".join(optics)) if x)
    if gear:
        parts.append(gear)
    if params:
        parts.append("settings: " + ", ".join(f"{k}={v}" for k, v in params.items()))
    parts.append(f"made with {software_name()}")
    description = _ascii(" | ".join(parts))

    # --- machine-readable facts --------------------------------------------
    facts = {"software": software_name(), "process": process, "kind": kind, "params": params,
             "frames": n, "first_frame": first_no, "last_frame": last_no, "sequence_frames": total,
             "source": source_name}
    if kind == "video":
        facts.update(blend=blend, step=step, output_frames=outputs, fps=fps)
    if times:
        facts.update(start=times[0].isoformat(), end=times[-1].isoformat(),
                     duration_s=round((times[-1] - times[0]).total_seconds(), 3))
    if interval:
        facts["interval_s"] = round(interval, 4)
    if exposures:
        facts["exposure_s"] = {"min": min(exposures), "max": max(exposures), "total": round(sum(exposures), 4)}
    if cam or camera.lens:
        facts["camera"] = cam
        facts["lens"] = camera.lens

    icc, xmp = _icc_and_xmp(reference) if reference else (None, None)
    return OutputMetadata(title=_ascii(title), description=description, facts=facts,
                          capture_time=times[-1] if times else None, start_time=times[0] if times else None,
                          reference=reference, camera=camera, icc=icc, xmp=xmp)


# ---------------------------------------------------------------------------
# Writing with exiftool (one persistent process per thread)
# ---------------------------------------------------------------------------

_local = threading.local()
_processes: list = []
_processes_lock = threading.Lock()


def _exiftool():
    exe = files.find_exiftool()
    if exe is None:
        return None
    et = getattr(_local, "et", None)
    if et is None or not et.running:
        import exiftool

        et = exiftool.ExifToolHelper(executable=exe, common_args=["-charset", "filename=utf8"])
        et.run()
        _local.et = et
        with _processes_lock:
            _processes.append(et)
    return et


@atexit.register
def close_exiftool() -> None:
    with _processes_lock:
        for et in _processes:
            try:
                et.terminate()
            except Exception:
                pass
        _processes.clear()


def exiftool_available() -> bool:
    return files.find_exiftool() is not None


def write_with_exiftool(path, meta: OutputMetadata, timestamp: Optional[datetime]) -> bool:
    """Copy camera/lens/XMP/IPTC metadata from the reference frame and write our fields.
    Returns False (after logging why) if it couldn't."""
    et = _exiftool()
    if et is None:
        return False
    args = ["-overwrite_original", "-m", "-charset", "iptc=utf8", "-IPTC:CodedCharacterSet=utf8"]
    if meta.reference:
        args += ["-TagsFromFile", meta.reference, "-EXIF:all", "-XMP:all", "-IPTC:all",
                 "--XMP-xmpMM:all",          # document IDs: the output is a new image
                 "--ImageUniqueID", "--ThumbnailImage", "--PreviewImage"]
    args += [f"-EXIF:ImageDescription={meta.description}", f"-XMP-dc:Description={meta.description}",
             f"-IPTC:Caption-Abstract={meta.description}",
             f"-XMP-dc:Title={meta.title}", f"-IPTC:ObjectName={meta.title[:64]}",
             f"-EXIF:UserComment={meta.comment_json}",
             f"-EXIF:Software={software_name()}", f"-XMP-xmp:CreatorTool={software_name()}",
             "-EXIF:SubSecTimeOriginal=", "-EXIF:SubSecTimeDigitized=", "-EXIF:SubSecTime="]
    if timestamp is not None:
        stamp = timestamp.strftime("%Y:%m:%d %H:%M:%S")
        args += [f"-EXIF:DateTimeOriginal={stamp}", f"-EXIF:CreateDate={stamp}", f"-EXIF:ModifyDate={stamp}",
                 f"-XMP-exif:DateTimeOriginal={stamp}", f"-XMP-photoshop:DateCreated={stamp}",
                 f"-XMP-xmp:CreateDate={stamp}"]
    try:
        et.execute(*args, str(path))
        return True
    except Exception as exc:
        logger.warning("Could not write full metadata to %s with exiftool: %s", Path(path).name, exc)
        return False


def ffmpeg_args(meta: Optional[OutputMetadata]) -> List[str]:
    """``-metadata`` options for a video file."""
    if meta is None:
        return []
    args = ["-metadata", f"title={meta.title}", "-metadata", f"comment={meta.description}",
            "-metadata", f"description={meta.description}", "-metadata", f"encoder={software_name()}"]
    if meta.start_time is not None:
        args += ["-metadata", f"creation_time={meta.start_time:%Y-%m-%d %H:%M:%S}"]
    return args
