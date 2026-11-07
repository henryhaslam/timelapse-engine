"""Sort a card's worth of raw files into one folder per timelapse sequence.

This is the step before the engine. A card dump is one long list of shots, and the
timelapse sequences in it are runs of evenly spaced shots: the camera fires every 0.5 s,
2 s or 30 s, then stops. This module reads each shot's capture time, splits the list
wherever the spacing breaks, and moves (or copies) each sequence into its own folder,
ready for Lightroom and then the engine.

How sequences are found (:func:`find_sequences`):

* Shots are put in capture order, separately for each camera body.
* A sequence is a run of at least ``min_frames`` shots whose spacing stays within
  ``tolerance`` of the recent spacing. Comparing with the recent spacing, not the first,
  keeps a slowly ramped interval together.
* A gap that doesn't fit the spacing ends the sequence. The next one may have the same
  spacing or a new one. When the spacing changes with no gap, the shot where it changes
  ends the earlier sequence.
* When the camera records a burst/interval shot counter (Panasonic ``SequenceNumber``
  and similar), a counter that restarts also ends the sequence. That catches a restart
  whose gap happens to look like the spacing.
* Shots in no sequence are "single shots" and stay where they are unless asked.

Files that share a name (``P1000123.RW2`` + ``P1000123.JPG`` + ``P1000123.xmp``) are one
shot and always move together. Nothing is ever overwritten, and every sort writes a log
next to the new folders that :func:`undo` uses to put the files back.

Command line: ``python -m timelapse_engine.sorter FOLDER`` shows what it would do;
add ``--move`` or ``--copy`` to do it, or ``--undo`` to reverse the last sort.
"""

from __future__ import annotations

import argparse
import base64
import errno
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .files import _norm, _pillow_frame_info, find_exiftool, load_small, natural_sort_key
from .progress import CancelToken, Cancelled, ProgressCallback, check_cancel, logger, report

RAW_EXTENSIONS = frozenset({
    ".3fr", ".arw", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf", ".fff", ".gpr", ".iiq", ".kdc", ".mos",
    ".mrw", ".nef", ".nrw", ".orf", ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f"})
PHOTO_EXTENSIONS = frozenset({".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".hif"})
SHOT_EXTENSIONS = RAW_EXTENSIONS | PHOTO_EXTENSIONS
SIDECAR_EXTENSIONS = frozenset({".xmp", ".pp3", ".dop", ".thm", ".on1"})

DEFAULT_TEMPLATE = "{date}_{time}_{interval}_{frames}f"
TEMPLATE_FIELDS = ("date", "time", "interval", "frames", "camera", "number")
SINGLES_NAME = "singles"
LOG_PREFIX = "_timelapse-sort-"

# Timestamp slack: EXIF times without sub-seconds are truncated to the second, so two
# shots 2.0 s apart can read as 1 s or 3 s. With sub-seconds the camera's own jitter
# (buffer writes during a burst) is the limit.
SLACK_PRECISE = 0.1
SLACK_WHOLE_SECONDS = 1.0
RECENT_SPACINGS = 5          # spacing reference: median of the last few gaps in the run

_EXIFTOOL_TAGS = ["-SubSecDateTimeOriginal", "-DateTimeOriginal", "-SubSecTimeOriginal", "-CreateDate",
                  "-Make", "-Model", "-SerialNumber", "-SequenceNumber", "-ExposureTime"]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash from the GUI on Windows


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    """One exposure: the file its capture time comes from plus same-name companions."""

    path: str                                   # raw file if there is one, else the JPEG/TIFF/HEIF
    companions: List[str] = field(default_factory=list)   # JPEG of a RAW+JPEG pair, .xmp sidecars...
    time: Optional[datetime] = None
    precise: bool = False                       # capture time has sub-seconds
    camera: str = ""                            # make, model and serial: bodies are never mixed
    counter: Optional[int] = None               # the camera's burst/interval shot counter
    exposure: Optional[float] = None            # seconds
    from_file_time: bool = False                # no EXIF time: the file's modified time is used

    @property
    def files(self) -> List[str]:
        return [self.path, *self.companions]

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def stem(self) -> str:
        return Path(self.path).stem


@dataclass
class Group:
    """A timelapse sequence (or, with ``singles=True``, the shots in no sequence)."""

    shots: List[Shot]
    interval: float = 0.0                       # typical spacing in seconds (median gap)
    name: str = ""                              # folder name
    singles: bool = False

    @property
    def frames(self) -> int:
        return len(self.shots)

    @property
    def start(self) -> Optional[datetime]:
        return self.shots[0].time if self.shots else None

    @property
    def end(self) -> Optional[datetime]:
        return self.shots[-1].time if self.shots else None

    @property
    def duration(self) -> float:
        return (self.end - self.start).total_seconds() if self.shots else 0.0

    @property
    def camera(self) -> str:
        models = {s.camera.split(" #")[0] for s in self.shots if s.camera}
        return ", ".join(sorted(models))

    @property
    def files(self) -> List[str]:
        return [f for s in self.shots for f in s.files]

    def describe(self) -> str:
        if self.singles:
            return f"{self.frames} single shot{'s' if self.frames != 1 else ''}"
        return (f"{self.frames} frames every {fmt_interval(self.interval)} from {self.start:%Y-%m-%d %H:%M:%S}, "
                f"lasting {fmt_duration(self.duration)}")


@dataclass
class Analysis:
    """Everything :func:`analyse` found in a folder."""

    folder: str
    groups: List[Group]                         # sequences, in time order
    singles: Group                              # shots in no sequence
    undated: List[Shot]                         # shots with no capture time at all: left alone
    ignored: List[str]                          # other files (videos, orphan sidecars...): left alone
    shots: List[Shot] = field(default_factory=list)   # every dated shot, in time order

    @property
    def dated(self) -> int:
        return len(self.shots)

    def summary(self) -> str:
        parts = [f"{len(self.groups)} sequence{'s' if len(self.groups) != 1 else ''} "
                 f"({sum(g.frames for g in self.groups)} shots)"]
        if self.singles.frames:
            parts.append(f"{self.singles.frames} single shot{'s' if self.singles.frames != 1 else ''}")
        if self.undated:
            parts.append(f"{len(self.undated)} without a capture time")
        if self.ignored:
            parts.append(f"{len(self.ignored)} other file{'s' if len(self.ignored) != 1 else ''} left alone")
        return ", ".join(parts)


@dataclass
class Move:
    src: str
    dst: str


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_interval(seconds: float) -> str:
    """0.48 -> '0.48s', 1.0 -> '1s', 2.5 -> '2.5s', 30 -> '30s'."""
    if seconds >= 10:
        text = f"{seconds:.0f}"
    elif seconds >= 1:
        text = f"{seconds:.1f}".rstrip("0").rstrip(".")
    else:
        text = f"{seconds:.2f}".rstrip("0").rstrip(".")
    return f"{text or '0'}s"


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    return f"{m} min {s:02d} s" if m else f"{s} s"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _shot_stem(path: Path) -> Optional[str]:
    """Name shared by all files of one shot, or None if the file isn't part of a shot.
    ``P1.RW2`` and ``P1.JPG`` -> ``P1``; sidecars ``P1.xmp`` and ``P1.RW2.xmp`` -> ``P1``."""
    ext = path.suffix.lower()
    if ext in SHOT_EXTENSIONS:
        return path.stem
    if ext in SIDECAR_EXTENSIONS:
        inner = Path(path.stem)
        return inner.stem if inner.suffix.lower() in SHOT_EXTENSIONS else path.stem
    return None


def collect_shots(folder, subfolders: bool = False) -> Tuple[List[Shot], List[str]]:
    """Group the files in ``folder`` into shots (no metadata read yet).

    Returns (shots in file-name order, other files that will be left alone).
    """
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    entries = root.rglob("*") if subfolders else root.iterdir()
    by_key: Dict[Tuple[str, str], List[Path]] = {}
    ignored: List[str] = []
    for p in entries:
        if not p.is_file() or p.name.startswith(LOG_PREFIX):
            continue
        stem = _shot_stem(p)
        if stem is None:
            ignored.append(str(p))
            continue
        by_key.setdefault((_norm(p.parent), stem.lower()), []).append(p)

    shots: List[Shot] = []
    for paths in by_key.values():
        main = sorted((p for p in paths if p.suffix.lower() in SHOT_EXTENSIONS),
                      key=lambda p: (p.suffix.lower() not in RAW_EXTENSIONS, p.name.lower()))
        if not main:                       # a sidecar whose image isn't here
            ignored.extend(str(p) for p in paths)
            continue
        companions = sorted((p for p in paths if p != main[0]), key=lambda p: p.name.lower())
        shots.append(Shot(path=str(main[0]), companions=[str(p) for p in companions]))
    shots.sort(key=lambda s: natural_sort_key(s.path))
    return shots, ignored


def _parse_time(value) -> Tuple[Optional[datetime], bool]:
    """'2026:11:07 16:19:00.015+00:00' -> (datetime with microseconds, True)."""
    if not value or not isinstance(value, str) or len(value) < 19:
        return None, False
    try:
        dt = datetime.strptime(value[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None, False
    rest = value[19:]
    if rest.startswith("."):
        digits = ""
        for ch in rest[1:]:
            if not ch.isdigit():
                break
            digits += ch
        if digits:
            return dt.replace(microsecond=int((digits + "000000")[:6])), True
    return dt, False


def _int_or_none(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _apply_exiftool(shot: Shot, meta: dict) -> None:
    time, precise = _parse_time(meta.get("SubSecDateTimeOriginal"))
    if time is None:
        time, _ = _parse_time(meta.get("DateTimeOriginal") or meta.get("CreateDate"))
        sub = str(meta.get("SubSecTimeOriginal") or "")
        digits = "".join(ch for ch in sub if ch.isdigit())
        if time is not None and digits:
            time, precise = time.replace(microsecond=int((digits + "000000")[:6])), True
    shot.time, shot.precise = time, precise
    make, model = str(meta.get("Make") or "").strip(), str(meta.get("Model") or "").strip()
    if model.lower().startswith(make.lower().split(" ")[0]) and make:
        make = ""                                   # "Canon" + "Canon EOS R5" -> "Canon EOS R5"
    camera = f"{make} {model}".strip()
    serial = str(meta.get("SerialNumber") or "").strip()
    shot.camera = f"{camera} #{serial}" if serial else camera
    shot.counter = _int_or_none(meta.get("SequenceNumber"))
    try:
        shot.exposure = float(meta["ExposureTime"]) if meta.get("ExposureTime") not in (None, "") else None
    except (TypeError, ValueError):
        shot.exposure = None


def _run_exiftool(exe: str, args: List[str], paths: Sequence[str]) -> List[dict]:
    """Run exiftool over ``paths`` (passed in a UTF-8 argfile) and return its JSON records."""
    fd, argfile = tempfile.mkstemp(suffix=".args", prefix="timelapse-sort-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(paths))
        proc = subprocess.run([exe, "-json", "-charset", "filename=utf8", *args, "-@", argfile],
                              capture_output=True, creationflags=_NO_WINDOW, timeout=600)
    finally:
        try:
            os.remove(argfile)
        except OSError:
            pass
    out = proc.stdout.decode("utf-8", "replace").strip()
    if not out:
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip() or f"exiftool exit {proc.returncode}")
        return []
    return json.loads(out)


def read_shot_times(shots: List[Shot], use_exiftool: bool = True, exiftool_path: Optional[str] = None,
                    workers: int = 4, progress: Optional[ProgressCallback] = None,
                    cancel: Optional[CancelToken] = None) -> None:
    """Fill in capture time, camera, counter and exposure for each shot (in place).

    exiftool reads every raw format; several exiftool processes run side by side on
    chunks of files. Without it, JPEG/TIFF EXIF is read with Pillow. Shots still without
    a time fall back to a same-name JPEG, then to the file's modified time.
    """
    total = len(shots)
    report(progress, 0, total, "Reading capture times")
    exe = find_exiftool(exiftool_path) if use_exiftool else None
    pending = list(shots)
    if exe and shots:
        chunk = max(16, min(150, -(-total // (workers * 3))))
        chunks = [shots[i:i + chunk] for i in range(0, total, chunk)]
        args = ["-n", "-fast", *_EXIFTOOL_TAGS]
        done = 0
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [(c, pool.submit(_run_exiftool, exe, args, [s.path for s in c])) for c in chunks]
            for chunk_shots, fut in futures:
                check_cancel(cancel)
                lookup = {_norm(s.path): s for s in chunk_shots}
                for meta in fut.result():
                    shot = lookup.get(_norm(meta.get("SourceFile", "")))
                    if shot is not None:
                        _apply_exiftool(shot, meta)
                done += len(chunk_shots)
                report(progress, done, total, "Reading capture times")
        except Cancelled:
            raise
        except Exception as exc:
            logger.warning("exiftool failed (%s); reading EXIF with Pillow instead.", exc)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        pending = [s for s in shots if s.time is None]
    elif shots and use_exiftool:
        logger.warning("exiftool not found: raw files will be dated by file modified time. "
                       "Put an exiftool-* folder next to the repo or set TIMELAPSE_EXIFTOOL.")

    # Pillow for JPEG/TIFF (the shot itself, or the JPEG half of a RAW+JPEG pair)
    def pillow(shot: Shot) -> None:
        for path in shot.files:
            if Path(path).suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
                fi = _pillow_frame_info(path)
                if fi.time is not None:
                    shot.time, shot.precise = fi.time, fi.time.microsecond != 0
                    shot.camera = shot.camera or f"{fi.make} {fi.model}".strip()
                    shot.exposure = shot.exposure or fi.exposure
                    return

    if pending:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(pillow, pending))
    stale = [s for s in pending if s.time is None]
    for shot in stale:
        try:
            shot.time = datetime.fromtimestamp(os.path.getmtime(shot.path))
            shot.precise, shot.from_file_time = False, True
        except OSError:
            pass
    if stale:
        logger.warning("%d shot(s) have no capture time in EXIF; using the file modified time for them.",
                       len(stale))
    report(progress, total, total, "Reading capture times")


# ---------------------------------------------------------------------------
# Finding sequences
# ---------------------------------------------------------------------------

def _fits(gap: float, reference: float, tolerance: float, slack: float) -> bool:
    return abs(gap - reference) <= max(slack, tolerance * reference)


def _counter_useful(shots: Sequence[Shot]) -> bool:
    """Only trust a camera's shot counter if it actually counts up shot by shot somewhere."""
    return any(a.counter and b.counter and b.counter == a.counter + 1 for a, b in zip(shots, shots[1:]))


def _counter_break(a: Shot, b: Shot) -> bool:
    ca, cb = a.counter or 0, b.counter or 0
    if ca <= 0 and cb <= 0:
        return False                 # single-shot drive mode: the counter says nothing
    if ca > 0 and cb > 0:
        return cb <= ca              # restarted (a jump forward is left to the spacing test)
    return True                      # entering or leaving burst/interval shooting


def spacing_runs(shots: Sequence[Shot], tolerance: float = 0.25, use_counter: bool = True) -> List[Tuple[int, int]]:
    """Index ranges ``(first, last)`` of evenly spaced runs of at least 3 shots.

    ``shots`` must be one camera's shots in time order. Shots outside every run are
    single shots.
    """
    n = len(shots)
    if n < 3:
        return []
    gaps = [(shots[i + 1].time - shots[i].time).total_seconds() for i in range(n - 1)]
    slack = [SLACK_PRECISE if shots[i].precise and shots[i + 1].precise else SLACK_WHOLE_SECONDS
             for i in range(n - 1)]
    counters = use_counter and _counter_useful(shots)
    cut = [counters and _counter_break(shots[i], shots[i + 1]) for i in range(n - 1)]

    runs: List[Tuple[int, int]] = []
    last_end = -1
    i = 0
    while i < n - 1:
        if cut[i]:
            i += 1
            continue
        recent = [gaps[i]]
        j = i + 1
        while j < n - 1 and not cut[j] and _fits(gaps[j], statistics.median(recent[-RECENT_SPACINGS:]),
                                                  tolerance, slack[j]):
            recent.append(gaps[j])
            j += 1
        first, last = i, j          # gaps i..j-1 join shots i..j
        if first == last_end:       # spacing changed with no gap: that shot ends the earlier run
            first += 1
        if last - first + 1 >= 3:
            runs.append((first, last))
            last_end = last
            i = j
        else:
            i += 1
    return runs


def find_sequences(shots: Sequence[Shot], tolerance: float = 0.25, min_frames: int = 5,
                   use_counter: bool = True) -> Tuple[List[Group], Group]:
    """Split dated shots into sequences. Returns (sequences in time order, single shots)."""
    dated = sorted((s for s in shots if s.time is not None), key=lambda s: (s.time, natural_sort_key(s.path)))
    by_camera: Dict[str, List[Shot]] = {}
    for s in dated:
        by_camera.setdefault(s.camera, []).append(s)

    groups: List[Group] = []
    singles: List[Shot] = []
    for cam_shots in by_camera.values():
        covered = 0
        for first, last in spacing_runs(cam_shots, tolerance, use_counter):
            singles.extend(cam_shots[covered:first])
            run = cam_shots[first:last + 1]
            covered = last + 1
            if len(run) >= max(3, min_frames):
                gaps = [(b.time - a.time).total_seconds() for a, b in zip(run, run[1:])]
                groups.append(Group(run, interval=statistics.median(gaps)))
            else:
                singles.extend(run)
        singles.extend(cam_shots[covered:])
    groups.sort(key=lambda g: (g.start, g.camera))
    singles.sort(key=lambda s: (s.time, natural_sort_key(s.path)))
    return groups, Group(singles, singles=True, name=SINGLES_NAME)


def _safe_name(name: str) -> str:
    cleaned = "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name).strip().rstrip(". ")
    return cleaned or "sequence"


def name_groups(groups: Sequence[Group], template: str = DEFAULT_TEMPLATE) -> None:
    """Give each group a unique folder name from ``template``.

    Fields: ``{date}`` 2026.06.07, ``{time}`` 131442, ``{interval}`` 2s, ``{frames}`` 270,
    ``{camera}`` DC-G9M2, ``{number}`` 1, 2, 3... in time order.
    """
    used = set()
    for number, g in enumerate(groups, 1):
        fields = {"date": f"{g.start:%Y.%m.%d}", "time": f"{g.start:%H%M%S}",
                  "interval": fmt_interval(g.interval), "frames": g.frames,
                  "camera": g.camera.replace(", ", "+") or "camera", "number": number}
        try:
            base = _safe_name(template.format(**fields))
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"Bad folder name pattern {template!r}: use the fields "
                             f"{', '.join('{' + f + '}' for f in TEMPLATE_FIELDS)}") from exc
        name, k = base, 2
        while name.lower() in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name.lower())
        g.name = name


def analyse(folder, subfolders: bool = False, tolerance: float = 0.25, min_frames: int = 5,
            use_counter: bool = True, template: str = DEFAULT_TEMPLATE, use_exiftool: bool = True,
            exiftool_path: Optional[str] = None, progress: Optional[ProgressCallback] = None,
            cancel: Optional[CancelToken] = None) -> Analysis:
    """Find the sequences in ``folder`` (nothing is moved)."""
    shots, ignored = collect_shots(folder, subfolders)
    read_shot_times(shots, use_exiftool=use_exiftool, exiftool_path=exiftool_path, progress=progress,
                    cancel=cancel)
    groups, singles = find_sequences(shots, tolerance, min_frames, use_counter)
    name_groups(groups, template)
    dated = sorted((s for s in shots if s.time is not None), key=lambda s: (s.time, natural_sort_key(s.path)))
    undated = [s for s in shots if s.time is None]
    result = Analysis(str(folder), groups, singles, undated, ignored, dated)
    logger.info("Sorter: %s in %s", result.summary(), folder)
    return result


# ---------------------------------------------------------------------------
# Moving files
# ---------------------------------------------------------------------------

def _renamed(name: str, stem: str, k: int) -> str:
    """P1.RW2 -> P1_2.RW2, keeping every file of a shot paired (P1.RW2.xmp -> P1_2.RW2.xmp)."""
    if k == 1:
        return name
    if name.lower().startswith(stem.lower()):
        return f"{name[:len(stem)]}_{k}{name[len(stem):]}"
    p = Path(name)
    return f"{p.stem}_{k}{p.suffix}"


def plan_moves(groups: Sequence[Group], dest_root) -> List[Move]:
    """Where each file goes: ``dest_root/<group name>/<file name>``.

    Files already in place are skipped. When a name is taken (two card folders both
    holding P1000001.RW2, or a file already at the destination), all files of that shot
    get the same ``_2`` suffix so raw, JPEG and sidecar stay paired. Nothing is overwritten.
    """
    dest_root = Path(dest_root)
    taken = set()
    moves: List[Move] = []
    for g in groups:
        folder = dest_root / g.name
        for shot in g.shots:
            k = 1
            while True:
                pairs = [(f, folder / _renamed(os.path.basename(f), shot.stem, k)) for f in shot.files]
                if all(_norm(d) == _norm(f) or (_norm(d) not in taken and not os.path.lexists(d))
                       for f, d in pairs):
                    break
                k += 1
            for f, d in pairs:
                taken.add(_norm(d))
                if _norm(d) != _norm(f):
                    moves.append(Move(f, str(d)))
    return moves


def _cross_device(exc: OSError) -> bool:
    return exc.errno == errno.EXDEV or getattr(exc, "winerror", None) == 17   # ERROR_NOT_SAME_DEVICE


def _move_file(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        raise FileExistsError(f"Not overwriting {dst}")
    try:
        os.rename(src, dst)                 # same drive: instant, whatever the size
    except OSError as exc:
        if not _cross_device(exc):
            raise
        partial = dst + ".partial"
        shutil.copy2(src, partial)
        os.rename(partial, dst)
        os.remove(src)


def _copy_file(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        raise FileExistsError(f"Not overwriting {dst}")
    partial = dst + ".partial"
    shutil.copy2(src, partial)
    os.rename(partial, dst)


def _write_log(path: Path, log: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def sort_files(groups: Sequence[Group], dest_root, mode: str = "move", progress: Optional[ProgressCallback] = None,
               cancel: Optional[CancelToken] = None) -> Path:
    """Move or copy each group into ``dest_root/<group.name>``. Returns the log path.

    The log lists every file that was moved or copied, even when the sort stops part way
    (an error or a cancel), so :func:`undo` can always put things back.
    """
    if mode not in ("move", "copy"):
        raise ValueError("mode must be 'move' or 'copy'")
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    moves = plan_moves(groups, dest_root)
    if mode == "copy":
        need = sum(os.path.getsize(m.src) for m in moves)
        free = shutil.disk_usage(dest_root).free
        if need > free * 0.98:
            raise OSError(f"Not enough space: copying needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free")

    stamp = datetime.now()
    log_path = dest_root / f"{LOG_PREFIX}{stamp:%Y-%m-%d_%H%M%S}.json"
    log = {"version": 1, "mode": mode, "created": stamp.isoformat(timespec="seconds"),
           "groups": [{"name": g.name, "frames": g.frames, "interval": g.interval} for g in groups],
           "folders": [], "moves": []}
    verb = "Moving" if mode == "move" else "Copying"
    action = _move_file if mode == "move" else _copy_file
    report(progress, 0, len(moves), f"{verb} files")
    try:
        for i, m in enumerate(moves, 1):
            check_cancel(cancel)
            parent = Path(m.dst).parent
            if not parent.is_dir():
                parent.mkdir(parents=True)
                log["folders"].append(str(parent))
            action(m.src, m.dst)
            log["moves"].append([m.src, m.dst])
            if i % 200 == 0:
                _write_log(log_path, log)
            report(progress, i, len(moves), f"{verb} files")
    finally:
        _write_log(log_path, log)
    logger.info("Sorter: %s %d files into %d folder%s in %s (log: %s)", "moved" if mode == "move" else "copied",
                len(log["moves"]), len(groups), "" if len(groups) == 1 else "s", dest_root, log_path.name)
    return log_path


def find_last_log(folder) -> Optional[Path]:
    """The most recent sort log in ``folder`` that hasn't been undone."""
    logs = sorted(Path(folder).glob(f"{LOG_PREFIX}*.json"), reverse=True)
    for path in logs:
        try:
            if not json.loads(path.read_text(encoding="utf-8")).get("undone"):
                return path
        except (OSError, ValueError):
            continue
    return None


def undo(log_path, progress: Optional[ProgressCallback] = None,
         cancel: Optional[CancelToken] = None) -> Tuple[int, List[str]]:
    """Reverse a sort from its log. Returns (files restored, problems).

    Moved files go back where they were; copies are deleted, but only while the original
    is still there at the same size. Folders the sort created are removed if empty.
    Anything that can't be restored safely is left where it is and listed.
    """
    log_path = Path(log_path)
    log = json.loads(log_path.read_text(encoding="utf-8"))
    if log.get("undone"):
        raise ValueError(f"{log_path.name} was already undone on {log['undone']}")
    moves = log.get("moves", [])
    problems: List[str] = []
    restored = 0
    report(progress, 0, len(moves), "Undoing sort")
    for i, (src, dst) in enumerate(reversed(moves), 1):
        check_cancel(cancel)
        if not os.path.exists(dst):
            problems.append(f"{dst} is no longer there")
        elif log["mode"] == "move":
            if os.path.lexists(src):
                problems.append(f"{src} exists again, so {os.path.basename(dst)} was left in its folder")
            else:
                Path(src).parent.mkdir(parents=True, exist_ok=True)
                _move_file(dst, src)
                restored += 1
        else:
            if os.path.exists(src) and os.path.getsize(src) == os.path.getsize(dst):
                os.remove(dst)
                restored += 1
            else:
                problems.append(f"original of {dst} is missing or different, so the copy was kept")
        report(progress, i, len(moves), "Undoing sort")
    for folder in reversed(log.get("folders", [])):
        try:
            os.rmdir(folder)
        except OSError:
            pass                      # not empty (something else is in it): keep it
    log["undone"] = datetime.now().isoformat(timespec="seconds")
    log["undo_problems"] = problems
    _write_log(log_path, log)
    logger.info("Sorter: undid %s, %d files restored%s", log_path.name, restored,
                f", {len(problems)} problem(s)" if problems else "")
    return restored, problems


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

_ROTATE = {3: 2, 6: 3, 8: 1}   # EXIF orientation -> np.rot90 quarter turns (anticlockwise)


def preview_image(path, max_side: int = 900, exiftool_path: Optional[str] = None) -> Optional[np.ndarray]:
    """A small RGB uint8 preview of a shot: decoded directly for JPEG/TIFF, otherwise the
    JPEG the camera embedded in the raw file (read with exiftool)."""
    import cv2

    path = str(path)
    if Path(path).suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
        img = load_small(path, max_side)
        if img.dtype != np.uint8:
            img = (img.astype(np.float32) * (255.0 / np.iinfo(img.dtype).max)).round().astype(np.uint8)
        return img
    exe = find_exiftool(exiftool_path)
    if exe is None:
        return None
    records = _run_exiftool(exe, ["-b", "-n", "-Orientation", "-PreviewImage", "-JpgFromRaw", "-ThumbnailImage"],
                            [path])
    if not records:
        return None
    meta = records[0]
    data = next((meta[k] for k in ("PreviewImage", "JpgFromRaw", "ThumbnailImage")
                 if isinstance(meta.get(k), str) and meta[k].startswith("base64:")), None)
    if data is None:
        return None
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(data[7:]), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    turns = _ROTATE.get(_int_or_none(meta.get("Orientation")) or 1, 0)
    return np.ascontiguousarray(np.rot90(rgb, turns)) if turns else rgb


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _print_plan(result: Analysis, dest_root: Path, include_singles: bool) -> None:
    print(f"\n{result.summary()}\n")
    width = max([len(g.name) for g in result.groups] + [len(SINGLES_NAME), 6])
    for g in result.groups:
        print(f"  {g.name:<{width}}  {g.frames:5d} frames  every {fmt_interval(g.interval):>6}  "
              f"{g.start:%Y-%m-%d %H:%M:%S}  {fmt_duration(g.duration):>12}  "
              f"{g.shots[0].name} - {g.shots[-1].name}")
    if result.singles.frames:
        where = f"-> {SINGLES_NAME}" if include_singles else "(left where they are)"
        print(f"  {'':<{width}}  {result.singles.frames:5d} single shots {where}")
    print(f"\nFolders go in {dest_root}")


def main(argv=None) -> int:
    from .progress import ConsoleProgress, enable_console_logging

    ap = argparse.ArgumentParser(prog="timelapse-sort",
                                 description="Sort raw files into one folder per timelapse sequence, using the "
                                             "spacing of their capture times. Without --move/--copy it only "
                                             "shows what it would do.")
    ap.add_argument("folder", help="folder of raw files (e.g. a card dump)")
    ap.add_argument("--subfolders", action="store_true", help="also look in subfolders (e.g. DCIM\\100_PANA)")
    ap.add_argument("--tolerance", type=float, default=25, help="spacing tolerance in %% (default 25)")
    ap.add_argument("--min-frames", type=int, default=5, help="fewest shots that count as a sequence (default 5)")
    ap.add_argument("--no-counter", action="store_true", help="ignore the camera's burst/interval shot counter")
    ap.add_argument("--names", default=DEFAULT_TEMPLATE,
                    help=f"folder name pattern (default {DEFAULT_TEMPLATE}; fields: "
                         f"{', '.join('{' + f + '}' for f in TEMPLATE_FIELDS)})")
    ap.add_argument("--dest", help="put the sequence folders here instead of inside FOLDER")
    ap.add_argument("--singles", action="store_true", help=f"also move single shots into '{SINGLES_NAME}'")
    act = ap.add_mutually_exclusive_group()
    act.add_argument("--move", action="store_true", help="move the files into the folders")
    act.add_argument("--copy", action="store_true", help="copy the files, leaving the originals")
    act.add_argument("--undo", action="store_true", help="undo the last sort into this folder (or --dest)")
    args = ap.parse_args(argv)

    enable_console_logging()
    dest_root = Path(args.dest or args.folder)
    if args.undo:
        log_path = find_last_log(dest_root)
        if log_path is None:
            print(f"No sort to undo in {dest_root}")
            return 1
        restored, problems = undo(log_path, progress=ConsoleProgress())
        print(f"Restored {restored} files from {log_path.name}")
        for p in problems:
            print(f"  ! {p}")
        return 0 if not problems else 2

    result = analyse(args.folder, subfolders=args.subfolders, tolerance=args.tolerance / 100,
                     min_frames=args.min_frames, use_counter=not args.no_counter, template=args.names,
                     progress=ConsoleProgress())
    groups = list(result.groups) + ([result.singles] if args.singles and result.singles.frames else [])
    _print_plan(result, dest_root, args.singles)
    if not (args.move or args.copy):
        print("Nothing moved. Add --move or --copy to sort the files.")
        return 0
    log_path = sort_files(groups, dest_root, mode="move" if args.move else "copy", progress=ConsoleProgress())
    dest_arg = f' --dest "{dest_root}"' if args.dest else ""
    print(f"Done (log: {log_path.name}). To put everything back:\n"
          f'  python -m timelapse_engine.sorter "{args.folder}" --undo{dest_arg}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
