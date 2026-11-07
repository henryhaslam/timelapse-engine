"""Stack raw files directly and save the result as a DNG to develop in Lightroom.

The usual route exports every frame from Lightroom as a 16-bit TIFF, stacks the TIFFs and
edits the result. This module skips the export: it stacks the camera's raw sensor data
(the Bayer mosaic, before demosaicing) and writes one DNG. Lightroom then demosaics and
develops that single file like a raw from the camera, with full white balance and
highlight latitude.

* Raw data is linear, so an average is a true long exposure (an average of gamma-encoded
  TIFFs makes moving highlights too dark).
* Only per-pixel stacks are meaningful on a mosaic: each photosite is one colour, and
  neighbouring photosites are different colours. ``PHOTO_STACKS`` lists them. Effects that
  mix colour channels or neighbouring pixels (entropy, colour maps, motion) need
  developed RGB frames.
* The DNG keeps the camera's CFA layout, black and white levels, colour matrix and as-shot
  white balance, and gets the camera EXIF of the last frame plus the usual caption.

Needs ``rawpy`` (``pip install rawpy``). Command line:
``python -m timelapse_engine.rawstack FOLDER --process average``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import files
from .progress import CancelToken, ProgressCallback, logger
from .sources import FrameSource

PHOTO_STACKS = ("average", "maximum", "minimum", "median", "clipped_average", "comet_trails")
RAW_EXTENSIONS = (".3fr", ".arw", ".cr2", ".cr3", ".dng", ".erf", ".iiq", ".nef", ".nrw", ".orf", ".pef",
                  ".raf", ".rw2", ".rwl", ".sr2", ".srf", ".srw")
_CFA_CODES = {"R": 0, "G": 1, "B": 2}
_LIBRAW_FLIP_TO_TIFF = {0: 1, 3: 3, 5: 8, 6: 6}   # LibRaw sizes.flip -> TIFF Orientation
DEFAULT_CROP_MARGIN = 4                          # px trimmed on each side (demosaic edges), as raw converters do


def _rawpy():
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError("Raw stacking needs rawpy: pip install rawpy  (or pip install -e .[raw])") from exc
    return rawpy


# ---------------------------------------------------------------------------
# Raw file details
# ---------------------------------------------------------------------------

@dataclass
class RawInfo:
    """What a DNG needs to describe the camera's sensor data."""

    width: int
    height: int
    cfa: Tuple[int, int, int, int]              # 2x2 CFA in DNG colour codes, row by row (0=R, 1=G, 2=B)
    black: Tuple[int, int, int, int]            # black level per CFA position
    white: int
    color_matrix: np.ndarray                    # 3x3, XYZ (D65) -> camera RGB (DNG ColorMatrix1)
    as_shot_neutral: Tuple[float, float, float]  # camera RGB of a neutral grey (inverse WB multipliers)
    make: str = ""
    model: str = ""
    orientation: int = 1                        # TIFF Orientation

    @property
    def cfa_name(self) -> str:
        return "".join("RGB"[c] for c in self.cfa)

    def same_layout(self, other: "RawInfo") -> bool:
        return (self.width, self.height, self.cfa, self.black, self.white) == \
               (other.width, other.height, other.cfa, other.black, other.white)


def raw_info(path) -> RawInfo:
    """Read the sensor layout, levels, colour matrix and white balance of a raw file."""
    rawpy = _rawpy()
    with rawpy.imread(str(path)) as raw:
        desc = raw.color_desc.decode()
        colors = raw.raw_colors_visible[:2, :2]
        letters = [desc[c] for c in colors.ravel()]
        if sorted(set(letters)) != ["B", "G", "R"] or len(desc) != 4:
            raise ValueError(f"{Path(path).name}: only RGB Bayer sensors are supported (found {desc!r})")
        black_by_colour = raw.black_level_per_channel
        wb = [float(x) for x in raw.camera_whitebalance]
        g = wb[1] or 1.0
        neutral = tuple(g / x if x > 0 else 1.0 for x in (wb[0], wb[1], wb[2]))
        matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
        h, w = raw.raw_image_visible.shape
        return RawInfo(width=w, height=h, cfa=tuple(_CFA_CODES[c] for c in letters),
                       black=tuple(int(black_by_colour[c]) for c in colors.ravel()), white=int(raw.white_level),
                       color_matrix=matrix, as_shot_neutral=neutral,
                       orientation=_LIBRAW_FLIP_TO_TIFF.get(int(raw.sizes.flip), 1))


def read_mosaic(path) -> np.ndarray:
    """The visible sensor data (H, W) uint16, before demosaicing."""
    rawpy = _rawpy()
    with rawpy.imread(str(path)) as raw:
        return np.array(raw.raw_image_visible, dtype=np.uint16, copy=True)


def neutralise_clipped(mosaic: np.ndarray, info: RawInfo, level: float = 0.98) -> int:
    """Set every 2x2 CFA block that has a clipped photosite to neutral white, in place.

    Green photosites clip first. In a single frame the raw converter shows such a block as
    white, but once frames are averaged the clipped values look like ordinary data with
    too little green, and bright water or sky turns magenta. Neutral white is what the
    converter would have shown. Returns the number of blocks changed.
    """
    h, w = mosaic.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    blocks = mosaic.reshape(h, w)[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).transpose(0, 2, 1, 3)
    black = np.asarray(info.black, dtype=np.float64).reshape(2, 2)
    threshold = (black + (info.white - black) * level).astype(np.uint16)
    clipped = (blocks >= threshold).any(axis=(2, 3))
    count = int(clipped.sum())
    if count:
        neutral = np.asarray(info.as_shot_neutral, dtype=np.float64)
        neutral = neutral / neutral.max()
        target = black + (info.white - black) * neutral[np.asarray(info.cfa)].reshape(2, 2)
        blocks[clipped] = np.round(target).astype(np.uint16)
    return count


# ---------------------------------------------------------------------------
# Frame source
# ---------------------------------------------------------------------------

def list_raws(folder) -> List[str]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    return [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS]


class RawSequence(FrameSource):
    """A folder (or list) of raw files, read as sensor mosaics: frames are (H, W, 1) uint16.

    Frames are ordered by capture time. Every frame must have the same sensor layout
    (size, CFA, black and white level), which is checked on the first and last frame.
    ``neutral_highlights`` (default on) turns clipped highlights neutral white in each
    frame before stacking (see :func:`neutralise_clipped`).
    """

    def __init__(self, folder=None, paths: Optional[Sequence[str]] = None, workers: int = 8,
                 exiftool_path: Optional[str] = None, neutral_highlights: bool = True):
        if (folder is None) == (paths is None):
            raise ValueError("Pass exactly one of 'folder' or 'paths'.")
        paths = list_raws(folder) if paths is None else [str(p) for p in paths]
        if not paths:
            raise FileNotFoundError(f"No raw files found in {folder}")
        self.name = files.shoot_name(folder if folder is not None else Path(paths[0]).parent)
        self.workers = workers
        self.neutral_highlights = neutral_highlights
        self._pool: Optional[ThreadPoolExecutor] = None
        self._info: Dict[str, files.FrameInfo] = files.read_frame_info(paths, exiftool_path, workers)
        self._times = {p: fi.time for p, fi in self._info.items()}
        self.paths = files.sort_by_capture_time(paths, self._times)
        self.raw = raw_info(self.paths[0])
        first = self._info.get(self.paths[0]) or files.FrameInfo()
        self.raw.make, self.raw.model = first.make, first.model     # rawpy doesn't report them; EXIF does
        if len(self.paths) > 1 and not self.raw.same_layout(raw_info(self.paths[-1])):
            raise files.SequenceMismatchError([(self.paths[-1], "different sensor layout from the first frame")])
        logger.info("%d raw frames found in %s (%dx%d %s).", len(self.paths), self.name, self.raw.width,
                    self.raw.height, self.raw.cfa_name)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="raw")
        return self._pool

    def probe(self):
        return (self.raw.height, self.raw.width, 1), np.dtype(np.uint16)

    def _load(self, index: int) -> np.ndarray:
        mosaic = read_mosaic(self.paths[index])
        if mosaic.shape != (self.raw.height, self.raw.width):
            raise files.SequenceMismatchError([(self.paths[index], f"sensor data {mosaic.shape[::-1]}")])
        if self.neutral_highlights:
            neutralise_clipped(mosaic, self.raw)
        return mosaic[:, :, None]

    def read_raw(self, indices: Sequence[int]) -> List[np.ndarray]:
        return list(self.pool.map(self._load, indices))

    def read_into(self, indices: Sequence[int], out: np.ndarray) -> None:
        def fill(k_i):
            k, i = k_i
            out[k] = self._load(i)

        list(self.pool.map(fill, enumerate(indices)))

    def frame_info(self, index: int) -> Optional[files.FrameInfo]:
        return self._info.get(self.paths[index])

    def frame_path(self, index: int) -> Optional[str]:
        return self.paths[index]

    def capture_time(self, index: int) -> Optional[datetime]:
        return self._times.get(self.paths[index])

    def latest_capture_time(self) -> Optional[datetime]:
        times = [t for t in self._times.values() if t is not None]
        return max(times) if times else None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


# ---------------------------------------------------------------------------
# DNG writing
# ---------------------------------------------------------------------------

def _rational(values, den: int = 1_000_000) -> List[int]:
    out: List[int] = []
    for v in values:
        out += [int(round(float(v) * den)), den]
    return out


def write_dng(path, mosaic: np.ndarray, info: RawInfo, unique_model: Optional[str] = None,
              software: str = "") -> Path:
    """Write a CFA (Bayer) mosaic as an uncompressed DNG that raw developers treat like the camera's raw.

    ``mosaic``: (H, W) or (H, W, 1) uint16 in the camera's own levels (``info.black`` to ``info.white``).
    """
    import tifffile

    mosaic = np.ascontiguousarray(np.asarray(mosaic).reshape(info.height, info.width))
    if mosaic.dtype != np.uint16:
        raise TypeError("mosaic must be uint16 in the camera's levels")
    model = unique_model or " ".join(x for x in (info.make, info.model) if x) or "Unknown camera"
    m = DEFAULT_CROP_MARGIN if min(info.width, info.height) > 8 * DEFAULT_CROP_MARGIN else 0
    tags = [
        (50706, 1, 4, (1, 4, 0, 0), True),                         # DNGVersion
        (50707, 1, 4, (1, 1, 0, 0), True),                         # DNGBackwardVersion
        (50708, "s", 0, model, True),                               # UniqueCameraModel
        (33421, 3, 2, (2, 2), True),                                # CFARepeatPatternDim
        (33422, 1, 4, tuple(info.cfa), True),                       # CFAPattern
        (50710, 1, 3, (0, 1, 2), True),                             # CFAPlaneColor
        (50711, 3, 1, 1, True),                                     # CFALayout: rectangular
        (50713, 3, 2, (2, 2), True),                                # BlackLevelRepeatDim
        (50714, 4, 4, tuple(info.black), True),                     # BlackLevel per CFA position
        (50717, 4, 1, int(info.white), True),                       # WhiteLevel
        (50719, 4, 2, (m, m), True),                                # DefaultCropOrigin
        (50720, 4, 2, (info.width - 2 * m, info.height - 2 * m), True),   # DefaultCropSize
        (50721, 10, 9, _rational(info.color_matrix.ravel()), True),       # ColorMatrix1 (XYZ -> camera)
        (50778, 3, 1, 21, True),                                    # CalibrationIlluminant1: D65
        (50728, 5, 3, _rational(info.as_shot_neutral), True),       # AsShotNeutral
        (274, 3, 1, int(info.orientation), True),                   # Orientation
    ]
    if info.make:
        tags.append((271, "s", 0, info.make, True))
    if info.model:
        tags.append((272, "s", 0, info.model, True))
    path = Path(path)
    tifffile.imwrite(path, mosaic, photometric=32803, compression=None, metadata=None, software=software or None,
                     extratags=tags)
    return path


# ---------------------------------------------------------------------------
# Stack to DNG
# ---------------------------------------------------------------------------

def stack_raw_to_dng(source, process: str = "average", export_dir=None, export_name: Optional[str] = None,
                     params: Optional[dict] = None, overwrite: str = "increment",
                     timestamp_offset: Optional[timedelta] = timedelta(minutes=1),
                     progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None,
                     device: str = "auto", batch_size: Optional[int] = None) -> Path:
    """Stack raw frames with a per-pixel ``process`` and save the result as a DNG.

    ``source``: a :class:`RawSequence`, a folder of raw files, or a list of raw paths.
    Returns the DNG path (``<export_dir>/<name>_<process>.dng``; ``export_dir`` defaults
    to the folder above the raw files).
    """
    from . import metadata
    from .engine import stack
    from .processors import create_processor

    if process not in PHOTO_STACKS:
        raise ValueError(f"'{process}' can't run on raw sensor data; use one of: {', '.join(PHOTO_STACKS)}")
    own = not isinstance(source, RawSequence)
    if own:
        source = RawSequence(paths=source) if isinstance(source, (list, tuple)) else RawSequence(source)
    try:
        params = dict(params or {})
        result = stack(source, create_processor(process, **params), batch_size=batch_size, progress=progress,
                       cancel=cancel, device=device)
        mosaic = np.clip(result[..., 0] * 65535.0 + 0.5, 0, source.raw.white).astype(np.uint16)

        first = source.frame_path(0)
        export_dir = Path(export_dir) if export_dir else Path(first).parent.parent
        export_dir.mkdir(parents=True, exist_ok=True)
        out = files.resolve_output_path(export_dir / f"{export_name or source.name}_{process}.dng", overwrite)
        write_dng(out, mosaic, source.raw, software=metadata.software_name())

        meta = metadata.summarise(source, range(len(source)), process, params, kind="stack")
        meta.description = meta.description.replace(" stack of ", " stack (raw sensor data) of ", 1)
        stamp = meta.capture_time + timestamp_offset if meta.capture_time and timestamp_offset else meta.capture_time
        if metadata.write_with_exiftool(out, meta, stamp):
            _copy_lens_name(out, meta.reference)
        else:
            logger.warning("Saved %s without camera EXIF (exiftool not available).", out.name)
        logger.info("Saved %s: %s", out.name, meta.title)
        return out
    finally:
        if own:
            source.close()


def _copy_lens_name(dng: Path, reference: Optional[str]) -> None:
    """Some cameras (Panasonic, Olympus) keep the lens name only in their maker notes; a
    developer reading the DNG looks for EXIF LensModel, so fill it in from there."""
    from . import metadata

    et = metadata._exiftool()
    if et is None or not reference:
        return
    try:
        et.execute("-overwrite_original", "-m", "-TagsFromFile", reference,
                   "-EXIF:LensModel<Panasonic:LensType", "-EXIF:LensModel<Olympus:LensType",
                   "-EXIF:LensModel<LensModel", "-EXIF:LensSerialNumber<Panasonic:LensSerialNumber",
                   "-EXIF:LensSerialNumber<LensSerialNumber", str(dng))
    except Exception as exc:
        logger.debug("Lens name not copied to %s: %s", dng.name, exc)


def main(argv=None) -> int:
    from .progress import ConsoleProgress, enable_console_logging

    ap = argparse.ArgumentParser(prog="python -m timelapse_engine.rawstack",
                                 description="Stack raw files and save a DNG to develop in Lightroom.")
    ap.add_argument("folder", help="folder of raw files (one sequence)")
    ap.add_argument("--process", default="average", choices=PHOTO_STACKS)
    ap.add_argument("--out", help="output folder (default: the folder above FOLDER)")
    ap.add_argument("--name", help="output name (default: the folder name)")
    ap.add_argument("--first", type=int, help="first frame to use (1-based)")
    ap.add_argument("--last", type=int, help="last frame to use (1-based, inclusive)")
    ap.add_argument("--cpu", action="store_true", help="don't use the GPU")
    ap.add_argument("--keep-clipped", action="store_true",
                    help="stack clipped highlights as they are (they can turn magenta in averages)")
    args = ap.parse_args(argv)

    enable_console_logging()
    seq = RawSequence(args.folder, neutral_highlights=not args.keep_clipped)
    try:
        paths = seq.paths[(args.first or 1) - 1:args.last or len(seq.paths)]
        source = seq if len(paths) == len(seq.paths) else \
            RawSequence(paths=paths, neutral_highlights=not args.keep_clipped)
        if source is not seq:
            source.name = seq.name
        out = stack_raw_to_dng(source, args.process, args.out, args.name, progress=ConsoleProgress(),
                               device="cpu" if args.cpu else "auto")
        print(f"Saved {out}")
    finally:
        seq.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
