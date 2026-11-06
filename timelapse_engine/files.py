"""Image file helpers: loading, sorting, validation, metadata and saving.

All pixel data inside the engine is float32 in the range 0.0 - 1.0 with shape
(H, W, C). Conversion to/from 8 or 16 bit only happens here.
"""

from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import tifffile
from PIL import Image

from .progress import logger

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff")
TIFF_EXTENSIONS = (".tif", ".tiff")

EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME_DIGITIZED = 36868
EXIF_DATETIME = 306
EXIF_MAKE = 271
EXIF_IFD_POINTER = 0x8769
TIFF_XMP_TAG = 700


class SequenceMismatchError(ValueError):
    """Raised when frames in a sequence differ in size or bit depth."""

    def __init__(self, mismatches: List[tuple]):
        self.mismatches = mismatches
        lines = "\n".join(f" - {os.path.basename(p)} ({info})" for p, info in mismatches[:50])
        more = f"\n ... and {len(mismatches) - 50} more" if len(mismatches) > 50 else ""
        super().__init__(f"{len(mismatches)} frame(s) differ from the first frame:\n{lines}{more}")


# ---------------------------------------------------------------------------
# Listing and sorting
# ---------------------------------------------------------------------------

def natural_sort_key(path) -> list:
    """Sort key that orders 'img2' before 'img10'."""
    name = os.path.basename(str(path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _norm(path) -> str:
    """Normalise a path so exiftool's forward-slash paths match glob's backslash paths."""
    return os.path.normcase(os.path.normpath(str(path)))


def list_images(folder, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[str]:
    """Return every image in ``folder`` (non-recursive, case-insensitive extension match)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    exts = tuple(e.lower() for e in extensions)
    return [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]


def shoot_name(path) -> str:
    """Name for a shoot: a Lightroom-style ``...\Shoot\export`` folder is called
    ``Shoot``; any other folder or file uses its own name (files without extension)."""
    p = Path(path)
    if p.suffix and not p.is_dir():
        return p.stem
    if p.name.lower().startswith("export") and p.parent != p and p.parent.name:
        return p.parent.name
    return p.name


def find_exiftool(explicit: Optional[str] = None) -> Optional[str]:
    """Locate exiftool: explicit path > $TIMELAPSE_EXIFTOOL > PATH > an exiftool-* folder next to this repo."""
    candidates = [explicit, os.environ.get("TIMELAPSE_EXIFTOOL"), shutil.which("exiftool")]
    repo_parent = Path(__file__).resolve().parents[2]
    candidates += [str(p) for p in sorted(repo_parent.glob("exiftool-*/exiftool.exe"))]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def _parse_exif_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    try:
        return datetime.strptime(str(value).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _with_subsec(dt: Optional[datetime], subsec) -> Optional[datetime]:
    """Add EXIF SubSecTimeOriginal ("303" = 0.303 s) to a capture time."""
    if dt is None or subsec in (None, ""):
        return dt
    digits = "".join(ch for ch in str(subsec) if ch.isdigit())
    return dt.replace(microsecond=int((digits + "000000")[:6])) if digits else dt


def _num(value) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


@dataclass
class FrameInfo:
    """Per-frame capture details read from EXIF (any field may be missing)."""

    time: Optional[datetime] = None        # capture time, with sub-seconds when recorded
    exposure: Optional[float] = None       # seconds
    fnumber: Optional[float] = None
    iso: Optional[float] = None
    focal_length: Optional[float] = None   # mm
    make: str = ""
    model: str = ""
    lens: str = ""


_EXIFTOOL_TAGS = ["EXIF:DateTimeOriginal", "EXIF:CreateDate", "EXIF:SubSecTimeOriginal", "EXIF:ExposureTime",
                  "EXIF:FNumber", "EXIF:ISO", "EXIF:FocalLength", "EXIF:Make", "EXIF:Model", "EXIF:LensModel"]


def _info_from_exiftool(meta: dict) -> FrameInfo:
    g = meta.get
    return FrameInfo(
        time=_with_subsec(_parse_exif_datetime(g("EXIF:DateTimeOriginal") or g("EXIF:CreateDate")),
                          g("EXIF:SubSecTimeOriginal")),
        exposure=_num(g("EXIF:ExposureTime")), fnumber=_num(g("EXIF:FNumber")), iso=_num(g("EXIF:ISO")),
        focal_length=_num(g("EXIF:FocalLength")), make=str(g("EXIF:Make") or "").strip(),
        model=str(g("EXIF:Model") or "").strip(), lens=str(g("EXIF:LensModel") or "").strip())


def _pillow_frame_info(path: str) -> FrameInfo:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(EXIF_IFD_POINTER) if exif else {}
            iso = sub.get(0x8827)
            return FrameInfo(
                time=_with_subsec(_parse_exif_datetime(sub.get(EXIF_DATETIME_ORIGINAL) or
                                                       sub.get(EXIF_DATETIME_DIGITIZED) or exif.get(EXIF_DATETIME)),
                                  sub.get(0x9291)),
                exposure=_num(sub.get(0x829A)), fnumber=_num(sub.get(0x829D)),
                iso=_num(iso[0] if isinstance(iso, tuple) else iso), focal_length=_num(sub.get(0x920A)),
                make=str(exif.get(EXIF_MAKE) or "").strip("\x00 "), model=str(exif.get(272) or "").strip("\x00 "),
                lens=str(sub.get(0xA434) or "").strip("\x00 "))
    except Exception:
        return FrameInfo()


def _pillow_capture_time(path: str) -> Optional[datetime]:
    return _pillow_frame_info(path).time


def read_frame_info(paths: Sequence[str], exiftool_path: Optional[str] = None,
                    workers: int = 8) -> Dict[str, FrameInfo]:
    """Return {path: FrameInfo} for every frame.

    Uses exiftool (one process for all files) when available, otherwise reads
    EXIF with Pillow in parallel.
    """
    info: Dict[str, FrameInfo] = {p: FrameInfo() for p in paths}
    if not paths:
        return info

    exe = find_exiftool(exiftool_path)
    if exe is not None:
        try:
            import exiftool

            lookup = {_norm(p): p for p in paths}
            with exiftool.ExifToolHelper(executable=exe) as et:
                for meta in et.get_tags(list(paths), tags=_EXIFTOOL_TAGS):
                    original = lookup.get(_norm(meta.get("SourceFile", "")))
                    if original is not None:
                        info[original] = _info_from_exiftool(meta)
            return info
        except Exception as exc:  # fall through to Pillow
            logger.warning("exiftool failed (%s); reading EXIF with Pillow instead.", exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, fi in zip(paths, pool.map(_pillow_frame_info, paths)):
            info[path] = fi
    return info


def read_capture_times(paths: Sequence[str], exiftool_path: Optional[str] = None,
                       workers: int = 8) -> Dict[str, Optional[datetime]]:
    """Return {path: capture datetime or None}."""
    return {p: fi.time for p, fi in read_frame_info(paths, exiftool_path, workers).items()}


def read_icc_profile(path) -> Optional[bytes]:
    """The embedded ICC colour profile (e.g. ProPhoto RGB from a Lightroom export), if any."""
    try:
        with Image.open(path) as img:
            return img.info.get("icc_profile") or None
    except Exception:
        return None


def sort_by_capture_time(paths: Sequence[str], times: Dict[str, Optional[datetime]]) -> List[str]:
    """Sort by EXIF capture time; ties and files without a date fall back to natural filename order."""
    return sorted(paths, key=lambda p: (times.get(p) is None, times.get(p) or datetime.min, natural_sort_key(p)))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _header_signature(path: str):
    """((width, height), bit depth, channel count) read from the file header only."""
    if path.lower().endswith(TIFF_EXTENSIONS):
        # Pillow can't describe 16-bit RGB TIFFs, so read the header with tifffile.
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            return (page.imagewidth, page.imagelength), f"{page.bitspersample}bit", page.samplesperpixel
    with Image.open(path) as img:
        return img.size, "8bit", len(img.getbands())


def validate_sequence(paths: Sequence[str], workers: int = 8) -> None:
    """Raise :class:`SequenceMismatchError` if any frame differs from the first in size, depth or channels."""
    if not paths:
        return
    reference = _header_signature(paths[0])

    def check(path):
        try:
            sig = _header_signature(path)
            if sig != reference:
                return path, f"size {sig[0]}, {sig[1]}, {sig[2]} channel(s); expected {reference[0]}, {reference[1]}, {reference[2]}"
        except Exception as exc:
            return path, f"unreadable: {exc}"
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        mismatches = [m for m in pool.map(check, paths) if m is not None]
    if mismatches:
        raise SequenceMismatchError(mismatches)
    logger.info("All %d frames match: %s, %s, %d channel(s).", len(paths), reference[0], reference[1], reference[2])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def as_hwc(img: np.ndarray) -> np.ndarray:
    """Ensure (H, W, C) layout and drop any alpha channel."""
    if img.ndim == 2:
        return img[:, :, None]
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def to_float(img: np.ndarray) -> np.ndarray:
    """Convert any integer or float image to float32 in 0-1 with shape (H, W, C)."""
    img = as_hwc(img)
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float32) / np.float32(np.iinfo(img.dtype).max)
    return img.astype(np.float32, copy=False)


# OpenCV decodes ~2.5x faster than Pillow here and releases the GIL, so it
# scales across threads. EXIF orientation is ignored (same as before).
_CV2_JPEG_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
_CV2_TIFF_FLAGS = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_IGNORE_ORIENTATION


def _cv2_decode(path: str) -> Optional[np.ndarray]:
    """Decode with OpenCV (BGR / gray), or None if OpenCV can't read it.
    Reads bytes with NumPy so non-ASCII Windows paths work."""
    flags = _CV2_TIFF_FLAGS if path.lower().endswith(TIFF_EXTENSIONS) else _CV2_JPEG_FLAGS
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except cv2.error:
        return None


def _fallback_decode(path: str) -> np.ndarray:
    """Slower decoders for files OpenCV can't handle. Returns RGB (H, W, C)."""
    if path.lower().endswith(TIFF_EXTENSIONS):
        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)  # separate colour planes -> interleaved
        return as_hwc(img)
    with Image.open(path) as im:
        return as_hwc(np.asarray(im.convert("RGB") if im.mode not in ("RGB", "L") else im))


_BGR_TO_RGB = {3: cv2.COLOR_BGR2RGB, 4: cv2.COLOR_BGRA2RGB}


def _tiff_into(path: str, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Read a TIFF with tifffile, straight into ``out`` when given. For uncompressed
    TIFFs this is a plain file read (~6x faster than OpenCV). Returns None if the
    file's layout doesn't fit ``out`` or tifffile can't decode it."""
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if out is None:
                img = page.asarray()
                if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
                    img = np.moveaxis(img, 0, -1)  # separate colour planes -> interleaved
                return as_hwc(img)
            if page.shape == out.shape and page.dtype == out.dtype:
                page.asarray(out=out)
                return out
            if page.ndim == 2 and out.shape[2] == 1 and page.dtype == out.dtype:
                page.asarray(out=out[:, :, 0])
                return out
    except Exception:
        pass
    return None


def load_raw(path) -> np.ndarray:
    """Load a JPEG or TIFF in its native dtype (uint8 / uint16), RGB, (H, W, C)."""
    path = str(path)
    if path.lower().endswith(TIFF_EXTENSIONS):
        img = _tiff_into(path)
        if img is not None:
            return img
    img = _cv2_decode(path)
    if img is None:
        return _fallback_decode(path)
    if img.ndim == 2:
        return img[:, :, None]
    return cv2.cvtColor(img, _BGR_TO_RGB[img.shape[2]])


def decode_into(path, out: np.ndarray) -> None:
    """Decode ``path`` straight into ``out`` (a preallocated (H, W, C) slot of the right dtype)."""
    path = str(path)
    if path.lower().endswith(TIFF_EXTENSIONS) and _tiff_into(path, out) is not None:
        return
    img = _cv2_decode(path)
    if img is not None and img.ndim == 3 and out.shape[2] == 3 and img.dtype == out.dtype:
        cv2.cvtColor(img, _BGR_TO_RGB[img.shape[2]], dst=out)  # channel swap + copy in one pass
        return
    out[...] = as_hwc(img) if img is not None and img.ndim == 2 else load_raw(path)


def load_image(path) -> np.ndarray:
    """Load a JPEG or TIFF as float32 0-1, (H, W, C).

    Scaling is based on the file's real dtype, so 8-bit TIFFs are scaled by
    255 and 16-bit TIFFs by 65535.
    """
    return to_float(load_raw(path))


def read_xmp(path) -> Optional[bytes]:
    """Return the embedded XMP packet (e.g. Lightroom develop settings), if any."""
    try:
        with Image.open(path) as img:
            xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp")
        if xmp is None and str(path).lower().endswith(TIFF_EXTENSIONS):
            with tifffile.TiffFile(path) as tif:
                tag = tif.pages[0].tags.get(TIFF_XMP_TAG)
                xmp = tag.value if tag is not None else None
        if isinstance(xmp, str):
            xmp = xmp.encode("utf-8")
        return xmp
    except Exception as exc:
        logger.debug("Could not read XMP from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def to_uint8(img: np.ndarray) -> np.ndarray:
    """float 0-1 or uint16 -> uint8 (rounded)."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return ((img.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    """float 0-1 or uint8 -> uint16."""
    if img.dtype == np.uint16:
        return img
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * np.uint16(257)
    return (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def _squeeze_channels(img: np.ndarray) -> np.ndarray:
    return img[:, :, 0] if img.ndim == 3 and img.shape[2] == 1 else img


def resolve_output_path(path, overwrite: str = "increment") -> Path:
    """Decide what to do when ``path`` exists.

    overwrite: "overwrite" - replace it; "increment" - use name_1, name_2...;
    "error" - raise FileExistsError.
    """
    path = Path(path)
    if not path.exists() or overwrite == "overwrite":
        return path
    if overwrite == "error":
        raise FileExistsError(f"Output already exists: {path}")
    if overwrite != "increment":
        raise ValueError(f"Unknown overwrite policy: {overwrite!r}")
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _jpeg_exif(meta, timestamp: Optional[datetime]) -> "Image.Exif":
    """EXIF block for a JPEG written by Pillow (the no-exiftool path)."""
    exif = Image.Exif()
    sub = exif.get_ifd(EXIF_IFD_POINTER)
    if timestamp is not None:
        stamp = timestamp.strftime("%Y:%m:%d %H:%M:%S")
        exif[EXIF_DATETIME] = stamp
        sub[EXIF_DATETIME_ORIGINAL] = stamp
        sub[EXIF_DATETIME_DIGITIZED] = stamp
    if meta is not None:
        from .metadata import software_name

        cam = meta.camera
        exif[270] = meta.description
        exif[305] = software_name()
        if cam.make:
            exif[EXIF_MAKE] = cam.make
        if cam.model:
            exif[272] = cam.model
        for tag, value in ((0x829A, cam.exposure), (0x829D, cam.fnumber), (0x920A, cam.focal_length)):
            if value:
                sub[tag] = float(value)
        if cam.iso:
            sub[0x8827] = int(cam.iso)
        if cam.lens:
            sub[0xA434] = cam.lens
        sub[0x9286] = b"ASCII\x00\x00\x00" + meta.comment_json.encode("ascii", "replace")
    return exif


def save_image(path, img: np.ndarray, xmp: Optional[bytes] = None,
               timestamp: Optional[datetime] = None, jpeg_quality: int = 100, metadata=None) -> Path:
    """Save a float 0-1 (or uint16) image: TIFF -> 16-bit, JPEG -> 8-bit.

    ``metadata`` (a :class:`timelapse_engine.metadata.OutputMetadata`) adds the
    title/caption, camera and lens EXIF, the source's ICC colour profile and
    Lightroom XMP; see :mod:`timelapse_engine.metadata`. ``timestamp`` (default:
    the metadata's capture time) becomes the capture date and file modified time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = _squeeze_channels(img)
    ext = path.suffix.lower()
    if timestamp is None and metadata is not None:
        timestamp = metadata.capture_time
    use_exiftool = False
    if metadata is not None:
        from .metadata import exiftool_available

        use_exiftool = exiftool_available()
        xmp = None if use_exiftool else (xmp or metadata.xmp)  # exiftool copies XMP itself
    icc = metadata.icc if metadata is not None else None

    if ext in TIFF_EXTENSIONS:
        data = to_uint16(img)
        extratags = [(TIFF_XMP_TAG, "B", len(xmp), xmp, True)] if xmp else []
        kwargs = {}
        if metadata is not None:
            from .metadata import software_name

            kwargs.update(description=metadata.description, software=software_name())
            for tag, value in ((EXIF_MAKE, metadata.camera.make), (272, metadata.camera.model)):
                if value:
                    extratags.append((tag, "s", 0, value, True))
        tifffile.imwrite(path, data, photometric="rgb" if data.ndim == 3 else "minisblack",
                         metadata=None,  # no tifffile JSON ({"shape": ...}) in ImageDescription
                         iccprofile=icc, datetime=timestamp, extratags=extratags, **kwargs)
    elif ext in (".jpg", ".jpeg"):
        pil = Image.fromarray(to_uint8(img))
        save_kwargs = {"icc_profile": icc} if icc else {}
        if xmp:
            save_kwargs["xmp"] = xmp
        pil.save(path, format="JPEG", quality=jpeg_quality, subsampling=0,
                 exif=_jpeg_exif(metadata, timestamp), **save_kwargs)
    else:
        raise ValueError(f"Unsupported output format: {ext} (use .tiff or .jpg)")

    if use_exiftool:
        from .metadata import write_with_exiftool

        write_with_exiftool(path, metadata, timestamp)
    if timestamp is not None:
        posix = timestamp.timestamp()
        os.utime(path, (posix, posix))
    return path
