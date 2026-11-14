"""Tests on small synthetic sequences.

Run with ``pytest`` or, without pytest installed, ``python tests/test_engine.py``.
"""

import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timelapse_engine as te  # noqa: E402
from timelapse_engine import files  # noqa: E402

H, W = 48, 64


def make_frames(n=25, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, H, W, 3), dtype=np.float32)


class ArraySource(te.FrameSource):
    """In-memory source for testing processors without touching disk."""

    def __init__(self, frames):
        self.frames = frames
        self.name = "array"

    def __len__(self):
        return len(self.frames)

    def read_raw(self, indices):
        return [self.frames[i] for i in indices]


def write_jpeg_sequence(folder, n=12, start=datetime(2024, 5, 1, 12, 0, 0)):
    """JPEGs whose filenames are in REVERSE capture order, to prove EXIF sorting works."""
    rng = np.random.default_rng(1)
    for i in range(n):
        img = Image.fromarray((rng.random((H, W, 3)) * 255).astype(np.uint8))
        exif = Image.Exif()
        sub = exif.get_ifd(files.EXIF_IFD_POINTER)
        sub[files.EXIF_DATETIME_ORIGINAL] = (start + timedelta(minutes=i)).strftime("%Y:%m:%d %H:%M:%S")
        img.save(Path(folder) / f"frame_{n - i:03d}.jpg", quality=95, exif=exif)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------

def test_average_not_double_counting_first_batch():
    frames = make_frames(25)
    result = te.stack(ArraySource(frames), te.create_processor("average"), batch_size=10)
    np.testing.assert_allclose(result, frames.mean(axis=0), atol=1e-6)


def test_max_min():
    frames = make_frames(25)
    np.testing.assert_array_equal(te.stack(ArraySource(frames), te.create_processor("maximum"), 7), frames.max(0))
    np.testing.assert_array_equal(te.stack(ArraySource(frames), te.create_processor("minimum"), 7), frames.min(0))


def test_median_and_mode_are_not_black():
    frames = make_frames(25)
    med = te.stack(ArraySource(frames), te.create_processor("median"), 10)
    np.testing.assert_allclose(med, np.median(frames, axis=0), atol=2 / 65535)
    constant = np.full((5, H, W, 3), 0.5, np.float32)
    mode = te.stack(ArraySource(constant), te.create_processor("mode"), 2)
    np.testing.assert_allclose(mode, 128 / 255, atol=1e-6)


def test_gradient_first_frame_has_no_false_jump():
    frames = np.repeat(make_frames(1), 10, axis=0)  # static scene
    out = te.stack(ArraySource(frames), te.create_processor("temporal_gradient"), 3)
    assert out.max() == 0.0


def test_entropy_cpu_matches_gpu_and_bounds():
    frames = make_frames(20)
    cpu = te.stack(ArraySource(frames), te.create_processor("temporal_entropy", use_gpu=False, noise_threshold=0), 6)
    assert cpu.shape == (H, W, 3) and 0 <= cpu.min() and cpu.max() <= 1
    static = te.stack(ArraySource(np.repeat(frames[:1], 8, 0)),
                      te.create_processor("temporal_entropy", use_gpu=False), 3)
    assert static.max() == 0.0
    if te.HAS_CUPY:
        gpu = te.stack(ArraySource(frames), te.create_processor("temporal_entropy", use_gpu=True, noise_threshold=0), 6)
        np.testing.assert_allclose(gpu, cpu, atol=1e-5)


def test_processor_reusable_across_windows():
    frames = make_frames(12)
    for name in te.PROCESSORS:
        proc = te.create_processor(name, **({"use_gpu": False} if name == "temporal_entropy" else {}))
        a = te.process_window(proc, frames[:6])
        te.process_window(proc, frames[6:])
        b = te.process_window(proc, frames[:6])
        np.testing.assert_allclose(a, b, atol=1e-6, err_msg=name)
        assert a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1, name


def test_motion_overlay_pastes_moving_object():
    frames = np.full((10, H, W, 3), 0.2, np.float32)
    frames[5, 10:20, 10:20] = 1.0  # bright object in one frame only
    out = te.stack(ArraySource(frames), te.create_processor("motion_overlay", motion_threshold=20), 4)
    np.testing.assert_allclose(out[15, 15], 1.0)
    np.testing.assert_allclose(out[40, 50], 0.2, atol=1e-6)


def test_unknown_processor_and_bad_params():
    for bad in (lambda: te.create_processor("percentile"), lambda: te.create_processor("average", foo=1)):
        try:
            bad()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("expected an error")


def test_cancel():
    token = te.CancelToken()
    token.cancel()
    try:
        te.stack(ArraySource(make_frames(5)), te.create_processor("average"), 2, cancel=token)
    except te.Cancelled:
        return
    raise AssertionError("expected Cancelled")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def test_windows_reuse_frames():
    frames = make_frames(20)

    class CountingSource(ArraySource):
        reads = 0

        def read_raw(self, indices):
            CountingSource.reads += len(indices)
            return super().read_raw(indices)

    src = CountingSource(frames)
    src.probe()
    CountingSource.reads = 0
    windows = list(te.iter_windows(src, blend=8, step=3))
    assert len(windows) == len(range(0, 20, 3))
    assert CountingSource.reads == 20  # every frame decoded exactly once
    start, w = windows[1]
    np.testing.assert_array_equal(w, frames[3:11])

    # step larger than blend skips frames entirely
    CountingSource.reads = 0
    windows = list(te.iter_windows(src, blend=2, step=5))
    assert CountingSource.reads == 8 and len(windows) == 4


def test_windows_gpu_matches_cpu():
    frames = make_frames(14)
    outs = {}
    for device in ("cpu", "gpu") if te.HAS_CUPY else ("cpu",):
        tmp = Path(tempfile.mkdtemp())
        try:
            paths = te.render_frames(ArraySource(frames), "average", tmp, "w", blend=4, step=3, ext="tiff",
                                     device=device)
            outs[device] = [tifffile.imread(p) for p in paths]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    expected = [files.to_uint16(frames[s:s + 4].mean(axis=0)) for s in range(0, 14, 3)]
    for device, imgs in outs.items():
        assert len(imgs) == len(expected)
        for got, exp in zip(imgs, expected):
            assert np.abs(got.astype(int) - exp.astype(int)).max() <= 1, device


class CaptureSink(te.outputs.Sink):
    def __init__(self):
        self.frames = []

    def write(self, frame, meta=None):
        self.frames.append(te.backend.asnumpy(frame).copy())


def test_running_sums_match_recompute():
    rng = np.random.default_rng(11)
    frames16 = rng.integers(0, 65535, (23, 16, 20, 3), dtype=np.uint16)  # raw 16-bit, like TIFFs
    for name in ("average", "temporal_std"):
        for blend, step in ((5, 1), (6, 4), (3, 7)):
            for device in ("cpu", "gpu") if te.HAS_CUPY else ("cpu",):
                sliding, recompute = CaptureSink(), CaptureSink()
                te.render_windows(ArraySource(frames16), name, [sliding], blend, step, device=device)
                proc = te.create_processor(name)
                proc.sliding = False
                te.render_windows(ArraySource(frames16), proc, [recompute], blend, step, device=device)
                assert len(sliding.frames) == len(recompute.frames) == len(range(0, 23, step))
                for a, b in zip(sliding.frames, recompute.frames):
                    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1, (name, blend, step, device)


# ---------------------------------------------------------------------------
# GPU parity and vectorised maths
# ---------------------------------------------------------------------------

def test_gpu_matches_cpu_for_every_processor():
    if not te.HAS_CUPY:
        print("  (CuPy not available, skipping)")
        return
    frames = make_frames(13, seed=3)
    for name, cls in te.PROCESSORS.items():
        if not cls.gpu or name == "temporal_entropy":
            continue
        cpu = te.stack(ArraySource(frames), te.create_processor(name), batch_size=5, device="cpu")
        gpu = te.stack(ArraySource(frames), te.create_processor(name), batch_size=5, device="gpu")
        np.testing.assert_allclose(gpu, cpu, atol=2e-5, err_msg=name)


def test_binary_close_matches_opencv():
    import cv2
    from timelapse_engine.processors import binary_close

    rng = np.random.default_rng(5)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for density in (0.05, 0.3, 0.7):
        mask = rng.random((40, 57)) < density
        expected = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
        np.testing.assert_array_equal(binary_close(mask, 2), expected)


def _motion_overlay_reference(frames, threshold, priority):
    """Original frame-by-frame algorithm, for checking the vectorised version."""
    import cv2

    bg = frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    canvas, max_dev = bg.copy(), np.zeros(bg.shape[:2], np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for f in frames:
        dev = np.abs(f - bg).max(axis=2)
        mask = cv2.morphologyEx((dev > threshold).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
        if priority:
            mask &= dev > max_dev
            max_dev[mask] = dev[mask]
        canvas[mask] = f[mask]
    return canvas


def test_motion_overlay_vectorised_matches_loop():
    frames = make_frames(17, seed=7) * 0.3
    frames[3, 5:15, 5:15] += 0.5
    frames[9, 10:30, 20:40] += 0.6
    frames[14, 8:12, 8:50] += 0.4
    for priority in (True, False):
        expected = _motion_overlay_reference(frames, 40 / 255, priority)
        for device in ("cpu", "gpu") if te.HAS_CUPY else ("cpu",):
            proc = te.create_processor("motion_overlay", motion_threshold=40, intensity_priority=priority)
            got = te.stack(ArraySource(frames), proc, batch_size=4, device=device)
            np.testing.assert_allclose(got, np.clip(expected, 0, 1), atol=1e-6, err_msg=f"{priority} {device}")


def _smooth_rgb16(h=96, w=128):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([xx / w, yy / h, 0.5 + 0.4 * np.sin(xx / 9) * np.cos(yy / 7)], axis=-1)
    return (np.clip(img, 0, 1) * 65535).astype(np.uint16)


def test_yuv_packing_matches_ffmpeg_bt709():
    import subprocess

    from timelapse_engine import colour

    if shutil.which("ffmpeg") is None:
        print("  (ffmpeg not on PATH, skipping)")
        return
    # Exact BT.709 limited-range levels for greys (FFmpeg's own rgb48 path is ~0.3% off: white = 943)
    greys = np.array([0, 16384, 32768, 49152, 65535], np.uint16)
    grey_img = np.repeat(np.repeat(greys, 2)[None, :, None], 2, axis=0).repeat(3, axis=2)
    y = colour.pack(grey_img, "yuv422p10le")[0][: grey_img.shape[1]: 2]
    np.testing.assert_array_equal(y, [64, 283, 502, 721, 940])

    img = _smooth_rgb16()
    h, w = img.shape[:2]
    # FFmpeg as an independent reference, with tolerance for its rounding error
    for pix_fmt, tol in (("yuv422p10le", 4), ("yuv420p10le", 4), ("p010le", 4 << 6), ("yuv420p", 2)):
        ref = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-i", "-",
             "-vf", f"scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format={pix_fmt}",
             "-f", "rawvideo", "-"], input=img.tobytes(), capture_output=True, check=True).stdout
        dtype = np.uint8 if pix_fmt == "yuv420p" else np.uint16
        ref = np.frombuffer(ref, dtype)
        for device in ("cpu", "gpu") if te.HAS_CUPY else ("cpu",):
            frame = img if device == "cpu" else te.backend.cp.asarray(img)
            got, size = colour.pack(frame, pix_fmt)
            got = te.backend.asnumpy(got)
            assert size == (h, w) and got.shape == ref.shape and got.dtype == ref.dtype, pix_fmt
            luma = h * w
            assert np.abs(got[:luma].astype(int) - ref[:luma].astype(int)).max() <= tol, (pix_fmt, device, "Y")
            assert np.abs(got[luma:].astype(int) - ref[luma:].astype(int)).max() <= tol, (pix_fmt, device, "C")


def test_area_resize_matches_opencv():
    import cv2

    from timelapse_engine import colour

    img = _smooth_rgb16(97, 130).astype(np.float32) / 65535
    for h, w in ((27, 36), (54, 72), (97, 130)):
        expected = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(colour.resize_area(img, h, w), expected, atol=2e-3)
    assert colour.output_size(3888, 5184, (1080, None)) == (1080, 1440)
    assert colour.output_size(3888, 5184, (None, 3840)) == (2880, 3840)
    assert colour.output_size(1080, 1920, (2160, None)) == (1080, 1920)  # never upscales


def test_memory_plan():
    from timelapse_engine.engine import MemoryPlan

    shape = (3888, 5184, 3)
    assert MemoryPlan(te.create_processor("average"), shape, np.uint8, False, 100).batch == 16
    assert MemoryPlan(te.create_processor("average"), shape, np.uint16, False, 100).batch == 8
    try:
        MemoryPlan(te.create_processor("median"), shape, np.uint16, False, 100_000)
    except MemoryError as exc:
        assert "Process fewer frames" in str(exc)
    else:
        raise AssertionError("expected MemoryError")
    if te.HAS_CUPY:
        small = MemoryPlan(te.create_processor("average"), shape, np.uint16, True, 100, window=4)
        huge = MemoryPlan(te.create_processor("average"), shape, np.uint16, True, 100, window=500)
        assert small.cache_on_gpu and not huge.cache_on_gpu and huge.chunk >= 1


def test_opencv_decode_matches_pillow():
    tmp = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(2)
        img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(tmp / "a.jpg", quality=95)
        tifffile.imwrite(tmp / "b.tif", (rng.random((H, W, 3)) * 65535).astype(np.uint16), photometric="rgb")
        with Image.open(tmp / "a.jpg") as im:
            pil = np.asarray(im)
        cv = te.load_raw(tmp / "a.jpg")
        assert cv.dtype == np.uint8 and cv.shape == (H, W, 3)
        assert np.abs(cv.astype(int) - pil.astype(int)).max() <= 2
        np.testing.assert_array_equal(te.load_raw(tmp / "b.tif"), tifffile.imread(tmp / "b.tif"))
        slot = np.empty((H, W, 3), np.uint16)
        files.decode_into(tmp / "b.tif", slot)
        np.testing.assert_array_equal(slot, tifffile.imread(tmp / "b.tif"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def test_files_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    try:
        src_dir, out_dir = tmp / "in", tmp / "out"
        src_dir.mkdir()
        write_jpeg_sequence(src_dir)

        seq = te.ImageSequence(src_dir)
        names = [Path(p).name for p in seq.paths]
        assert names[0] == "frame_012.jpg" and names[-1] == "frame_001.jpg", names  # EXIF order, not name order

        out = te.stack_to_file(seq, "average", out_dir, "test", ext="tiff")
        assert out.name == "test_average.tiff"
        assert tifffile.imread(out).dtype == np.uint16
        expected = datetime(2024, 5, 1, 12, 11) + timedelta(minutes=1)
        assert abs(out.stat().st_mtime - expected.timestamp()) < 2

        out2 = te.stack_to_file(seq, "average", out_dir, "test", ext="jpg")
        out3 = te.stack_to_file(seq, "average", out_dir, "test", ext="jpg")
        assert out3.name == "test_average_1.jpg"  # no overwrite, no input() prompt
        assert files._pillow_capture_time(str(out2)) == expected

        # 8-bit TIFFs must be scaled by 255, not 65535
        tifffile.imwrite(tmp / "eight.tif", np.full((4, 4, 3), 255, np.uint8), photometric="rgb")
        assert te.load_image(tmp / "eight.tif").max() == 1.0

        # mismatched sizes are reported
        Image.new("RGB", (10, 10)).save(src_dir / "odd.jpg")
        try:
            te.ImageSequence(src_dir, sort="name")
        except te.SequenceMismatchError as exc:
            assert "odd.jpg" in str(exc)
        else:
            raise AssertionError("expected SequenceMismatchError")
        seq.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_camera_sequence(folder, n=12, start=datetime(2025, 7, 1, 21, 0, 0), interval=2.5):
    """JPEGs with full camera EXIF: make/model/lens, 1/250 s, f/2.8, ISO 800, 24 mm, sub-second times, sRGB ICC."""
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    rng = np.random.default_rng(4)
    for i in range(n):
        t = start + timedelta(seconds=interval * i)
        exif = Image.Exif()
        exif[271], exif[272] = "Panasonic", "DC-GX9"
        sub = exif.get_ifd(files.EXIF_IFD_POINTER)
        sub[files.EXIF_DATETIME_ORIGINAL] = t.strftime("%Y:%m:%d %H:%M:%S")
        sub[0x9291] = f"{t.microsecond // 1000:03d}"           # SubSecTimeOriginal
        sub[0x829A], sub[0x829D], sub[0x920A] = 1 / 250, 2.8, 24.0
        sub[0x8827] = 800
        sub[0xA434] = "LUMIX G 20/F1.7 II"
        img = Image.fromarray((rng.random((H, W, 3)) * 255).astype(np.uint8))
        img.save(Path(folder) / f"P{i:04d}.jpg", quality=95, exif=exif, icc_profile=icc)
    return icc


def _read_meta(path):
    """(ImageDescription, Make, Model, exif sub-IFD, ICC, DateTimeOriginal) read back with Pillow."""
    with Image.open(path) as im:
        exif = im.getexif()
        sub = exif.get_ifd(files.EXIF_IFD_POINTER)
        return (exif.get(270), exif.get(271), exif.get(272), sub, im.info.get("icc_profile"),
                files._pillow_capture_time(str(path)))


def _check_outputs(tmp, icc, use_exiftool):
    import json

    seq = te.ImageSequence(tmp / "Shoot" / "export")
    assert seq.name == "Shoot"
    info = seq.frame_info(0)
    assert info.exposure == 1 / 250 and info.iso == 800 and info.lens == "LUMIX G 20/F1.7 II"
    out = tmp / ("out_et" if use_exiftool else "out_plain")
    for ext in ("tiff", "jpg"):
        path = te.stack_to_file(te.SubsetSource(seq, range(2, 10)), "maximum", out, ext=ext)
        desc, make, model, sub, got_icc, when = _read_meta(path)
        assert desc.startswith("Maximum stack of 8 frames | source frames 3-10 of 12 from Shoot"), desc
        assert "1/250 s exposures every 2.5 s" in desc and "total exposure 1/31 s" in desc, desc
        assert "Panasonic DC-GX9" in desc and "24 mm f/2.8 ISO 800" in desc, desc
        assert '"shape"' not in desc
        assert (make, model) == ("Panasonic", "DC-GX9")
        assert got_icc == icc, "colour profile not copied"
        assert when == datetime(2025, 7, 1, 21, 0, 22) + timedelta(minutes=1), when  # last frame + 1 min
        if ext == "jpg" or use_exiftool:
            assert sub.get(0xA434) == "LUMIX G 20/F1.7 II" and float(sub.get(0x829A)) == 1 / 250
            comment = sub.get(0x9286)
            comment = comment.decode("ascii", "ignore") if isinstance(comment, bytes) else str(comment)
            facts = json.loads(comment[comment.index("{"):])
            assert facts["frames"] == 8 and facts["interval_s"] == 2.5 and facts["process"] == "maximum"
    frames = te.render_frames(seq, "average", out / "seq", blend=4, step=4, ext="jpg")
    desc, *_, when = _read_meta(frames[1])
    assert desc.startswith("Average blend of 4 frames | source frames 5-8 of 12"), desc
    assert when == datetime(2025, 7, 1, 21, 0, 17), when  # last frame of that window (seconds precision)
    seq.close()


def test_output_metadata():
    from timelapse_engine import metadata

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "Shoot" / "export").mkdir(parents=True)
        icc = write_camera_sequence(tmp / "Shoot" / "export")
        if metadata.exiftool_available():
            _check_outputs(tmp, icc, use_exiftool=True)
        original = files.find_exiftool
        files.find_exiftool = lambda explicit=None: None  # the no-exiftool fallback
        try:
            _check_outputs(tmp, icc, use_exiftool=False)
        finally:
            files.find_exiftool = original
    finally:
        metadata.close_exiftool()
        shutil.rmtree(tmp, ignore_errors=True)


def test_video_metadata():
    import subprocess

    if shutil.which("ffmpeg") is None:
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "Shoot" / "export").mkdir(parents=True)
        write_camera_sequence(tmp / "Shoot" / "export")
        outs = te.render_video(tmp / "Shoot" / "export", "average", tmp, blend=4, step=2, master=None,
                               preview_1080p=True)
        tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(outs[0])],
                              capture_output=True, text=True, check=True).stdout
        assert "Average rolling blend of 4 frames per output frame, step 2" in tags, tags
        assert "1/250 s exposures every 2.5 s" in tags and "creation_time" in tags
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_video_and_frames():
    if shutil.which("ffmpeg") is None:
        print("  (ffmpeg not on PATH, skipping video test)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        frames = make_frames(12)
        outs = te.render_video(ArraySource(frames), "average", tmp, "clip", blend=4, step=2,
                               master="prores", preview_1080p=True)
        assert all(p.exists() and p.stat().st_size > 0 for p in outs), outs
        stills = te.render_frames(ArraySource(frames), "maximum", tmp / "stills", "clip", blend=4, step=4, ext="jpg")
        assert len(stills) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_video_sizes_and_formats():
    """Each ticked size/format makes its own file at the right size: 1080p is short side 1080,
    4K never upscales past the frames, and the names match the ones the app always used."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("  (ffmpeg/ffprobe not on PATH, skipping)")
        return
    import subprocess

    def dims(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,codec_name", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
        codec, w, h = out.split(",")
        return codec, int(w), int(h)

    tmp = Path(tempfile.mkdtemp())
    try:
        frames = np.random.default_rng(3).random((4, 1200, 1600, 3), dtype=np.float32)   # 4:3 landscape
        outs = te.render_video(ArraySource(frames), "average", tmp, "clip", blend=2, step=2,
                               videos=[("full", "h264"), ("4k", "h264"), {"size": "1080p", "format": "prores"}])
        assert [p.name for p in outs] == ["clip_full.mp4", "clip_4K.mp4", "clip_1080p.mov"], outs
        assert dims(outs[0]) == ("h264", 1600, 1200)
        assert dims(outs[1]) == ("h264", 1600, 1200)                   # 4K is bigger than the frames: no upscale
        assert dims(outs[2]) == ("prores", 1440, 1080)
        portrait = te.render_video(ArraySource(frames.transpose(0, 2, 1, 3).copy()), "average", tmp, "tall",
                                   blend=2, step=2, videos=[("1080p", "h264")])
        assert dims(portrait[0])[1:] == (1080, 1440)                   # short side 1080 when upright too
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_job_videos_and_old_queue_files():
    from timelapse_engine.jobs import Job

    old = Job.from_dict({"kind": "video", "master": "hevc_nvenc", "preview_1080p": True, "preview_4k": False})
    assert old.videos == [{"size": "full", "format": "hevc_nvenc"}, {"size": "1080p", "format": "h264"}]
    assert "full HEVC + 1080p H.264" in old.describe()
    assert Job.from_dict({"kind": "video", "master": None, "preview_1080p": True}).videos == \
        [{"size": "1080p", "format": "h264"}]
    assert Job().videos == [{"size": "full", "format": "prores"}]       # default: the ProRes master as before
    assert Job.from_dict(Job(videos=[{"size": "4k", "format": "h264"}]).to_dict()).videos == \
        [{"size": "4k", "format": "h264"}]
    nothing = Job(kind="video", source=".", export_dir=".", videos=[])
    assert any("Tick at least one video size" in e for e in nothing.validate())
    bad = Job(kind="video", source=".", export_dir=".", videos=[{"size": "8k", "format": "h264"}])
    assert any("Unknown video output" in e for e in bad.validate())


def test_ffmpeg_message_explains_failures():
    """FFmpeg's reason is kept, without its [codec @ 0x...] prefixes and the generic follow-on lines."""
    from timelapse_engine.outputs import ffmpeg_message

    nvenc = [  # what FFmpeg 9.0 printed on a laptop with NVIDIA driver 566.07
        "[hevc_nvenc @ 000001e25ac45b00] Driver does not support the required nvenc API version. Required: 13.1 Found: 12.2",
        "[hevc_nvenc @ 000001e25ac45b00] The minimum required Nvidia driver for nvenc is 610.00 or newer",
        "[vost#0:0/hevc_nvenc @ 000001e25ac45080] [enc:hevc_nvenc @ 000001e25ac43d80] Error while opening encoder - "
        "maybe incorrect parameters such as bit_rate, rate, width or height.",
        "[vf#0:0 @ 000001e25ac609c0] Task finished with error code: -40 (Function not implemented)",
    ]
    msg = ffmpeg_message(nvenc, -40, te.outputs.hevc_nvenc_master("x.mov").codec_args)
    assert "610.00 or newer" in msg and "update the NVIDIA driver" in msg, msg
    assert "@" not in msg and "maybe incorrect parameters" not in msg and "Task finished" not in msg, msg
    assert ffmpeg_message([], 4294967256) == "FFmpeg exited with code 4294967256."


def test_video_sink_stops_before_rendering_when_encoder_cannot_start():
    if shutil.which("ffmpeg") is None:
        print("  (ffmpeg not on PATH, skipping)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        bad = te.VideoOutput(tmp / "clip.mp4", ["-c:v", "no_such_encoder"], "yuv420p")
        try:
            te.FFmpegVideoSink([bad], fps=24)
            raise AssertionError("sink accepted an encoder FFmpeg doesn't have")
        except RuntimeError as exc:
            assert "clip.mp4" in str(exc) and "no_such_encoder" in str(exc), exc
        assert not any(tmp.iterdir())                                 # nothing created
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_console_progress_on_any_encoding():
    """Output redirected on Windows uses the ANSI code page (cp1252), which has no block character:
    the bar falls back to '#', and characters it can't write in the message become '?'."""
    import io

    for encoding, fill, tick in (("utf-8", "█", "✓"), ("cp1252", "#", "?"), ("ascii", "#", "?")):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding=encoding)
        bar = te.ConsoleProgress(bar_length=10, stream=stream)
        bar(5, 10, "Reading ✓")
        bar(10, 10, "Reading ✓")
        text = raw.getvalue().decode(encoding)
        assert fill * 10 in text and f"Reading {tick} [" in text and "(10/10)" in text, (encoding, text)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
