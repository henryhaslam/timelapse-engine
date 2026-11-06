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

    def read_many(self, indices):
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

        def read_many(self, indices):
            CountingSource.reads += len(indices)
            return super().read_many(indices)

    src = CountingSource(frames)
    windows = list(te.iter_windows(src, blend=8, step=3))
    assert len(windows) == len(range(0, 20, 3))
    assert CountingSource.reads == 20  # every frame decoded exactly once
    start, w = windows[1]
    np.testing.assert_array_equal(w, frames[3:11])


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
