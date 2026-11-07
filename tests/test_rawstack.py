"""Tests for stacking raw sensor data into a DNG. Synthetic DNGs, no camera files needed
(one extra check runs on real G9 II files when they're on this machine).
Run with pytest or ``python tests/test_rawstack.py``. Skips when rawpy isn't installed."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timelapse_engine as te  # noqa: E402

try:
    import rawpy
except ImportError:  # pragma: no cover
    rawpy = None

if rawpy is not None:
    from timelapse_engine import rawstack
    from timelapse_engine.rawstack import RawInfo

H, W = 48, 64
BLACK, WHITE = 512, 16383
DEVICES = ("cpu", "gpu") if te.HAS_CUPY else ("cpu",)
REAL_RAWS = Path(r"C:\Users\henry\Videos\tl_temp\terifa\raw\2026.11.07_162035_0.48s_177f")


def info(**kw):
    base = dict(width=W, height=H, cfa=(0, 1, 1, 2), black=(BLACK,) * 4, white=WHITE,
                color_matrix=np.array([[0.83, -0.35, -0.06], [-0.43, 1.21, 0.25], [-0.09, 0.26, 0.60]]),
                as_shot_neutral=(0.5, 1.0, 0.7), make="TestCam", model="TC-RAW")
    base.update(kw)
    return RawInfo(**base)


def mosaic(seed, lo=BLACK + 100, hi=WHITE - 4000):
    return np.random.default_rng(seed).integers(lo, hi, (H, W)).astype(np.uint16)


def skip():
    if rawpy is None:
        print("  (rawpy not installed, skipping)")
        return True
    return False


def test_dng_round_trip():
    if skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        m = mosaic(0)
        rawstack.write_dng(tmp / "a.dng", m, info())
        with rawpy.imread(str(tmp / "a.dng")) as raw:
            assert np.array_equal(raw.raw_image_visible, m)
            desc = raw.color_desc.decode()
            assert "".join(desc[c] for c in raw.raw_colors_visible[:2, :2].ravel()) in ("RGGB",)
            assert list(raw.black_level_per_channel) == [BLACK] * 4 and raw.white_level == WHITE
            wb = raw.camera_whitebalance
            assert abs(wb[0] / wb[1] - 2.0) < 1e-3 and abs(wb[2] / wb[1] - 1 / 0.7) < 1e-3   # 1 / AsShotNeutral
            assert raw.postprocess(half_size=True).shape == (H // 2, W // 2, 3)
        back = rawstack.raw_info(tmp / "a.dng")
        assert back.cfa == (0, 1, 1, 2) and back.black == (BLACK,) * 4 and back.white == WHITE
        assert np.allclose(back.as_shot_neutral, (0.5, 1.0, 0.7), atol=1e-4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_raw_average_and_maximum_are_per_photosite():
    """An average on raw data is the plain mean of the sensor values (linear light), and a
    maximum is the per-photosite maximum, on CPU and GPU."""
    if skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        frames = [mosaic(i) for i in range(7)]
        (tmp / "seq").mkdir()
        for i, m in enumerate(frames):
            rawstack.write_dng(tmp / "seq" / f"F{i:03d}.dng", m, info())
        stack = np.stack(frames).astype(np.float64)
        for device in DEVICES:
            out = rawstack.stack_raw_to_dng(tmp / "seq", "average", tmp / "out", device=device, overwrite="overwrite")
            with rawpy.imread(str(out)) as raw:
                got = raw.raw_image_visible.astype(np.float64)
            assert np.abs(got - stack.mean(0)).max() <= 0.5 + 1e-6, device
            out = rawstack.stack_raw_to_dng(tmp / "seq", "maximum", tmp / "out", device=device, overwrite="overwrite")
            with rawpy.imread(str(out)) as raw:
                assert np.array_equal(raw.raw_image_visible, stack.max(0).astype(np.uint16)), device
        assert out.name == "seq_maximum.dng"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clipped_highlights_average_to_neutral():
    """A block where only green clipped would turn magenta when averaged; it is made neutral
    white first, so the average stays neutral after white balance."""
    if skip():
        return
    i = info()
    neutral_grey = np.array([[BLACK + 500, BLACK + 1000], [BLACK + 1000, BLACK + 700]], np.uint16)  # R G / G B
    frames = []
    for k in range(4):
        m = np.tile(neutral_grey, (H // 2, W // 2))
        if k < 2:                                  # glitter: green clipped; red and blue, balanced, above it
            m[10, 21] = m[11, 20] = WHITE           # the two greens of the RGGB block at (10, 20)
            m[10, 20], m[11, 21] = BLACK + 9000, BLACK + 12000
        frames.append(m)
    before = np.stack(frames).astype(np.float64).mean(0)
    for m in frames:
        rawstack.neutralise_clipped(m, i)
    after = np.stack(frames).astype(np.float64).mean(0)
    rng = WHITE - BLACK

    def balanced(block):   # R, G, B after white balance (divide by AsShotNeutral)
        r, g, b = block[0, 0] - BLACK, block[0, 1] - BLACK, block[1, 1] - BLACK
        return np.array([r / 0.5, g, b / 0.7]) / rng

    rb, ra = balanced(before[10:12, 20:22]), balanced(after[10:12, 20:22])
    assert rb[0] / rb[1] > 1.05                     # unfixed: red too strong against clipped green
    assert np.allclose(ra / ra[1], 1.0, atol=1e-3)  # fixed: neutral
    assert np.array_equal(after[0:2, 0:2], before[0:2, 0:2])   # unclipped blocks untouched


def test_only_per_pixel_stacks_run_on_raw():
    if skip():
        return
    for process in ("temporal_entropy", "time_of_peak", "motion_overlay"):
        try:
            rawstack.stack_raw_to_dng("nowhere", process)
            raise AssertionError(f"{process} accepted")
        except ValueError as exc:
            assert "average" in str(exc)


def test_real_raw_round_trip():
    """On a machine with the terifa card: one G9 II RW2 written as a DNG renders identically."""
    if skip() or not REAL_RAWS.is_dir():
        print("  (no real raw files here, skipping)")
        return
    src = sorted(REAL_RAWS.glob("*.RW2"))[0]
    tmp = Path(tempfile.mkdtemp())
    try:
        dng = rawstack.write_dng(tmp / "one.dng", rawstack.read_mosaic(src), rawstack.raw_info(src))
        kw = dict(half_size=True, use_camera_wb=True, no_auto_bright=True, output_bps=16)
        with rawpy.imread(str(src)) as a, rawpy.imread(str(dng)) as b:
            assert np.array_equal(a.raw_image_visible, b.raw_image_visible)
            diff = np.abs(a.postprocess(**kw).astype(np.int32) - b.postprocess(**kw).astype(np.int32))
            assert diff.mean() < 1.0, diff.mean()
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
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
