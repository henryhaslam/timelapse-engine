"""Behaviour tests for the extra effects (entropy family, colour maps, time-slice, clean-up,
deflicker, interpolation). Run with pytest or ``python tests/test_effects.py``."""

import colorsys
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import timelapse_engine as te  # noqa: E402
from test_engine import ArraySource  # noqa: E402

H, W = 32, 40
DEVICES = ("cpu", "gpu") if te.HAS_CUPY else ("cpu",)


def run(name, frames, device="cpu", **params):
    return te.stack(ArraySource(frames), te.create_processor(name, **params), batch_size=7, device=device)


def noise_frames(n=40, seed=0):
    return np.random.default_rng(seed).random((n, H, W, 3), dtype=np.float32)


def _hue(rgb):
    return colorsys.rgb_to_hsv(*[float(c) for c in rgb])[0]


# ---------------------------------------------------------------------------
# Entropy family
# ---------------------------------------------------------------------------

def test_renyi_order_orders_entropy():
    f = noise_frames() ** 3  # skewed values so the orders differ
    low = run("temporal_entropy", f, renyi_order=0.5, noise_threshold=0, use_gpu=False)
    mid = run("temporal_entropy", f, renyi_order=1.0, noise_threshold=0, use_gpu=False)
    high = run("temporal_entropy", f, renyi_order=3.0, noise_threshold=0, use_gpu=False)
    assert (low >= mid - 1e-5).all() and (mid >= high - 1e-5).all()
    assert low.mean() > mid.mean() > high.mean()


def test_entropy_neighbourhood_spreads():
    f = np.full((30, H, W, 3), 0.5, np.float32)
    f[:, 16, 20] = np.random.default_rng(1).random((30, 3))
    alone = run("temporal_entropy", f, noise_threshold=0, use_gpu=False)
    spread = run("temporal_entropy", f, noise_threshold=0, neighbourhood=2, use_gpu=False)
    assert alone[16, 22].max() == 0 and spread[16, 22].max() > 0
    assert spread[16, 30].max() == 0


def test_entropy_gpu_matches_cpu_for_variants():
    if not te.HAS_CUPY:
        return
    f = noise_frames(25, 3)
    for name, kw in (("temporal_entropy", dict(renyi_order=0.7, neighbourhood=1)), ("change_entropy", {}),
                     ("colour_entropy", dict(renyi_order=2.0))):
        a = run(name, f, use_gpu=False, **kw)
        b = run(name, f, use_gpu=True, **kw)
        np.testing.assert_allclose(a, b, atol=1e-4, err_msg=name)


def test_change_entropy_turbulence_vs_drift():
    t = np.arange(40, dtype=np.float32)[:, None, None, None]
    drift = np.broadcast_to(0.2 + t * 0.015, (40, H, W, 3)).copy()          # steady change
    turbulent = noise_frames(40, 2)
    out_drift = run("change_entropy", drift, noise_threshold=0, use_gpu=False)
    out_turb = run("change_entropy", turbulent, noise_threshold=0, use_gpu=False)
    assert out_drift.max() < 0.05 and out_turb.mean() > 0.5


def test_colour_entropy_separates_brightness_from_colour():
    levels = np.random.default_rng(5).random(40).astype(np.float32)
    grey = np.broadcast_to(levels[:, None, None, None], (40, H, W, 3)).copy()  # brightness-only flicker
    out = run("colour_entropy", grey, noise_threshold=0, use_gpu=False)
    assert out[..., 1].mean() > 0.5                 # green = brightness entropy
    assert out[..., 0].max() < 0.05 and out[..., 2].max() < 0.05


def test_temporal_shape_skewness_sign():
    f = np.full((40, H, W, 3), 0.2, np.float32)
    f[::8, :, :20] = 0.9                            # left: mostly dark, rare bright -> positive skew
    f[:, :, 20:] = 0.8
    f[::8, :, 20:] = 0.1                            # right: mostly bright, rare dark -> negative skew
    for device in DEVICES:
        out = run("temporal_shape", f, device)
        assert out[5, 5, 0] > 0.6 and out[5, 30, 0] < 0.4, (device, out[5, 5], out[5, 30])
        kurt = run("temporal_shape", f, device, statistic="kurtosis")
        assert kurt[5, 5, 0] > 0.6, device          # rare spikes = heavy tails


# ---------------------------------------------------------------------------
# Motion and time
# ---------------------------------------------------------------------------

def test_time_of_peak_orders_hue():
    f = np.full((30, H, W, 3), 0.1, np.float32)
    f[3, :, :10] = 1.0       # left peaks early
    f[25, :, 30:] = 1.0      # right peaks late
    for device in DEVICES:
        out = run("time_of_peak", f, device, event="brightest")
        assert _hue(out[5, 2]) < _hue(out[5, 35]), device
        assert out[5, 2].max() > 0.9


def test_rhythm_map_period_sets_hue():
    t = np.arange(96, dtype=np.float32)
    fast = 0.5 + 0.3 * np.sin(2 * np.pi * t / 4)
    slow = 0.5 + 0.3 * np.sin(2 * np.pi * t / 24)
    f = np.full((96, H, W, 3), 0.5, np.float32)
    f[:, :, :12] = fast[:, None, None, None]
    f[:, :, 28:] = slow[:, None, None, None]
    for device in DEVICES:
        out = run("rhythm_map", f, device)
        assert out[5, 20].max() < 1e-3, device                  # static -> black
        assert _hue(out[5, 3]) < _hue(out[5, 35]), device        # short period -> lower hue
        assert out[5, 3].max() > 0.5 and out[5, 35].max() > 0.5


def test_eigen_colour_separates_patterns():
    t = np.linspace(0, 1, 50, dtype=np.float32)
    f = np.full((50, H, W, 3), 0.5, np.float32)
    f[:, :, :12] = (0.2 + 0.6 * t)[:, None, None, None]                         # ramp
    f[:, :, 28:] = (0.5 + 0.3 * np.sin(2 * np.pi * t * 3))[:, None, None, None]  # oscillation
    for device in DEVICES:
        out = run("eigen_colour", f, device)
        assert out[5, 20].max() < 1e-3, device
        a, b = out[5, 3], out[5, 35]
        assert np.abs(a / a.max() - b / b.max()).max() > 0.3, (device, a, b)


def test_memory_map_smooth_vs_noise():
    t = np.arange(120, dtype=np.float32)
    f = np.full((120, H, W, 3), 0.5, np.float32)
    f[:, :, :20] = (0.5 + 0.3 * np.sin(2 * np.pi * t / 40))[:, None, None, None]
    f[:, :, 20:] = noise_frames(120, 7)[:, :, 20:]
    for device in DEVICES:
        out = run("memory_map", f, device)
        assert out[5, 5, 0] > 0.8 and out[5, 30, 0] < 0.3, (device, out[5, 5], out[5, 30])


# ---------------------------------------------------------------------------
# Time slice
# ---------------------------------------------------------------------------

def ramp_frames(n=21):
    """Frame i is a flat grey of brightness i/(n-1)."""
    return np.broadcast_to(np.linspace(0, 1, n, dtype=np.float32)[:, None, None, None], (n, H, W, 3)).copy()


def test_time_slice_smooth_left_to_right():
    for device in DEVICES:
        out = run("time_slice", ramp_frames(), device)
        np.testing.assert_allclose(out[0, :, 0], np.linspace(0, 1, W), atol=1e-5, err_msg=device)
        rev = run("time_slice", ramp_frames(), device, reverse=True)
        np.testing.assert_allclose(rev[0, :, 0], np.linspace(1, 0, W), atol=1e-5)
        mir = run("time_slice", ramp_frames(), device, mirror=True)
        assert mir[0, 0, 0] < 0.05 and mir[0, W // 2, 0] > 0.9 and mir[0, -1, 0] < 0.05


def test_time_slice_bands_read_only_needed_frames():
    class Counting(ArraySource):
        reads = 0

        def read_raw(self, indices):
            Counting.reads += len(indices)
            return super().read_raw(indices)

    src = Counting(ramp_frames(21))
    src.probe()
    Counting.reads = 0
    out = te.stack(src, te.create_processor("time_slice", slices=4))
    assert Counting.reads == 4                                       # frames 0, 7, 13, 20
    np.testing.assert_allclose(sorted(set(np.round(out[0, :, 0], 4))), [0, 0.35, 0.65, 1], atol=1e-4)


def test_time_slice_patterns_and_feather():
    for pattern in ("top_to_bottom", "diagonal", "radial", "circular", "brightness"):
        out = run("time_slice", ramp_frames(), pattern=pattern)
        assert out.shape == (H, W, 3) and 0 <= out.min() and out.max() <= 1, pattern
    radial = run("time_slice", ramp_frames(), pattern="radial")
    assert radial[H // 2, W // 2, 0] < radial[0, 0, 0]               # centre first, corners last
    hard = run("time_slice", ramp_frames(), slices=3)
    soft = run("time_slice", ramp_frames(), slices=3, feather=0.8)
    assert len(np.unique(np.round(hard[0, :, 0], 4))) == 3
    assert len(np.unique(np.round(soft[0, :, 0], 4))) > 3            # edges pass through in-between frames


def test_time_slice_rolling_video_window():
    out = te.process_window(te.create_processor("time_slice"), ramp_frames(5))
    np.testing.assert_allclose(out[0, :, 0], np.linspace(0, 1, W), atol=1e-5)


# ---------------------------------------------------------------------------
# Clean-up and trails
# ---------------------------------------------------------------------------

def test_temporal_denoise_keeps_motion():
    rng = np.random.default_rng(9)
    clean = np.full((5, H, W, 3), 0.5, np.float32)
    noisy = clean + rng.normal(0, 0.03, clean.shape).astype(np.float32)
    noisy[2, 10:14, 10:14] = 1.0                                     # object only in the centre frame
    for device in DEVICES:
        out = run("temporal_denoise", noisy, device)
        assert np.abs(out - 0.5)[20:, 20:].mean() < np.abs(noisy[2] - 0.5)[20:, 20:].mean() * 0.8
        np.testing.assert_allclose(out[11, 11], 1.0)


def test_comet_trails_fade():
    f = np.zeros((10, H, W, 3), np.float32)
    for i in range(10):
        f[i, 5, 2 + 3 * i] = 1.0                                     # dot moving right
    out = run("comet_trails", f, trail_length=3)
    assert out[5, 2 + 27, 0] == 1.0 and 0 < out[5, 2, 0] < out[5, 2 + 15, 0] < 1.0


def test_clipped_average_rejects_outliers():
    rng = np.random.default_rng(3)
    f = (0.4 + rng.normal(0, 0.01, (30, H, W, 3))).astype(np.float32)
    f[7, 10:20, 10:20] = 1.0                                         # a plane flies through
    plain = run("average", f)
    clipped = run("clipped_average", f)
    assert plain[15, 15, 0] > 0.41 and abs(clipped[15, 15, 0] - 0.4) < 0.005


# ---------------------------------------------------------------------------
# Deflicker and interpolation
# ---------------------------------------------------------------------------

class MeanSink(te.outputs.Sink):
    def __init__(self):
        self.frames = []

    def write(self, frame, meta=None):
        self.frames.append(float(te.backend.asnumpy(frame).astype(np.float64).mean()))


def test_deflicker_gains_and_render():
    from timelapse_engine import adjust

    levels = 0.2 * (1 + 0.1 * np.tile([1, -1], 20))                  # +-10 % flicker
    gains = adjust.deflicker_gains(levels, window=9)
    np.testing.assert_allclose(levels * gains, np.median(levels), rtol=0.02)

    lin = 0.25 * (1 + 0.15 * np.tile([1, -1], 15)).astype(np.float32)
    f = np.broadcast_to((lin ** (1 / 2.2))[:, None, None, None], (30, H, W, 3)).copy()
    raw, fixed = MeanSink(), MeanSink()
    te.render_windows(ArraySource(f), "average", [raw], blend=1, step=1)
    te.render_windows(ArraySource(f), "average", [fixed], blend=1, step=1, deflicker=True, deflicker_window=7)
    assert np.std(fixed.frames) < 0.1 * np.std(raw.frames), (np.std(fixed.frames), np.std(raw.frames))


def test_interpolation_moves_objects_halfway():
    from timelapse_engine.interpolate import FrameInterpolator

    h, w = 96, 128
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w, 3), np.float32)
    b = np.zeros((h, w, 3), np.float32)
    a[((yy - 48) ** 2 + (xx - 40) ** 2) < 150] = 1.0
    b[((yy - 48) ** 2 + (xx - 56) ** 2) < 150] = 1.0
    mid = FrameInterpolator(2, flow_side=128).in_between(a, b)[0]
    cx = float((mid[..., 0] * xx).sum() / mid[..., 0].sum())
    assert 44 < cx < 52, cx                                          # roughly halfway (48)


def test_render_with_interpolation_count():
    sink = MeanSink()
    te.render_windows(ArraySource(noise_frames(10)), "average", [sink], blend=2, step=2, interpolate=3)
    assert len(sink.frames) == (5 - 1) * 3 + 1


def test_fast_small_jpeg_decode():
    from PIL import Image

    tmp = Path(tempfile.mkdtemp())
    try:
        Image.fromarray((np.random.default_rng(0).random((400, 600, 3)) * 255).astype(np.uint8)).save(tmp / "a.jpg")
        small = te.files.load_small(tmp / "a.jpg", 100)
        assert small.shape == (67, 100, 3) and small.dtype == np.uint8
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
