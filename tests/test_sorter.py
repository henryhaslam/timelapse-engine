"""Behaviour tests for the raw sorter: finding sequences from capture-time spacing, and
moving/copying/undoing safely. Run with pytest or ``python tests/test_sorter.py``."""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timelapse_engine import sorter  # noqa: E402
from timelapse_engine.files import find_exiftool  # noqa: E402
from timelapse_engine.sorter import Shot  # noqa: E402

T0 = datetime(2026, 6, 7, 12, 0, 0, 123000)
EXIFTOOL_MODES = (False, True) if find_exiftool() else (False,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(start, interval, n, jitter=0.0, seed=0):
    """Offsets (s) of ``n`` shots ``interval`` apart from ``start``, with random jitter."""
    rng = np.random.default_rng(seed)
    return [start + i * interval + (rng.uniform(-jitter, jitter) if jitter else 0.0) for i in range(n)]


def make_shots(offsets, camera="Panasonic DC-G9M2", precise=True, counters=None, prefix="P"):
    shots = []
    for i, off in enumerate(offsets):
        t = T0 + timedelta(seconds=off)
        if not precise:
            t = t.replace(microsecond=0)
        shots.append(Shot(path=f"C:\\card\\{prefix}{i:07d}.RW2", time=t, precise=precise, camera=camera,
                          counter=counters[i] if counters else None))
    return shots


def sizes(shots, **kw):
    groups, singles = sorter.find_sequences(shots, **kw)
    return [g.frames for g in groups], singles.frames


def counters_for(*lengths):
    """Camera shot counters that restart at 1 for each run (Panasonic SequenceNumber)."""
    return [i + 1 for n in lengths for i in range(n)]


def write_jpeg(path: Path, t: datetime, make="TestCam", model="TC-1", value=128):
    exif = Image.Exif()
    exif[271], exif[272] = make, model
    ifd = exif.get_ifd(0x8769)
    ifd[36867] = t.strftime("%Y:%m:%d %H:%M:%S")
    ifd[0x9291] = f"{t.microsecond // 1000:03d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (value, value, value)).save(path, quality=90, exif=exif)


def make_card(root: Path):
    """A folder like a card dump: two sequences (2 s and 0.5 s), a RAW+JPEG+xmp shot in each,
    two single shots and a video clip. Returns the expected {stem: sequence index or None}."""
    expected = {}
    k = 1000
    for i, off in enumerate(run(0, 2.0, 12) + run(100, 0.5, 9) + [300.0, 420.0]):
        t = T0 + timedelta(seconds=off)
        stem = f"P{k:07d}"
        k += 1
        write_jpeg(root / f"{stem}.JPG", t, value=40 + i * 5)
        expected[stem] = 0 if i < 12 else 1 if i < 21 else None
        if i in (3, 15):   # RAW+JPEG pair with sidecars (the dummy raw has no readable EXIF)
            (root / f"{stem}.RW2").write_bytes(b"not really a raw file")
            (root / f"{stem}.xmp").write_text("<x:xmpmeta/>", encoding="utf-8")
            (root / f"{stem}.RW2.xmp").write_text("<x:xmpmeta/>", encoding="utf-8")
    (root / "P1009999.MP4").write_bytes(b"video")
    return expected


def tree(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Finding sequences
# ---------------------------------------------------------------------------

def test_sample_card_layout():
    """The pattern measured on the G9 II sample card: a 10 fps burst, then 0.48 s and 1 s runs
    separated by 17-86 s gaps, including a 7-frame false start."""
    lengths = [156, 14, 177, 199, 7, 85]
    offsets = (run(0, 0.1, 156, jitter=0.02, seed=1) + run(72.3, 0.48, 14, jitter=0.008, seed=2) +
               run(95.0, 0.48, 177, jitter=0.008, seed=3) + run(219.6, 1.0, 199, jitter=0.05, seed=4) +
               run(503.0, 1.0, 7, jitter=0.05, seed=5) + run(538.4, 1.0, 85, jitter=0.05, seed=6))
    shots = make_shots(offsets, counters=counters_for(*lengths))
    assert sizes(shots) == (lengths, 0)
    assert sizes(shots, use_counter=False) == (lengths, 0)      # the spacing alone is enough here
    groups, _ = sorter.find_sequences(shots)
    for g, interval in zip(groups, [0.1, 0.48, 0.48, 1.0, 1.0, 1.0]):
        assert abs(g.interval - interval) < 0.02, (g.interval, interval)


def test_same_spacing_after_a_gap_is_a_new_sequence():
    assert sizes(make_shots(run(0, 2, 50) + run(120, 2, 40))) == ([50, 40], 0)


def test_new_spacing_without_a_gap_splits_at_the_change():
    groups, singles = sorter.find_sequences(make_shots(run(0, 1, 30) + run(34, 5, 30)))
    assert [g.frames for g in groups] == [30, 30] and singles.frames == 0
    assert round(groups[0].interval) == 1 and round(groups[1].interval) == 5


def test_whole_second_timestamps_still_group():
    """Without sub-seconds a 2.3 s interval reads as 2 or 3 s; that's one sequence."""
    shots = make_shots(run(0, 2.3, 100) + run(400, 2.3, 60), precise=False)
    assert sizes(shots) == ([100, 60], 0)


def test_ramped_interval_stays_together():
    offsets, t, step = [], 0.0, 2.0
    for _ in range(150):
        offsets.append(t)
        t += step
        step *= 1.016          # 2 s slowly ramping to ~20 s (day to night)
    assert sizes(make_shots(offsets)) == ([150], 0)


def test_single_shots_are_left_out():
    offsets = [-300.0, -170.0] + run(0, 3, 40) + [400.0] + run(600, 0.5, 30) + [900.0, 1300.0]
    groups, singles = sorter.find_sequences(make_shots(offsets))
    assert [g.frames for g in groups] == [40, 30]
    assert singles.frames == 5 and singles.singles


def test_cameras_are_kept_apart():
    a = make_shots(run(0, 2, 60), camera="Panasonic DC-G9M2 #A1", prefix="A")
    b = make_shots(run(1, 3, 40), camera="Canon EOS R5 #B2", prefix="B")
    groups, singles = sorter.find_sequences(a + b)
    assert sorted(g.frames for g in groups) == [40, 60] and singles.frames == 0
    for g in groups:
        assert len({s.camera for s in g.shots}) == 1


def test_counter_restart_splits_a_lookalike_gap():
    """10 s spacing, restarted 11 s later: the gap fits the spacing, the counter doesn't."""
    offsets = run(0, 10, 30) + run(301, 10, 20)
    assert sizes(make_shots(offsets, counters=counters_for(30, 20))) == ([30, 20], 0)
    assert sizes(make_shots(offsets, counters=counters_for(30, 20)), use_counter=False) == ([50], 0)


def test_counter_ignored_when_it_does_not_count():
    """A camera that writes the same counter on every shot must not split anything."""
    assert sizes(make_shots(run(0, 2, 40), counters=[1] * 40)) == ([40], 0)


def test_min_frames():
    shots = make_shots(run(0, 1, 4) + run(100, 1, 20))
    assert sizes(shots, min_frames=5) == ([20], 4)
    assert sizes(shots, min_frames=3) == ([4, 20], 0)


def test_names_are_unique_and_pattern_checked():
    a = make_shots(run(0, 2, 10), camera="Cam #1", prefix="A")
    b = make_shots(run(0, 2, 10), camera="Cam #2", prefix="B")
    groups, _ = sorter.find_sequences(a + b)
    sorter.name_groups(groups)
    assert groups[0].name == "2026.06.07_120000_2s_10f" and groups[1].name == "2026.06.07_120000_2s_10f_2"
    sorter.name_groups(groups, "{number}-{camera}")
    assert [g.name for g in groups] == ["1-Cam", "2-Cam"]
    try:
        sorter.name_groups(groups, "{nope}")
        raise AssertionError("bad pattern accepted")
    except ValueError as exc:
        assert "{date}" in str(exc)


def test_formatting():
    assert [sorter.fmt_interval(x) for x in (0.1, 0.48, 1.0, 2.5, 30, 120)] == \
        ["0.1s", "0.48s", "1s", "2.5s", "30s", "120s"]
    assert sorter.fmt_duration(85) == "1 min 25 s" and sorter.fmt_duration(7300) == "2 h 01 min"


# ---------------------------------------------------------------------------
# Files: scanning, moving, copying, undo
# ---------------------------------------------------------------------------

def test_scan_groups_companions_and_ignores_other_files():
    tmp = Path(tempfile.mkdtemp())
    try:
        make_card(tmp)
        shots, ignored = sorter.collect_shots(tmp)
        pair = next(s for s in shots if s.stem == "P0001003")
        assert pair.name == "P0001003.RW2"                       # the raw is the main file
        assert sorted(os.path.basename(f) for f in pair.companions) == \
            ["P0001003.JPG", "P0001003.RW2.xmp", "P0001003.xmp"]
        assert [os.path.basename(f) for f in ignored] == ["P1009999.MP4"]
        assert len(shots) == 23
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_move_and_undo():
    for use_exiftool in EXIFTOOL_MODES:
        tmp = Path(tempfile.mkdtemp())
        try:
            card = tmp / "card"
            expected = make_card(card)
            before = tree(card)
            result = sorter.analyse(card, use_exiftool=use_exiftool)
            assert [g.frames for g in result.groups] == [12, 9], (use_exiftool, result.summary())
            assert result.singles.frames == 2 and len(result.ignored) == 1
            assert not any(s.from_file_time for s in result.shots)    # the dummy raw took its JPEG's time
            assert all(s.precise for s in result.shots)
            names = [g.name for g in result.groups]
            assert names == ["2026.06.07_120000_2s_12f", "2026.06.07_120140_0.5s_9f"], names

            log = sorter.sort_files(result.groups, card)
            for stem, seq in expected.items():
                where = card / names[seq] / "raw" if seq is not None else card
                assert (where / f"{stem}.JPG").exists(), (stem, where)
            for name in names:                                         # empty export\ ready for Lightroom
                export = card / name / "export"
                assert export.is_dir() and not any(export.iterdir())
            pair = card / names[0] / "raw" / "P0001003"
            for suffix in (".RW2", ".JPG", ".xmp", ".RW2.xmp"):        # the whole shot moved together
                assert Path(f"{pair}{suffix}").exists(), suffix
            assert (card / "P1009999.MP4").exists()                   # other files left alone
            assert sorter.find_last_log(card) == log

            restored, problems = sorter.undo(log)
            assert problems == [] and restored == 12 + 9 + 2 * 3
            assert tree(card) == sorted(before + [log.name])          # back as it was, plus the log
            assert not [p for p in card.iterdir() if p.is_dir()]      # sequence, raw\ and export\ folders gone
            assert json.loads(log.read_text())["undone"]
            assert sorter.find_last_log(card) is None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_name_clash_across_card_folders_keeps_pairs():
    """DCIM\\100_PANA and 101_PANA can both hold P1000001: the second shot gets _2 on every file."""
    tmp = Path(tempfile.mkdtemp())
    try:
        card = tmp / "DCIM"
        for i, off in enumerate(run(0, 2, 10)):
            folder = card / ("100_PANA" if i < 5 else "101_PANA")
            stem = f"P100{i % 5:04d}"
            write_jpeg(folder / f"{stem}.JPG", T0 + timedelta(seconds=off))
            (folder / f"{stem}.xmp").write_text(f"shot {i}", encoding="utf-8")
        result = sorter.analyse(card, subfolders=True, use_exiftool=False)
        assert [g.frames for g in result.groups] == [10]
        sorter.sort_files(result.groups, tmp / "sorted")
        out = tmp / "sorted" / result.groups[0].name / "raw"
        assert len(list(out.glob("*.JPG"))) == 10
        for i in range(5):
            stem = f"P100{i:04d}"
            assert (out / f"{stem}.xmp").read_text() == f"shot {i}"
            assert (out / f"{stem}_2.JPG").exists() and (out / f"{stem}_2.xmp").read_text() == f"shot {i + 5}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_never_overwrites():
    tmp = Path(tempfile.mkdtemp())
    try:
        for i, off in enumerate(run(0, 2, 6)):
            write_jpeg(tmp / f"IMG_{i:04d}.JPG", T0 + timedelta(seconds=off))
        result = sorter.analyse(tmp, use_exiftool=False)
        folder = tmp / result.groups[0].name / "raw"
        folder.mkdir(parents=True)
        (folder / "IMG_0002.JPG").write_bytes(b"someone else's file")
        sorter.sort_files(result.groups, tmp)
        assert (folder / "IMG_0002.JPG").read_bytes() == b"someone else's file"
        assert (folder / "IMG_0002_2.JPG").exists() and len(list(folder.glob("*.JPG"))) == 7
        try:
            sorter._move_file(str(folder / "IMG_0001.JPG"), str(folder / "IMG_0002.JPG"))
            raise AssertionError("overwrote a file")
        except FileExistsError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_flat_layout_and_single_shots():
    """subfolders=False puts files straight into each sequence folder; single shots are
    always flat (no raw\\ or export\\), whatever the layout."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for layout in (False, True):
            card = tmp / f"card_{layout}"
            make_card(card)
            result = sorter.analyse(card, use_exiftool=False)
            sorter.sort_files(list(result.groups) + [result.singles], card, subfolders=layout)
            first = card / result.groups[0].name
            assert len(list((first / "raw" if layout else first).glob("*.JPG"))) == 12
            assert (first / "export").is_dir() == layout
            singles = card / sorter.SINGLES_NAME
            assert len(list(singles.glob("*.JPG"))) == 2
            assert not (singles / "raw").exists() and not (singles / "export").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_copy_then_undo_removes_only_the_copies():
    tmp = Path(tempfile.mkdtemp())
    try:
        card = tmp / "card"
        make_card(card)
        before = tree(card)
        result = sorter.analyse(card, use_exiftool=False)
        log = sorter.sort_files(result.groups, tmp / "sorted", mode="copy")
        assert tree(card) == before                                   # originals untouched
        assert len(list((tmp / "sorted").rglob("*.JPG"))) == 21
        restored, problems = sorter.undo(log)
        assert restored == 27 and problems == []
        assert [p.name for p in (tmp / "sorted").iterdir()] == [log.name]   # copies and folders gone
        assert tree(card) == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_file_time_fallback():
    """Raw files nothing can read (no exiftool) are dated by their file modified time."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for i, off in enumerate(run(0, 5, 8) + run(200, 5, 8)):
            p = tmp / f"P{i:07d}.RW2"
            p.write_bytes(b"raw")
            ts = (T0 + timedelta(seconds=off)).timestamp()
            os.utime(p, (ts, ts))
        result = sorter.analyse(tmp, use_exiftool=False)
        assert [g.frames for g in result.groups] == [8, 8]
        assert all(s.from_file_time and not s.precise for s in result.shots)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cancel_part_way_can_be_undone():
    from timelapse_engine.progress import Cancelled, CancelToken

    tmp = Path(tempfile.mkdtemp())
    try:
        card = tmp / "card"
        make_card(card)
        before = tree(card)
        result = sorter.analyse(card, use_exiftool=False)
        token = CancelToken()

        def progress(done, total, message):
            if done == 7:
                token.cancel()

        try:
            sorter.sort_files(result.groups, card, progress=progress, cancel=token)
            raise AssertionError("not cancelled")
        except Cancelled:
            pass
        log = sorter.find_last_log(card)
        assert len(json.loads(log.read_text())["moves"]) == 7
        sorter.undo(log)
        assert tree(card) == sorted(before + [log.name])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_dry_run_moves_nothing():
    tmp = Path(tempfile.mkdtemp())
    try:
        make_card(tmp)
        before = tree(tmp)
        assert sorter.main([str(tmp)]) == 0
        assert tree(tmp) == before
        assert sorter.main([str(tmp), "--move"]) == 0
        assert len([p for p in tmp.iterdir() if p.is_dir()]) == 2
        assert sorter.main([str(tmp), "--undo"]) == 0
        assert [p for p in tmp.iterdir() if p.is_dir()] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preview_from_jpeg():
    tmp = Path(tempfile.mkdtemp())
    try:
        write_jpeg(tmp / "a.JPG", T0, value=200)
        img = sorter.preview_image(tmp / "a.JPG", max_side=16)
        assert img.dtype == np.uint8 and img.shape == (12, 16, 3) and abs(int(img.mean()) - 200) < 4
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
