"""Headless GUI test (Qt offscreen platform): load, preview, queue three jobs, run them.

Run with ``pytest`` or ``python tests/test_gui.py``. Skips when PySide6 isn't installed.
"""

import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PySide6.QtCore import QPoint, QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None

from timelapse_engine import files  # noqa: E402
from timelapse_engine.jobs import Job, load_queue, save_queue  # noqa: E402


def make_shoot(root: Path, n=24, h=120, w=160) -> Path:
    """<root>/Shoot/export with a bright square moving across a dim gradient."""
    folder = root / "Shoot" / "export"
    folder.mkdir(parents=True)
    base = np.linspace(40, 90, w, dtype=np.float32)[None, :, None].repeat(h, 0).repeat(3, 2)
    start = datetime(2025, 3, 1, 18, 0, 0)
    for i in range(n):
        img = base.copy()
        x = 5 + i * 5
        img[50:70, x:x + 15] = 250
        exif = Image.Exif()
        exif.get_ifd(files.EXIF_IFD_POINTER)[files.EXIF_DATETIME_ORIGINAL] = \
            (start + timedelta(seconds=10 * i)).strftime("%Y:%m:%d %H:%M:%S")
        Image.fromarray(img.astype(np.uint8)).save(folder / f"IMG_{i:04d}.jpg", quality=95, exif=exif)
    return folder


def wait_until(app, condition, timeout=120.0, what="condition"):
    deadline = time.time() + timeout
    while not condition():
        app.processEvents()
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {what}")
        time.sleep(0.01)


def test_gui_end_to_end(screenshot: str = None):
    if QApplication is None:
        print("  (PySide6 not installed, skipping)")
        return
    from timelapse_engine.gui.app import MainWindow, apply_dark_theme

    tmp = Path(tempfile.mkdtemp())
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp / "settings"))
    app = QApplication.instance() or QApplication([])
    apply_dark_theme(app)
    win = MainWindow()
    win.resize(1500, 950)
    win.show()
    try:
        src = make_shoot(tmp)
        out = tmp / "out"

        # Load the source: info, frame range, default output folder, first frame shown
        win.source_picker.set_path(str(src), emit=True)
        wait_until(app, lambda: win._source_info is not None, what="source load")
        assert win._source_info["frames"] == 24
        assert win.export_picker.path() == str(src.parent)  # ...\Shoot, not ...\Shoot\export
        assert win.name_edit.placeholderText() == "Shoot"
        wait_until(app, lambda: not win.view._item.pixmap().isNull(), what="first frame")

        # Scrub to a frame
        win.slider.setValue(10)
        wait_until(app, lambda: "IMG_0010" in win.frame_label.text(), what="frame 11")

        # Frame selection: set start/end at the playhead, type numbers, drag a handle
        assert win.timeline.selection() == (0, 23) and "all 24" in win.range_label.text()
        win.timeline.setValue(4)
        win.set_start_here()
        win.timeline.setValue(17)
        win.set_end_here()
        assert (win.first_spin.value(), win.last_spin.value()) == (5, 18)
        assert win.timeline.selection() == (4, 17)
        job = win.current_job()
        assert (job.first_frame, job.end_frame) == (4, 18)
        assert "5–18" in win.range_label.text() and "14 of 24" in win.range_label.text()
        assert "14 frames" in win.output_count.text()
        win.timeline.setValue(2)
        wait_until(app, lambda: "not processed" in win.frame_label.text(), what="outside-selection note")
        win.last_spin.setValue(3)              # end before start pulls the start back with it
        assert win.timeline.selection() == (2, 2)
        win.first_spin.setValue(10)            # start after end pushes the end forward
        assert win.timeline.selection() == (9, 9)
        win.timeline.setValue(1)
        win.set_end_here()                     # end before start resets the start to the first frame
        assert win.timeline.selection() == (0, 1)
        win._reset_range()
        tl = win.timeline
        y = int(tl._track().center().y())
        QTest.mousePress(tl, Qt.LeftButton, Qt.NoModifier, QPoint(int(tl._x(0)), y))
        QTest.mouseMove(tl, QPoint(int(tl._x(6)), y))
        QTest.mouseRelease(tl, Qt.LeftButton, Qt.NoModifier, QPoint(int(tl._x(6)), y))
        assert tl.selection() == (6, 23) and win.first_spin.value() == 7
        QTest.mouseClick(tl, Qt.LeftButton, Qt.NoModifier, QPoint(int(tl._x(12)), 4))  # top strip scrubs
        assert tl.value() == 12 and tl.selection() == (6, 23)
        win._reset_range()

        # Preview a still stack
        win.kind_combo.setCurrentIndex(win.kind_combo.findData("still"))
        win.process_combo.setCurrentIndex(win.process_combo.findData("maximum"))
        win.run_preview()
        wait_until(app, lambda: win._preview_shown, what="preview")
        assert "Preview" in win.view_status.text(), win.view_status.text()

        # Queue: still (TIFF), video (ProRes + 1080p preview), image sequence (JPEG) on frames 3-20
        win.export_picker.set_path(str(out))
        win.ext_combo.setCurrentIndex(win.ext_combo.findData("tiff"))
        win.add_to_queue()
        win.kind_combo.setCurrentIndex(win.kind_combo.findData("video"))
        win.process_combo.setCurrentIndex(win.process_combo.findData("average"))
        win.blend_spin.setValue(4)
        win.step_spin.setValue(2)
        win.master_combo.setCurrentIndex(win.master_combo.findData("prores"))
        win.p1080_check.setChecked(True)
        win.add_to_queue()
        win.kind_combo.setCurrentIndex(win.kind_combo.findData("frames"))
        win.process_combo.setCurrentIndex(win.process_combo.findData("motion_overlay"))
        win.ext_combo.setCurrentIndex(win.ext_combo.findData("jpg"))
        win.first_spin.setValue(3)
        win.last_spin.setValue(20)
        win.add_to_queue()
        assert len(win.jobs) == 3
        assert win.jobs[2].first_frame == 2 and win.jobs[2].end_frame == 20

        win.start_queue()
        wait_until(app, lambda: win._queue_thread is None, timeout=300, what="queue")
        statuses = [win._status(r) for r in range(3)]
        assert all(s.startswith("Done") for s in statuses), statuses
        assert (out / "Shoot_maximum.tiff").exists()
        assert (out / "Shoot_MASTER.mov").exists() and (out / "Shoot_1080p.mp4").exists()
        frames = sorted((out / "Shoot_motion_overlay_frames").glob("*.jpg"))
        assert len(frames) == len(range(0, 18, 2)), len(frames)

        # Double-clicking a job loads it back into the form
        win._on_queue_double_click(0, 0)
        assert win.kind_combo.currentData() == "still" and win.process_combo.currentData() == "maximum"

        # Queue files round-trip
        save_queue(tmp / "q.json", win.jobs)
        assert load_queue(tmp / "q.json") == win.jobs

        if screenshot:
            win.slider.setValue(12)
            app.processEvents()
            win.grab().save(screenshot)
    finally:
        win.close()
        app.processEvents()
        shutil.rmtree(tmp, ignore_errors=True)


def test_job_validation():
    errors = Job().validate()
    assert any("image folder or video" in e for e in errors) and any("output folder" in e for e in errors)
    bad = Job(source=".", export_dir=".", process="motion_overlay", params={"nope": 1})
    assert any("Bad parameters" in e for e in bad.validate())


if __name__ == "__main__":
    failed = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]:
        try:
            fn(*([sys.argv[1]] if name == "test_gui_end_to_end" and len(sys.argv) > 1 else []))
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
