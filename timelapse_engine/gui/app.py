"""Timelapse Engine desktop app.

Run with ``python -m timelapse_engine.gui`` (or ``timelapse-gui`` once installed).

Left: source, effect and output settings. Right: the frame viewer / live
preview with a frame scrubber, and below it the job queue and log.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from .. import __version__, backend
from ..jobs import (KINDS, MASTERS, Job, SourceCache, default_export, is_video, load_queue, render_preview,
                    save_queue)
from ..processors import CATEGORY_ORDER, PROCESSORS
from ..progress import CancelToken, logger
from .widgets import ImageView, ParamForm, PathPicker, Timeline
from .worker import QtLogHandler, QueueRunner, TaskRunner

DISPLAY_NAMES = {
    "average": "Average (long exposure)",
    "maximum": "Maximum (light trails)",
    "minimum": "Minimum (darkest)",
    "median": "Median (remove passers-by)",
    "mode": "Mode (most common value)",
    "temporal_contrast": "Temporal contrast",
    "temporal_std": "Temporal variation (std dev)",
    "temporal_gradient": "Temporal gradient (change trails)",
    "temporal_hue_motion": "Motion coloured by time",
    "temporal_entropy": "Temporal entropy",
    "optical_flow_accum": "Optical flow",
    "motion_overlay": "Motion overlay (clean plate + movers)",
    "change_entropy": "Change entropy (turbulence vs drift)",
    "colour_entropy": "Colour entropy (colour vs brightness)",
    "temporal_shape": "Temporal shape (skewness / kurtosis)",
    "time_of_peak": "Time of peak (when it happened)",
    "rhythm_map": "Rhythm map (repeating changes)",
    "eigen_colour": "Eigen-colour (patterns of change)",
    "memory_map": "Memory map (how long pixels stay similar)",
    "time_slice": "Time slice",
    "temporal_denoise": "Temporal denoise (video)",
    "comet_trails": "Comet trails",
    "clipped_average": "Clipped average (no planes/people)",
}
VIDEO_FILTER = "Video files (*.mp4 *.mov *.mkv *.avi *.m4v *.mts *.mxf);;All files (*)"
PREVIEW_SIZES = {"Fast (960 px)": 960, "Normal (1280 px)": 1280, "Large (1920 px)": 1920}
DISPLAY_MAX_SIDE = 1920

STATUS_WAITING, STATUS_RUNNING, STATUS_DONE = "Waiting", "Running", "Done"
STATUS_FAILED, STATUS_CANCELLED = "Failed", "Cancelled"


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _gpu_status() -> str:
    if not backend.HAS_CUPY:
        return "GPU: not available (CPU mode)"
    try:
        props = backend.cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        return f"GPU: {name.replace('NVIDIA GeForce ', '')} · {backend.gpu_free_bytes() / 1e9:.1f} GB free"
    except Exception:
        return "GPU: CuPy"


def open_in_explorer(path: str) -> None:
    p = Path(path)
    if sys.platform == "win32":
        if p.is_file():
            subprocess.Popen(["explorer", "/select,", str(p)])
        else:
            os.startfile(str(p if p.exists() else p.parent))  # noqa: S606 - local folder
    else:
        subprocess.Popen(["xdg-open", str(p.parent if p.is_file() else p)])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Timelapse Engine")
        # %APPDATA%\timelapse-engine\gui.ini (tests redirect it with QSettings.setPath)
        self.settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "timelapse-engine", "gui")
        self.sources = SourceCache()
        self.tasks = TaskRunner(self, max_threads=3)
        self.jobs: List[Job] = []
        self._source_info: Optional[dict] = None
        self._auto_export_dir = ""
        self._frame_request = 0
        self._preview_request = 0
        self._preview_cancel: Optional[CancelToken] = None
        self._preview_shown = False
        self._queue_thread: Optional[QThread] = None
        self._queue_runner: Optional[QueueRunner] = None
        self._job_started_at: Dict[int, float] = {}
        self._loading_form = False

        self._build_ui()
        self._connect_logging()
        self._frame_timer = QTimer(self, singleShot=True, interval=120, timeout=self._fetch_frame)
        self._preview_timer = QTimer(self, singleShot=True, interval=400, timeout=self.run_preview)
        self._restore_settings()
        self.statusBar().showMessage("Ready")
        self._gpu_label.setText(_gpu_status())

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1100])
        self._main_splitter = splitter
        self.setCentralWidget(splitter)

        self._gpu_label = QLabel()
        self.statusBar().addPermanentWidget(self._gpu_label)

        file_menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (("Open image folder…", QKeySequence.Open, self._pick_source_folder),
                                     ("Open video…", None, self._pick_source_video),
                                     (None, None, None),
                                     ("Load queue…", None, self.load_queue_file),
                                     ("Save queue…", QKeySequence.Save, self.save_queue_file),
                                     (None, None, None),
                                     ("Quit", QKeySequence.Quit, self.close)):
            if text is None:
                file_menu.addSeparator()
                continue
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            file_menu.addAction(act)
        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, "Timelapse Engine",
            f"Timelapse Engine {__version__}\n\nGPU stacking and rolling-window blending for timelapse "
            f"sequences and video.\n\n{_gpu_status()}"))
        help_menu.addAction(about)

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)

        # Source
        src_box = QGroupBox("Source")
        src_form = QFormLayout(src_box)
        self.source_picker = PathPicker("Image folder or video file", "Folder…", "Video…", VIDEO_FILTER)
        self.source_picker.changed.connect(self._on_source_changed)
        src_form.addRow(self.source_picker)
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Capture time (EXIF)", "exif")
        self.sort_combo.addItem("File name", "name")
        self.sort_combo.currentIndexChanged.connect(lambda: self._on_source_changed(self.source_picker.path()))
        src_form.addRow("Order", self.sort_combo)
        self.source_info = QLabel("No source loaded")
        self.source_info.setWordWrap(True)
        self.source_info.setStyleSheet("color: palette(mid);")
        src_form.addRow(self.source_info)
        lay.addWidget(src_box)

        # Effect
        fx_box = QGroupBox("Effect")
        fx_form = QFormLayout(fx_box)
        self.kind_combo = QComboBox()
        for key, label in KINDS.items():
            self.kind_combo.addItem(label, key)
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        fx_form.addRow("Create", self.kind_combo)
        self.process_combo = QComboBox()
        self.process_combo.setMaxVisibleItems(30)
        categories = CATEGORY_ORDER + sorted({c.category for c in PROCESSORS.values()} - set(CATEGORY_ORDER))
        for category in categories:
            names = [n for n, c in PROCESSORS.items() if c.category == category]
            if not names:
                continue
            self.process_combo.addItem(category.upper())        # group heading (not selectable)
            heading = self.process_combo.model().item(self.process_combo.count() - 1)
            heading.setEnabled(False)
            heading.setSelectable(False)
            for name in names:
                self.process_combo.addItem("    " + DISPLAY_NAMES.get(name, name), name)
        self.process_combo.setCurrentIndex(self.process_combo.findData("average"))
        self.process_combo.currentIndexChanged.connect(self._on_process_changed)
        fx_form.addRow("Effect", self.process_combo)
        self.process_desc = QLabel()
        self.process_desc.setWordWrap(True)
        self.process_desc.setStyleSheet("color: palette(mid);")
        fx_form.addRow(self.process_desc)
        self.param_form = ParamForm()
        self.param_form.changed.connect(self._on_settings_changed)
        fx_form.addRow(self.param_form)
        self.blend_spin = QSpinBox(minimum=1, maximum=10000, value=8)
        self.blend_spin.setToolTip("Frames blended into each output frame.")
        self.step_spin = QSpinBox(minimum=1, maximum=10000, value=1)
        self.step_spin.setToolTip("Source frames to advance between output frames.")
        self.fps_spin = QSpinBox(minimum=1, maximum=240, value=24)
        for w in (self.blend_spin, self.step_spin, self.fps_spin):
            w.valueChanged.connect(self._on_settings_changed)
        self._blend_label, self._step_label, self._fps_label = QLabel("Blend"), QLabel("Step"), QLabel("Frame rate")
        fx_form.addRow(self._blend_label, self.blend_spin)
        fx_form.addRow(self._step_label, self.step_spin)
        fx_form.addRow(self._fps_label, self.fps_spin)
        self.interp_combo = QComboBox()
        self.interp_combo.addItem("Off", 1)
        for k in (2, 3, 4):
            self.interp_combo.addItem(f"{k}x (motion-compensated in-betweens)", k)
        self.interp_combo.setToolTip("Adds in-between frames estimated from the motion: smoother clouds, or slow "
                                     "motion at the same frame rate.")
        self.interp_combo.currentIndexChanged.connect(self._on_settings_changed)
        self._interp_label = QLabel("Smooth motion")
        fx_form.addRow(self._interp_label, self.interp_combo)
        flicker_row = QHBoxLayout()
        self.deflicker_check = QCheckBox("Deflicker")
        self.deflicker_check.setToolTip("Evens out frame-to-frame brightness flicker before the effect is applied.")
        self.deflicker_spin = QSpinBox(minimum=3, maximum=301, value=15)
        self.deflicker_spin.setToolTip("Smoothing width in frames. Larger removes slower flicker but follows real "
                                       "changes (like sunset) less closely.")
        self.deflicker_spin.setSuffix(" frames")
        self.deflicker_check.toggled.connect(self.deflicker_spin.setEnabled)
        self.deflicker_spin.setEnabled(False)
        for w in (self.deflicker_check, self.deflicker_spin):
            flicker_row.addWidget(w)
        flicker_row.addStretch(1)
        fx_form.addRow(flicker_row)
        self.output_count = QLabel()
        self.output_count.setStyleSheet("color: palette(mid);")
        fx_form.addRow(self.output_count)
        lay.addWidget(fx_box)

        # Output
        out_box = QGroupBox("Output")
        out_form = QFormLayout(out_box)
        self.export_picker = PathPicker("Output folder", "Folder…")
        out_form.addRow(self.export_picker)
        self.name_edit = QLineEdit()
        out_form.addRow("Name", self.name_edit)
        self.ext_combo = QComboBox()
        self.ext_combo.addItem("16-bit TIFF", "tiff")
        self.ext_combo.addItem("JPEG", "jpg")
        self._ext_label = QLabel("Format")
        out_form.addRow(self._ext_label, self.ext_combo)
        self.master_combo = QComboBox()
        for key, label in MASTERS.items():
            self.master_combo.addItem(label, key)
        self._master_label = QLabel("Master")
        out_form.addRow(self._master_label, self.master_combo)
        prev_row = QHBoxLayout()
        self.p1080_check = QCheckBox("1080p preview")
        self.p4k_check = QCheckBox("4K preview")
        prev_row.addWidget(self.p1080_check)
        prev_row.addWidget(self.p4k_check)
        prev_row.addStretch(1)
        self._previews_label = QLabel("Previews")
        out_form.addRow(self._previews_label, prev_row)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU when possible)", "auto")
        self.device_combo.addItem("GPU", "gpu")
        self.device_combo.addItem("CPU", "cpu")
        out_form.addRow("Compute", self.device_combo)
        self.overwrite_combo = QComboBox()
        self.overwrite_combo.addItem("Add a number", "increment")
        self.overwrite_combo.addItem("Overwrite", "overwrite")
        self.overwrite_combo.addItem("Stop with an error", "error")
        out_form.addRow("If the file exists", self.overwrite_combo)
        lay.addWidget(out_box)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setToolTip("Quick low-resolution render of the current settings (Ctrl+P).")
        self.preview_btn.setShortcut("Ctrl+P")
        self.preview_btn.clicked.connect(self.run_preview)
        self.add_btn = QPushButton("Add to queue")
        self.add_btn.setDefault(True)
        self.add_btn.clicked.connect(self.add_to_queue)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(self.add_btn)
        lay.addLayout(buttons)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(380)
        scroll.setFrameShape(QScrollArea.NoFrame)
        return scroll

    def _build_right_panel(self) -> QWidget:
        right = QSplitter(Qt.Vertical)

        viewer = QWidget()
        vlay = QVBoxLayout(viewer)
        vlay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        self.frame_btn = QToolButton(text="Frame", checkable=True, checked=True)
        self.result_btn = QToolButton(text="Preview", checkable=True)
        for i, b in enumerate((self.frame_btn, self.result_btn)):
            self.view_group.addButton(b, i)
            top.addWidget(b)
        self.view_group.idClicked.connect(self._on_view_mode)
        top.addSpacing(12)
        self.auto_preview = QCheckBox("Update automatically")
        self.auto_preview.setToolTip("Re-render the preview when settings or the current frame change.")
        top.addWidget(self.auto_preview)
        self.preview_size = QComboBox()
        for label, px in PREVIEW_SIZES.items():
            self.preview_size.addItem(label, px)
        self.preview_size.setCurrentIndex(1)
        top.addWidget(self.preview_size)
        top.addStretch(1)
        self.view_status = QLabel()
        self.view_status.setStyleSheet("color: palette(mid);")
        top.addWidget(self.view_status)
        vlay.addLayout(top)
        self.view = ImageView()
        vlay.addWidget(self.view, 1)
        # Timeline: playhead (current frame) + start/end handles (frames to process)
        scrub = QHBoxLayout()
        self.timeline = self.slider = Timeline()
        self.timeline.valueChanged.connect(self._on_frame_changed)
        self.timeline.rangeChanged.connect(self._on_timeline_range)
        self.frame_spin = QSpinBox(minimum=1, maximum=1)
        self.frame_spin.setToolTip("Current frame")
        self.frame_spin.valueChanged.connect(lambda v: self.timeline.setValue(v - 1))
        self.frame_label = QLabel()
        self.frame_label.setMinimumWidth(240)
        scrub.addWidget(self.timeline, 1)
        scrub.addWidget(QLabel("Frame"))
        scrub.addWidget(self.frame_spin)
        scrub.addWidget(self.frame_label)
        vlay.addLayout(scrub)

        trim = QHBoxLayout()
        self.first_spin = QSpinBox(minimum=1, maximum=1)
        self.first_spin.setToolTip("First frame to process (type a number or drag the left blue handle)")
        self.last_spin = QSpinBox(minimum=1, maximum=1)
        self.last_spin.setToolTip("Last frame to process (type a number or drag the right blue handle)")
        for spin in (self.first_spin, self.last_spin, self.frame_spin):
            spin.setMinimumWidth(72)  # room for 5-digit frame numbers
        self.first_spin.valueChanged.connect(self._on_first_spin)
        self.last_spin.valueChanged.connect(self._on_last_spin)
        set_start = QPushButton("Set start here  (I)")
        set_start.setToolTip("Start processing at the current frame")
        set_start.clicked.connect(self.set_start_here)
        set_end = QPushButton("Set end here  (O)")
        set_end.setToolTip("Stop processing at the current frame")
        set_end.clicked.connect(self.set_end_here)
        use_all = QPushButton("Use all frames")
        use_all.clicked.connect(self._reset_range)
        self.range_label = QLabel()
        self.range_label.setStyleSheet("color: palette(mid);")
        for w in (QLabel("Start"), self.first_spin, set_start):
            trim.addWidget(w)
        trim.addStretch(1)
        trim.addWidget(self.range_label)
        trim.addStretch(1)
        for w in (set_end, QLabel("End"), self.last_spin, use_all):
            trim.addWidget(w)
        vlay.addLayout(trim)
        for key, slot in (("I", self.set_start_here), ("O", self.set_end_here)):
            QShortcut(QKeySequence(key), self, activated=slot)
        right.addWidget(viewer)

        tabs = QTabWidget()
        queue_tab = QWidget()
        qlay = QVBoxLayout(queue_tab)
        self.queue_table = QTableWidget(0, 4)
        self.queue_table.setHorizontalHeaderLabels(["Job", "Status", "Progress", "Output"])
        hdr = self.queue_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.Interactive)
        self.queue_table.setColumnWidth(2, 160)
        self.queue_table.setColumnWidth(3, 260)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.queue_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.queue_table.cellDoubleClicked.connect(self._on_queue_double_click)
        qlay.addWidget(self.queue_table, 1)
        qbtns = QHBoxLayout()
        self.start_btn = QPushButton("Start queue")
        self.start_btn.clicked.connect(self.start_queue)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_queue)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_selected)
        clear_btn = QPushButton("Clear finished")
        clear_btn.clicked.connect(self.clear_finished)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self.save_queue_file)
        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self.load_queue_file)
        for b in (self.start_btn, self.cancel_btn, remove_btn, clear_btn):
            qbtns.addWidget(b)
        qbtns.addStretch(1)
        qbtns.addWidget(save_btn)
        qbtns.addWidget(load_btn)
        qlay.addLayout(qbtns)
        tabs.addTab(queue_tab, "Queue")
        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(5000)
        tabs.addTab(self.log_view, "Log")
        self.tabs = tabs
        right.addWidget(tabs)
        right.setSizes([650, 280])
        self._right_splitter = right
        return right

    def _connect_logging(self):
        self._log_handler = QtLogHandler()
        self._log_handler.bridge.message.connect(self._append_log)
        logger.addHandler(self._log_handler)
        logger.setLevel(logging.INFO)

    def _append_log(self, text: str, level: int):
        if level >= logging.ERROR:
            text = f"⚠ {text}"
        self.log_view.appendPlainText(text)

    # ------------------------------------------------------------- form <-> job
    def current_job(self) -> Job:
        n = self._source_info["frames"] if self._source_info else None
        first = self.first_spin.value() - 1
        last = self.last_spin.value()
        return Job(
            kind=self.kind_combo.currentData(),
            source=self.source_picker.path(),
            process=self.process_combo.currentData(),
            params=self.param_form.values(),
            export_dir=self.export_picker.path(),
            export_name=self.name_edit.text().strip(),
            sort=self.sort_combo.currentData(),
            first_frame=first if n else 0,
            end_frame=None if (not n or last >= n) else last,
            ext=self.ext_combo.currentData(),
            blend=self.blend_spin.value(),
            step=self.step_spin.value(),
            fps=float(self.fps_spin.value()),
            master=self.master_combo.currentData(),
            preview_1080p=self.p1080_check.isChecked(),
            preview_4k=self.p4k_check.isChecked(),
            device=self.device_combo.currentData(),
            overwrite=self.overwrite_combo.currentData(),
            deflicker=self.deflicker_check.isChecked(),
            deflicker_window=self.deflicker_spin.value(),
            interpolate=self.interp_combo.currentData(),
        )

    def load_job(self, job: Job, load_source: bool = True):
        self._loading_form = True
        try:
            for combo, value in ((self.kind_combo, job.kind), (self.process_combo, job.process),
                                 (self.sort_combo, job.sort), (self.ext_combo, job.ext),
                                 (self.master_combo, job.master), (self.device_combo, job.device),
                                 (self.overwrite_combo, job.overwrite)):
                idx = combo.findData(value)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self.param_form.set_processor(job.process, job.params)
            self.process_desc.setText(PROCESSORS[job.process].description)
            self.blend_spin.setValue(job.blend)
            self.step_spin.setValue(job.step)
            self.fps_spin.setValue(int(round(job.fps)))
            self.p1080_check.setChecked(job.preview_1080p)
            self.p4k_check.setChecked(job.preview_4k)
            self.deflicker_check.setChecked(job.deflicker)
            self.deflicker_spin.setValue(job.deflicker_window)
            self.interp_combo.setCurrentIndex(max(0, self.interp_combo.findData(job.interpolate)))
            self.export_picker.set_path(job.export_dir)
            self.name_edit.setText(job.export_name)
            self._pending_range = (job.first_frame, job.end_frame)
        finally:
            self._loading_form = False
        self._on_kind_changed()
        if load_source and job.source and job.source != self.source_picker.path():
            self.source_picker.set_path(job.source)
            self._on_source_changed(job.source)
        elif self._source_info:
            self._apply_range()

    # ------------------------------------------------------------------ source
    def _pick_source_folder(self):
        self.source_picker._pick_folder()

    def _pick_source_video(self):
        self.source_picker._pick_file(VIDEO_FILTER)

    def _on_source_changed(self, path: str):
        if self._loading_form:
            return
        path = path.strip()
        self._source_info = None
        self.slider.setMaximum(0)
        if not path:
            return
        if not Path(path).exists():
            self.source_info.setText("Path not found")
            return
        self.settings.setValue("last_source_dir", str(Path(path).parent))
        self.sort_combo.setEnabled(not is_video(path))
        self.source_info.setText("Loading… (reading capture times and checking frames)")
        self.view.set_message("Loading source…")
        if self._queue_thread is None:
            self.sources.forget(path)  # re-scan: the folder may have changed since it was opened
        sort = self.sort_combo.currentData()

        def load():
            src = self.sources.open(path, sort)
            shape, dtype = src.probe()
            return {"path": path, "frames": len(src), "shape": shape, "dtype": str(dtype), "name": src.name,
                    "fps": src.fps, "kind": "video" if is_video(path) else "images"}

        self.tasks.submit(load, on_done=self._on_source_loaded, on_failed=self._on_source_failed)

    def _on_source_loaded(self, info: dict):
        if info["path"] != self.source_picker.path():
            return  # user picked something else meanwhile
        self._source_info = info
        h, w, c = info["shape"]
        depth = {"uint8": "8-bit", "uint16": "16-bit"}.get(info["dtype"], info["dtype"])
        what = f"video, {info['fps']:.3g} fps" if info["kind"] == "video" else "images"
        self.source_info.setText(f"{info['frames']} frames · {w}×{h} · {depth} {what}")
        shoot = default_export(info["path"])[1]
        self.setWindowTitle(f"Timelapse Engine — {shoot}")
        n = info["frames"]
        for spin in (self.first_spin, self.last_spin):
            spin.blockSignals(True)
            spin.setRange(1, n)
            spin.blockSignals(False)
        self.frame_spin.setRange(1, n)
        self.timeline.setMaximum(n - 1)
        self._apply_range()
        if info["kind"] == "video" and info["fps"]:
            self.fps_spin.setValue(int(round(info["fps"])))
        out_dir, _ = default_export(info["path"])
        if not self.export_picker.path() or self.export_picker.path() == self._auto_export_dir:
            self.export_picker.set_path(out_dir)
            self._auto_export_dir = out_dir
        self.name_edit.setPlaceholderText(default_export(info["path"])[1])
        self._preview_shown = False
        self.frame_btn.setChecked(True)
        self._fetch_frame()
        self._update_output_count()
        self.statusBar().showMessage(f"Loaded {info['frames']} frames from {shoot}", 5000)

    def _on_source_failed(self, message: str):
        self.source_info.setText(f"Could not load: {message}")
        self.view.set_message(f"Could not load this source.\n\n{message}")
        logger.error("Could not load source: %s", message)

    # ------------------------------------------------------- frame selection
    # The selection lives in three places that must agree: the timeline's blue
    # handles and the Start/End boxes (both 1-based on screen, 0-based here).
    def _set_range(self, lo: int, hi: int) -> None:
        top = self.timeline.maximum()
        lo, hi = min(max(lo, 0), top), min(max(hi, 0), top)
        hi = max(hi, lo)
        self.timeline.blockSignals(True)
        self.timeline.setSelection(lo, hi)
        self.timeline.blockSignals(False)
        for spin, value in ((self.first_spin, lo + 1), (self.last_spin, hi + 1)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        n = top + 1
        count = hi - lo + 1
        self.range_label.setText(f"Processing all {n} frames" if count == n else
                                 f"Processing frames {lo + 1}–{hi + 1}  ·  {count} of {n}")
        self._update_frame_label()
        self._on_settings_changed()

    def _on_timeline_range(self, lo: int, hi: int):
        self._set_range(lo, hi)

    def _on_first_spin(self, value: int):
        lo = value - 1
        self._set_range(lo, max(self.last_spin.value() - 1, lo))  # pushing start past end moves the end too

    def _on_last_spin(self, value: int):
        hi = value - 1
        self._set_range(min(self.first_spin.value() - 1, hi), hi)

    def set_start_here(self):
        if not self._source_info:
            return
        here = self.timeline.value()
        lo, hi = self.timeline.selection()
        self._set_range(here, hi if here <= hi else self.timeline.maximum())
        self.statusBar().showMessage(f"Start set to frame {here + 1}", 3000)

    def set_end_here(self):
        if not self._source_info:
            return
        here = self.timeline.value()
        lo, hi = self.timeline.selection()
        self._set_range(lo if lo <= here else 0, here)
        self.statusBar().showMessage(f"End set to frame {here + 1}", 3000)

    def _apply_range(self):
        """Apply a range from a loaded job (or select everything)."""
        first, end = getattr(self, "_pending_range", (0, None))
        self._pending_range = (0, None)
        self._set_range(first, self.timeline.maximum() if end is None else end - 1)

    def _reset_range(self):
        if self._source_info:
            self._set_range(0, self.timeline.maximum())

    # ------------------------------------------------------------- frame viewer
    def _on_frame_changed(self, value: int):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value + 1)
        self.frame_spin.blockSignals(False)
        if self.result_btn.isChecked() and self.auto_preview.isChecked() and self.kind_combo.currentData() != "still":
            self._preview_timer.start()  # rolling-window preview follows the scrubber
        else:
            self.frame_btn.setChecked(True)
            self._frame_timer.start()

    def _fetch_frame(self):
        if not self._source_info or self.result_btn.isChecked():
            return
        self._frame_request += 1
        req, idx, path, sort = self._frame_request, self.slider.value(), self._source_info["path"], \
            self.sort_combo.currentData()

        def load():
            import cv2

            src = self.sources.open(path, sort)
            frame = src.read_raw([idx])[0]
            h, w = frame.shape[:2]
            s = DISPLAY_MAX_SIDE / max(h, w)
            if s < 1:
                frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            name = Path(src.paths[idx]).name if hasattr(src, "paths") else f"frame {idx + 1}"
            when = src.capture_time(idx)
            return req, idx, frame, name, when

        self.tasks.submit(load, on_done=self._on_frame_loaded,
                          on_failed=lambda m: self.view.set_message(f"Could not read frame: {m}"))

    def _on_frame_loaded(self, result):
        req, idx, frame, name, when = result
        if req != self._frame_request or self.result_btn.isChecked():
            return
        self.view.set_image(frame)
        self._frame_meta = (idx, name, when)
        self._update_frame_label()
        self.view_status.setText("Source frame (double-click to fit, wheel to zoom)")

    def _update_frame_label(self):
        meta = getattr(self, "_frame_meta", None)
        if not meta:
            return
        idx, name, when = meta
        lo, hi = self.timeline.selection()
        outside = "  ·  not processed" if not lo <= idx <= hi else ""
        self.frame_label.setText(f"{name}" + (f"  ·  {when:%Y-%m-%d %H:%M:%S}" if when else "") + outside)

    def _on_view_mode(self, mode: int):
        if mode == 0:
            self._fetch_frame()
        elif not self._preview_shown:
            self.run_preview()

    # ----------------------------------------------------------------- preview
    def run_preview(self):
        if not self._source_info:
            self.statusBar().showMessage("Load a source first.", 4000)
            return
        job = self.current_job()
        errors = [e for e in job.validate() if "output folder" not in e]
        if errors:
            self.statusBar().showMessage(" ".join(errors), 6000)
            return
        if self._preview_cancel is not None:
            self._preview_cancel.cancel()
        self._preview_cancel = cancel = CancelToken()
        self._preview_request += 1
        req, at, max_side = self._preview_request, self.slider.value(), self.preview_size.currentData()
        self.result_btn.setChecked(True)
        self.view_status.setText("Rendering preview…")
        started = time.perf_counter()

        def render():
            return req, render_preview(job, self.sources, at, max_side=max_side, cancel=cancel), \
                time.perf_counter() - started

        self.tasks.submit(render, on_done=self._on_preview_done, on_failed=self._on_preview_failed)

    def _on_preview_done(self, result):
        req, img, seconds = result
        if req != self._preview_request:
            return
        self._preview_shown = True
        if self.result_btn.isChecked():
            self.view.set_image(img)
        job = self.current_job()
        if job.kind == "still":
            what = "spread over the frame range"
        else:
            what = f"window starting at frame {self.slider.value() + 1}"
        self.view_status.setText(f"Preview: {DISPLAY_NAMES.get(job.process, job.process)}, {what} "
                                 f"({img.shape[1]}×{img.shape[0]}, {seconds:.1f} s)")
        self._gpu_label.setText(_gpu_status())

    def _on_preview_failed(self, message: str):
        self.view_status.setText(f"Preview failed: {message}")
        logger.error("Preview failed: %s", message)

    # ---------------------------------------------------------- settings events
    def _on_kind_changed(self, *_):
        kind = self.kind_combo.currentData()
        rolling = kind in ("video", "frames")
        for w in (self._blend_label, self.blend_spin, self._step_label, self.step_spin, self._interp_label,
                  self.interp_combo):
            w.setVisible(rolling)
        for w in (self._fps_label, self.fps_spin, self._master_label, self.master_combo, self._previews_label,
                  self.p1080_check, self.p4k_check):
            w.setVisible(kind == "video")
        for w in (self._ext_label, self.ext_combo):
            w.setVisible(kind != "video")
        self._on_settings_changed()

    def _on_process_changed(self, *_):
        name = self.process_combo.currentData()
        if name is None:  # a group heading
            return
        self.process_desc.setText(PROCESSORS[name].description)
        if not self._loading_form:
            self.param_form.set_processor(name)
        self._on_settings_changed()

    def _on_settings_changed(self, *_):
        if self._loading_form:
            return
        self._update_output_count()
        if self.auto_preview.isChecked() and self._preview_shown and self.result_btn.isChecked():
            self._preview_timer.start()

    def _update_output_count(self):
        if not self._source_info:
            self.output_count.setText("")
            return
        job = self.current_job()
        n = (job.end_frame or self._source_info["frames"]) - job.first_frame
        if job.kind == "still":
            self.output_count.setText(f"{n} frames → 1 image")
        else:
            outs = len(range(0, n, job.step))
            if job.interpolate > 1 and outs > 1:
                outs = (outs - 1) * job.interpolate + 1
            extra = f" · {outs / job.fps:.1f} s at {job.fps:g} fps" if job.kind == "video" else ""
            self.output_count.setText(f"{n} frames → {outs} output frames{extra}")

    # ------------------------------------------------------------------- queue
    def add_to_queue(self):
        job = self.current_job()
        errors = job.validate()
        if errors:
            QMessageBox.warning(self, "Can't add this job", "\n".join(errors))
            return
        self._append_job(job)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Added to queue", 3000)

    def _append_job(self, job: Job, status: str = STATUS_WAITING):
        row = self.queue_table.rowCount()
        self.jobs.append(job)
        self.queue_table.insertRow(row)
        item = QTableWidgetItem(job.describe())
        item.setToolTip(json.dumps(job.to_dict(), indent=1))
        self.queue_table.setItem(row, 0, item)
        self.queue_table.setItem(row, 1, QTableWidgetItem(status))
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        self.queue_table.setCellWidget(row, 2, bar)
        self.queue_table.setItem(row, 3, QTableWidgetItem(""))

    def _status(self, row: int) -> str:
        return self.queue_table.item(row, 1).text()

    def _set_status(self, row: int, text: str, color: Optional[str] = None):
        item = self.queue_table.item(row, 1)
        item.setText(text)
        if color:
            item.setForeground(QColor(color))

    def start_queue(self):
        if self._queue_thread is not None:
            return
        pending = [(r, self.jobs[r]) for r in range(len(self.jobs))
                   if self._status(r) in (STATUS_WAITING, STATUS_FAILED, STATUS_CANCELLED)]
        if not pending:
            self.statusBar().showMessage("Nothing waiting in the queue.", 4000)
            return
        for r, _ in pending:
            self._set_status(r, STATUS_WAITING, None)
            self.queue_table.cellWidget(r, 2).setValue(0)
        self._queue_runner = runner = QueueRunner(pending, self.sources)
        self._queue_thread = thread = QThread(self)
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.job_started.connect(self._on_job_started)
        runner.job_progress.connect(self._on_job_progress)
        runner.job_finished.connect(self._on_job_finished)
        runner.job_failed.connect(self._on_job_failed)
        runner.job_cancelled.connect(self._on_job_cancelled)
        runner.finished.connect(thread.quit)
        thread.finished.connect(self._on_queue_finished)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        thread.start()

    def cancel_queue(self):
        if self._queue_runner is not None:
            self._queue_runner.stop()
            self.cancel_btn.setEnabled(False)
            self.statusBar().showMessage("Cancelling after the current batch…")

    def _on_job_started(self, row: int):
        self._job_started_at[row] = time.perf_counter()
        self._set_status(row, STATUS_RUNNING, "#5aa9ff")
        self.queue_table.selectRow(row)
        self.statusBar().showMessage(f"Running: {self.jobs[row].describe()}")

    def _on_job_progress(self, row: int, done: int, total: int, message: str):
        bar = self.queue_table.cellWidget(row, 2)
        pct = int(100 * done / max(total, 1))
        bar.setValue(pct)
        if not hasattr(self, "_phase") or self._phase.get(row) != message or done == 0:
            self._phase = {row: message}
            self._phase_start = (time.perf_counter(), done)
        t0, d0 = self._phase_start
        rate = (done - d0) / max(time.perf_counter() - t0, 1e-6)
        eta = f" · {_fmt_eta((total - done) / rate)} left" if rate > 0 and done > d0 else ""
        speed = f" · {rate:.1f} fps" if rate > 0 and done > d0 else ""
        phase = f" {message.split('(')[-1].rstrip(')')}" if "pass" in message else ""
        bar.setFormat(f"{done}/{total}{phase}")
        self._set_status(row, f"{STATUS_RUNNING}{speed}{eta}")

    def _on_job_finished(self, row: int, outputs: list):
        elapsed = time.perf_counter() - self._job_started_at.get(row, time.perf_counter())
        self._set_status(row, f"{STATUS_DONE} ({_fmt_eta(elapsed)})", "#4caf50")
        self.queue_table.cellWidget(row, 2).setValue(100)
        item = self.queue_table.item(row, 3)
        item.setText(Path(outputs[0]).name if outputs else "")
        item.setToolTip("\n".join(outputs) + "\n\nDouble-click to show in Explorer")
        item.setData(Qt.UserRole, outputs)
        self._gpu_label.setText(_gpu_status())

    def _on_job_failed(self, row: int, message: str):
        self._set_status(row, STATUS_FAILED, "#ef5350")
        self.queue_table.item(row, 1).setToolTip(message)
        self.queue_table.item(row, 3).setText(message)
        self.queue_table.item(row, 3).setToolTip(message)

    def _on_job_cancelled(self, row: int):
        self._set_status(row, STATUS_CANCELLED, "#ffb74d")

    def _on_queue_finished(self):
        self._queue_thread.deleteLater()
        self._queue_thread = None
        self._queue_runner = None
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        done = sum(self._status(r).startswith(STATUS_DONE) for r in range(len(self.jobs)))
        self.statusBar().showMessage(f"Queue finished · {done} of {len(self.jobs)} jobs done", 10000)
        QApplication.alert(self)  # flash the taskbar when a long queue ends

    def _on_queue_double_click(self, row: int, col: int):
        if col == 3:
            outs = self.queue_table.item(row, 3).data(Qt.UserRole)
            if outs:
                open_in_explorer(outs[0])
            return
        self.load_job(self.jobs[row])
        self.statusBar().showMessage("Loaded job settings into the form", 3000)

    def remove_selected(self):
        rows = sorted({i.row() for i in self.queue_table.selectedIndexes()}, reverse=True)
        for r in rows:
            if self._status(r) == STATUS_RUNNING or (self._queue_thread is not None and
                                                     self._status(r) == STATUS_WAITING):
                continue  # can't remove what the running queue will use
            self.queue_table.removeRow(r)
            del self.jobs[r]

    def clear_finished(self):
        if self._queue_thread is not None:
            self.statusBar().showMessage("Wait for the queue to finish first.", 4000)
            return
        for r in reversed(range(len(self.jobs))):
            if self._status(r).startswith(STATUS_DONE):
                self.queue_table.removeRow(r)
                del self.jobs[r]

    def save_queue_file(self):
        if not self.jobs:
            self.statusBar().showMessage("The queue is empty.", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save queue", self.settings.value("last_queue_dir", ""),
                                              "Queue files (*.json)")
        if path:
            save_queue(path, self.jobs)
            self.settings.setValue("last_queue_dir", str(Path(path).parent))
            self.statusBar().showMessage(f"Saved {len(self.jobs)} jobs to {Path(path).name}", 5000)

    def load_queue_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load queue", self.settings.value("last_queue_dir", ""),
                                              "Queue files (*.json)")
        if not path:
            return
        try:
            jobs = load_queue(path)
        except Exception as exc:
            QMessageBox.warning(self, "Can't load queue", str(exc))
            return
        for job in jobs:
            self._append_job(job)
        self.settings.setValue("last_queue_dir", str(Path(path).parent))
        self.tabs.setCurrentIndex(0)

    # ---------------------------------------------------------------- settings
    def _restore_settings(self):
        geo = self.settings.value("geometry")
        if isinstance(geo, QByteArray):
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 950)
        for key, splitter in (("main_splitter", self._main_splitter), ("right_splitter", self._right_splitter)):
            state = self.settings.value(key)
            if isinstance(state, QByteArray):
                splitter.restoreState(state)
        self.auto_preview.setChecked(self.settings.value("auto_preview", "true") == "true")
        self.source_picker.start_dir = self.settings.value("last_source_dir", "")
        self.param_form.set_processor(self.process_combo.currentData())
        self.process_desc.setText(PROCESSORS[self.process_combo.currentData()].description)
        last = self.settings.value("last_job")
        if last:
            try:
                self.load_job(Job.from_dict(json.loads(last)))
                return
            except Exception:
                pass
        self._on_kind_changed()

    def closeEvent(self, event):
        if self._queue_thread is not None:
            answer = QMessageBox.question(self, "Jobs are running",
                                          "A job is still running. Cancel it and quit?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._queue_runner.stop()
            self._queue_thread.quit()
            self._queue_thread.wait(30000)
        if self._preview_cancel is not None:
            self._preview_cancel.cancel()
        self.tasks.wait(10000)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("main_splitter", self._main_splitter.saveState())
        self.settings.setValue("right_splitter", self._right_splitter.saveState())
        self.settings.setValue("auto_preview", "true" if self.auto_preview.isChecked() else "false")
        self.settings.setValue("last_job", json.dumps(self.current_job().to_dict()))
        logger.removeHandler(self._log_handler)
        self.sources.close()
        super().closeEvent(event)


def apply_dark_theme(app: QApplication) -> None:
    """Neutral dark theme: easier on the eyes when judging images."""
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, text, mid = QColor(37, 37, 40), QColor(45, 45, 49), QColor(225, 225, 228), QColor(140, 140, 146)
    for role, color in ((QPalette.Window, QColor(32, 32, 35)), (QPalette.WindowText, text),
                        (QPalette.Base, base), (QPalette.AlternateBase, alt), (QPalette.Text, text),
                        (QPalette.Button, alt), (QPalette.ButtonText, text), (QPalette.ToolTipBase, alt),
                        (QPalette.ToolTipText, text), (QPalette.Highlight, QColor(53, 132, 228)),
                        (QPalette.HighlightedText, QColor(255, 255, 255)), (QPalette.Mid, mid),
                        (QPalette.PlaceholderText, mid), (QPalette.Link, QColor(90, 169, 255))):
        p.setColor(role, color)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(110, 110, 115))
    app.setPalette(p)
    app.setStyleSheet(
        "QToolButton { padding: 3px 12px; border: 1px solid #4a4a50; border-radius: 3px; background: #2d2d31; }"
        "QToolButton:checked { background: #3584e4; border-color: #3584e4; color: white; }"
        "QGroupBox { font-weight: 600; }")


def main(argv=None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Timelapse Engine")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
