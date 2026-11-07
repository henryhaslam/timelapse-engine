"""Sort tab: split a card dump of raw files into one folder per timelapse sequence.

Pick the folder, Analyse, check the groups (table, first/last frame, spacing chart),
then Move or Copy. Settings changes regroup instantly without re-reading the files.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import sorter
from ..progress import CancelToken
from .widgets import ImageView, PathPicker
from .worker import TaskRunner

GROUP_COLOURS = ["#4e9bff", "#f5a742", "#5fd17a", "#e86a8a", "#b48cff", "#4fd1d9", "#e8d44d", "#ff7a59"]
BREAK_COLOUR = "#6e6e75"
COLUMNS = ["Folder", "Start", "Length", "Frames", "Spacing", "Camera", "Files"]


def group_colour(index: int) -> QColor:
    return QColor(GROUP_COLOURS[index % len(GROUP_COLOURS)]) if index >= 0 else QColor(BREAK_COLOUR)


class _Progress(QObject):
    changed = Signal(int, int, str)


class SpacingChart(QWidget):
    """Spacing between each shot and the one before it, log scale, one colour per sequence.

    Flat coloured lines are sequences; grey dots are the gaps between them and single
    shots. Click a dot to select its sequence.
    """

    groupClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setToolTip("Spacing between each shot and the one before it (log scale).\n"
                        "Each colour is one sequence; grey dots are gaps between sequences and single shots.\n"
                        "Click a dot to select its sequence.")
        self._points: List[tuple] = []     # (spacing seconds or None, group index or -1)
        self._ranges: Dict[int, tuple] = {}
        self._selected = -1
        self._lo, self._hi = 0.05, 100.0

    def set_data(self, analysis: Optional[sorter.Analysis]) -> None:
        self._points, self._ranges, self._selected = [], {}, -1
        if analysis is not None:
            where = {id(s): gi for gi, g in enumerate(analysis.groups) for s in g.shots}
            previous: Dict[str, sorter.Shot] = {}
            for x, shot in enumerate(analysis.shots):
                prev = previous.get(shot.camera)
                previous[shot.camera] = shot
                gi = where.get(id(shot), -1)
                spacing = (shot.time - prev.time).total_seconds() if prev is not None else None
                same = prev is not None and gi >= 0 and where.get(id(prev), -2) == gi
                self._points.append((spacing, gi if same else -1))
                if gi >= 0:
                    lo, hi = self._ranges.get(gi, (x, x))
                    self._ranges[gi] = (min(lo, x), max(hi, x))
            values = [p[0] for p in self._points if p[0] is not None and p[0] > 0]
            if values:
                self._lo = 10 ** math.floor(math.log10(max(min(values), 0.01)))
                self._hi = 10 ** math.ceil(math.log10(max(max(values), self._lo * 10)))
        self.update()

    def set_selected(self, gi: int) -> None:
        self._selected = gi
        self.update()

    def _plot(self) -> QRectF:
        return QRectF(48, 8, max(10, self.width() - 56), max(10, self.height() - 16))

    def _x(self, i: int, plot: QRectF) -> float:
        n = max(len(self._points) - 1, 1)
        return plot.left() + plot.width() * i / n

    def _y(self, seconds: float, plot: QRectF) -> float:
        lo, hi = math.log10(self._lo), math.log10(self._hi)
        v = (math.log10(max(seconds, self._lo)) - lo) / max(hi - lo, 1e-9)
        return plot.bottom() - plot.height() * min(max(v, 0.0), 1.0)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(28, 28, 30))
        plot = self._plot()
        if not self._points:
            p.setPen(QColor(150, 150, 155))
            p.drawText(self.rect(), Qt.AlignCenter, "Spacing between shots appears here after Analyse.")
            return
        if self._selected in self._ranges:
            lo, hi = self._ranges[self._selected]
            band = QRectF(self._x(lo, plot) - 3, plot.top(), self._x(hi, plot) - self._x(lo, plot) + 6, plot.height())
            colour = group_colour(self._selected)
            colour.setAlpha(45)
            p.fillRect(band, colour)
        p.setPen(QPen(QColor(60, 60, 65), 1))
        decade = self._lo
        while decade <= self._hi * 1.001:
            y = self._y(decade, plot)
            p.setPen(QPen(QColor(55, 55, 60), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(140, 140, 146))
            label = f"{decade:g} s"
            p.drawText(QRectF(0, y - 8, plot.left() - 6, 16), Qt.AlignRight | Qt.AlignVCenter, label)
            decade *= 10
        radius = 2.4 if len(self._points) < 2000 else 1.6
        p.setPen(Qt.NoPen)
        for i, (spacing, gi) in enumerate(self._points):
            if spacing is None:
                continue
            colour = group_colour(gi)
            if self._selected >= 0 and gi != self._selected:
                colour.setAlpha(110)
            p.setBrush(colour)
            p.drawEllipse(QPointF(self._x(i, plot), self._y(spacing, plot)), radius, radius)

    def mousePressEvent(self, event):
        if not self._points:
            return
        plot = self._plot()
        n = max(len(self._points) - 1, 1)
        i = round((event.position().x() - plot.left()) / max(plot.width(), 1) * n)
        i = min(max(i, 0), len(self._points) - 1)
        for gi, (lo, hi) in self._ranges.items():
            if lo <= i <= hi:
                self.groupClicked.emit(gi)
                return


class SortPanel(QWidget):
    """The 'Sort raw files' tab."""

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.tasks = TaskRunner(self, max_threads=3)
        self.analysis: Optional[sorter.Analysis] = None
        self.confirm = True                 # tests switch the Move/Undo confirmation dialogs off
        self.last_log: Optional[str] = None
        self._cancel: Optional[CancelToken] = None
        self._busy = False
        self._preview_request = 0
        self._filling = False
        self._progress = _Progress()
        self._progress.changed.connect(self._on_progress)
        self._regroup_timer = QTimer(self, singleShot=True, interval=250, timeout=self.regroup)
        self._build_ui()
        self._restore()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        lay = QVBoxLayout(self)

        intro = QLabel("Split a card's worth of raw files into one folder per timelapse sequence. Shots are "
                       "grouped by the spacing of their capture times: a run of evenly spaced shots is one "
                       "sequence, and a gap or a new spacing starts the next.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: palette(mid);")
        lay.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Raw files"))
        self.folder_picker = PathPicker("Folder of raw files from the card")
        self.folder_picker.changed.connect(lambda _: self.analyse())
        row.addWidget(self.folder_picker, 1)
        self.subfolders_check = QCheckBox("Include subfolders")
        self.subfolders_check.setToolTip("Also look in subfolders, e.g. DCIM\\100_PANA, 101_PANA from a card. "
                                         "Leave off if the folder already holds sorted or exported files.")
        row.addWidget(self.subfolders_check)
        self.analyse_btn = QPushButton("Analyse")
        self.analyse_btn.clicked.connect(self.analyse)
        row.addWidget(self.analyse_btn)
        lay.addLayout(row)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Spacing tolerance"))
        self.tolerance_spin = QSpinBox(minimum=5, maximum=100, value=25, suffix=" %")
        self.tolerance_spin.setToolTip("How far a gap may differ from the recent spacing and still continue the "
                                       "sequence. Raise it if one sequence is split in two; lower it if two "
                                       "sequences are joined.")
        opts.addWidget(self.tolerance_spin)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Fewest frames"))
        self.min_frames_spin = QSpinBox(minimum=3, maximum=10000, value=5)
        self.min_frames_spin.setToolTip("Shorter runs count as single shots.")
        opts.addWidget(self.min_frames_spin)
        opts.addSpacing(12)
        self.counter_check = QCheckBox("Split where the camera's shot counter restarts")
        self.counter_check.setChecked(True)
        self.counter_check.setToolTip("Many cameras number the shots of each burst or interval run "
                                      "(Panasonic SequenceNumber). A restart is a new sequence even when the gap "
                                      "looks like the spacing.")
        opts.addWidget(self.counter_check)
        opts.addStretch(1)
        lay.addLayout(opts)

        names = QHBoxLayout()
        names.addWidget(QLabel("Folder names"))
        self.template_edit = QLineEdit(sorter.DEFAULT_TEMPLATE)
        self.template_edit.setToolTip("Pattern for new folder names. Fields: "
                                      + ", ".join("{" + f + "}" for f in sorter.TEMPLATE_FIELDS)
                                      + "\ne.g. {date}_{time}_{interval}_{frames}f gives 2026.06.07_131442_2s_270f."
                                      "\nYou can also edit each name in the table.")
        self.template_edit.editingFinished.connect(self._apply_template)
        names.addWidget(self.template_edit, 1)
        names.addSpacing(12)
        names.addWidget(QLabel("Put folders in"))
        self.dest_picker = PathPicker("the raw folder itself")
        self.dest_picker.setToolTip("Leave empty to create the sequence folders inside the raw folder.")
        names.addWidget(self.dest_picker, 1)
        lay.addLayout(names)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(COLUMNS)):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemChanged.connect(self._on_item_changed)

        previews = QWidget()
        play = QHBoxLayout(previews)
        play.setContentsMargins(0, 0, 0, 0)
        self.first_view, self.last_view = ImageView(), ImageView()
        self.first_label, self.last_label = QLabel("First frame"), QLabel("Last frame")
        for view, label in ((self.first_view, self.first_label), (self.last_view, self.last_label)):
            view.set_message("Select a sequence")
            col = QVBoxLayout()
            col.addWidget(label)
            col.addWidget(view, 1)
            play.addLayout(col, 1)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.table)
        top.addWidget(previews)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)
        top.setSizes([800, 600])
        self.chart = SpacingChart()
        self.chart.groupClicked.connect(self._select_group)
        body = QSplitter(Qt.Vertical)
        body.addWidget(top)
        body.addWidget(self.chart)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 1)
        body.setSizes([520, 170])
        self._top_splitter, self._body_splitter = top, body
        lay.addWidget(body, 1)

        bottom = QHBoxLayout()
        self.summary_label = QLabel("Choose a folder of raw files to begin.")
        self.summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.summary_label.setWordWrap(True)
        bottom.addWidget(self.summary_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setVisible(False)
        bottom.addWidget(self.progress_bar)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel)
        bottom.addWidget(self.cancel_btn)
        self.move_btn = QPushButton("Move into folders")
        self.move_btn.setToolTip("Move the ticked sequences into their folders (instant on the same drive). "
                                 "A log is written so Undo can put everything back.")
        self.move_btn.clicked.connect(lambda: self.sort("move"))
        self.copy_btn = QPushButton("Copy into folders")
        self.copy_btn.setToolTip("Copy the ticked sequences, leaving the originals where they are.")
        self.copy_btn.clicked.connect(lambda: self.sort("copy"))
        self.undo_btn = QPushButton("Undo last sort")
        self.undo_btn.clicked.connect(self.undo)
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(self._open_folder)
        for b in (self.move_btn, self.copy_btn, self.undo_btn, open_btn):
            bottom.addWidget(b)
        lay.addLayout(bottom)

        for w in (self.tolerance_spin, self.min_frames_spin):
            w.valueChanged.connect(lambda *_: self._regroup_timer.start())
        self.counter_check.toggled.connect(lambda *_: self._regroup_timer.start())
        self._update_buttons()

    # ---------------------------------------------------------------- state
    @property
    def busy(self) -> bool:
        return self._busy

    def dest_root(self) -> str:
        return self.dest_picker.path() or self.folder_picker.path()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        self.progress_bar.setVisible(busy)
        self.cancel_btn.setVisible(busy)
        self.progress_bar.setRange(0, 0)
        for w in (self.folder_picker, self.subfolders_check, self.analyse_btn, self.tolerance_spin,
                  self.min_frames_spin, self.counter_check, self.template_edit, self.dest_picker, self.table):
            w.setEnabled(not busy)
        if message:
            self.summary_label.setText(message)
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_groups = self.analysis is not None and bool(self._checked_groups())
        self.move_btn.setEnabled(not self._busy and has_groups)
        self.copy_btn.setEnabled(not self._busy and has_groups)
        self.undo_btn.setEnabled(not self._busy and bool(self.dest_root()))

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"{message}  %v / %m" if message else "%v / %m")

    def _report(self, done: int, total: int, message: str = "") -> None:   # called on worker threads
        self._progress.changed.emit(done, total, message)

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel.cancel()

    # -------------------------------------------------------------- analyse
    def analyse(self) -> None:
        folder = self.folder_picker.path()
        if self._busy:
            return
        if not folder or not Path(folder).is_dir():
            self.summary_label.setText("Choose a folder of raw files to begin." if not folder
                                       else f"Not a folder: {folder}")
            return
        self.analysis = None
        self._fill_table()
        self._cancel = CancelToken()
        self._set_busy(True, f"Reading capture times in {folder}…")
        self.tasks.submit(sorter.analyse, folder, self.subfolders_check.isChecked(),
                          self.tolerance_spin.value() / 100, self.min_frames_spin.value(),
                          self.counter_check.isChecked(), self._template(), True, None, self._report, self._cancel,
                          on_done=self._on_analysed, on_failed=self._on_failed)
        self._watch_cancel()

    def _watch_cancel(self, message: str = "Cancelled.") -> None:
        """Tasks that are cancelled end silently: notice that and tidy up."""
        token = self._cancel

        def check():
            if self._busy and token is self._cancel and token.cancelled:
                if self.tasks.pool.activeThreadCount() == 0:
                    self._set_busy(False, message)
                    return
            if self._busy and token is self._cancel:
                QTimer.singleShot(200, check)

        QTimer.singleShot(200, check)

    def _template(self) -> str:
        return self.template_edit.text().strip() or sorter.DEFAULT_TEMPLATE

    def _on_analysed(self, analysis: sorter.Analysis) -> None:
        self.analysis = analysis
        self._set_busy(False)
        self._fill_table()

    def _on_failed(self, message: str) -> None:
        self._set_busy(False, f"⚠ {message}")

    def regroup(self) -> None:
        """Re-split the shots already read with the current settings (no file access)."""
        if self.analysis is None or self._busy:
            return
        a = self.analysis
        groups, singles = sorter.find_sequences(a.shots, self.tolerance_spin.value() / 100,
                                                self.min_frames_spin.value(), self.counter_check.isChecked())
        try:
            sorter.name_groups(groups, self._template())
        except ValueError as exc:
            self.summary_label.setText(f"⚠ {exc}")
            return
        self.analysis = sorter.Analysis(a.folder, groups, singles, a.undated, a.ignored, a.shots)
        self._fill_table()

    def _apply_template(self) -> None:
        if self.analysis is None:
            return
        try:
            sorter.name_groups(self.analysis.groups, self._template())
        except ValueError as exc:
            self.summary_label.setText(f"⚠ {exc}")
            return
        self._fill_table()

    # ---------------------------------------------------------------- table
    def _rows(self) -> List[sorter.Group]:
        if self.analysis is None:
            return []
        return list(self.analysis.groups) + ([self.analysis.singles] if self.analysis.singles.frames else [])

    def _fill_table(self) -> None:
        self._filling = True
        rows = self._rows()
        self.table.setRowCount(len(rows))
        for r, g in enumerate(rows):
            name = QTableWidgetItem(g.name)
            name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Unchecked if g.singles else Qt.Checked)
            name.setData(Qt.DecorationRole, group_colour(-1 if g.singles else r))
            name.setToolTip("Single shots: not part of any sequence. Tick to move them into this folder too."
                            if g.singles else "Double-click to rename. Untick to leave these files where they are.")
            cells = [name,
                     QTableWidgetItem("" if g.singles else f"{g.start:%Y-%m-%d  %H:%M:%S}"),
                     QTableWidgetItem("" if g.singles else sorter.fmt_duration(g.duration)),
                     QTableWidgetItem(str(g.frames)),
                     QTableWidgetItem("—" if g.singles else f"every {sorter.fmt_interval(g.interval)}"),
                     QTableWidgetItem(g.camera),
                     QTableWidgetItem(f"{g.shots[0].name} – {g.shots[-1].name}" if g.frames > 1
                                      else g.shots[0].name)]
            for c, item in enumerate(cells):
                if c:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if c == 3:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
        self._filling = False
        self.chart.set_data(self.analysis)
        if self.analysis is not None:
            text = self.analysis.summary()
            if any(s.from_file_time for s in self.analysis.shots):
                text += ". Some shots had no EXIF capture time, so their file time was used"
            self.summary_label.setText(text + ".")
            if rows:
                self.table.selectRow(0)
        else:
            self.first_view.set_message("Select a sequence")
            self.last_view.set_message("Select a sequence")
        self._update_buttons()

    def _checked_groups(self) -> List[sorter.Group]:
        rows = self._rows()
        return [g for r, g in enumerate(rows)
                if self.table.item(r, 0) is not None and self.table.item(r, 0).checkState() == Qt.Checked]

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != 0:
            return
        rows = self._rows()
        if item.row() < len(rows):
            rows[item.row()].name = item.text().strip()
        self._update_buttons()

    def _select_group(self, gi: int) -> None:
        if 0 <= gi < self.table.rowCount():
            self.table.selectRow(gi)

    def _on_selection(self) -> None:
        rows = self._rows()
        sel = self.table.selectionModel().selectedRows()
        if not sel or sel[0].row() >= len(rows):
            return
        r = sel[0].row()
        g = rows[r]
        self.chart.set_selected(-1 if g.singles else r)
        self._preview_request += 1
        request = self._preview_request
        for view, label, shot, which in ((self.first_view, self.first_label, g.shots[0], "First"),
                                         (self.last_view, self.last_label, g.shots[-1], "Last")):
            when = f"  {shot.time:%H:%M:%S}" if shot.time else ""
            label.setText(f"{which} frame: {shot.name}{when}")
            view.set_message("Loading…")
            self.tasks.submit(sorter.preview_image, shot.path,
                              on_done=lambda img, v=view, req=request: self._show_preview(v, img, req),
                              on_failed=lambda msg, v=view: v.set_message(f"No preview: {msg}"))

    def _show_preview(self, view: ImageView, img, request: int) -> None:
        if request != self._preview_request:
            return
        if img is None:
            view.set_message("No preview in this file (exiftool needed for raw files)")
        else:
            view.set_image(img)

    # ------------------------------------------------------------ sort/undo
    def _validate_names(self, groups: List[sorter.Group]) -> Optional[str]:
        seen = set()
        for g in groups:
            if not g.name:
                return "Every ticked folder needs a name."
            if sorter._safe_name(g.name) != g.name:
                return f"'{g.name}' can't be a folder name (no < > : \" / \\ | ? * or trailing dot)."
            if g.name.lower() in seen:
                return f"Two ticked groups are both called '{g.name}'."
            seen.add(g.name.lower())
        return None

    def sort(self, mode: str) -> None:
        if self.analysis is None or self._busy:
            return
        groups = self._checked_groups()
        problem = self._validate_names(groups)
        if problem:
            QMessageBox.warning(self, "Folder names", problem)
            return
        dest = self.dest_root()
        files = [f for g in groups for f in g.files]
        if self.confirm:
            size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
            verb = "Move" if mode == "move" else "Copy"
            answer = QMessageBox.question(
                self, f"{verb} files",
                f"{verb} {len(files)} files ({size / 1e9:.1f} GB) into {len(groups)} folder"
                f"{'s' if len(groups) != 1 else ''} in\n{dest}?\n\nNothing is overwritten, and a log is written "
                f"so 'Undo last sort' can put everything back.")
            if answer != QMessageBox.Yes:
                return
        self._cancel = CancelToken()
        self._set_busy(True, f"{'Moving' if mode == 'move' else 'Copying'} {len(files)} files…")
        self.tasks.submit(sorter.sort_files, groups, dest, mode, self._report, self._cancel,
                          on_done=lambda log, m=mode, n=len(groups): self._on_sorted(log, m, n),
                          on_failed=self._on_sort_failed)
        self._watch_cancel("Cancelled. Files already moved are in the log; 'Undo last sort' puts them back.")

    def _on_sorted(self, log_path, mode: str, n_groups: int) -> None:
        self.last_log = str(log_path)
        self.analysis = None
        self._fill_table()
        self.chart.set_data(None)
        self._set_busy(False, f"{'Moved' if mode == 'move' else 'Copied'} into {n_groups} folder"
                              f"{'s' if n_groups != 1 else ''} in {self.dest_root()}. "
                              f"'Undo last sort' puts everything back.")

    def _on_sort_failed(self, message: str) -> None:
        self._set_busy(False, f"⚠ Stopped: {message}. Files already moved are in the log; "
                              f"'Undo last sort' puts them back.")

    def undo(self) -> None:
        if self._busy:
            return
        dest = self.dest_root()
        log = sorter.find_last_log(dest) if dest and Path(dest).is_dir() else None
        if log is None:
            QMessageBox.information(self, "Undo", f"There is no sort to undo in\n{dest or '(no folder chosen)'}")
            return
        if self.confirm:
            answer = QMessageBox.question(self, "Undo last sort",
                                          f"Put back the files from the sort logged in\n{log}?")
            if answer != QMessageBox.Yes:
                return
        self._cancel = CancelToken()
        self._set_busy(True, "Undoing the last sort…")
        self.tasks.submit(sorter.undo, log, self._report, self._cancel,
                          on_done=self._on_undone, on_failed=self._on_failed)
        self._watch_cancel()

    def _on_undone(self, result) -> None:
        restored, problems = result
        text = f"Undo: {restored} files put back."
        if problems:
            text += f" {len(problems)} could not be (see the Log tab)."
            from ..progress import logger
            for p in problems:
                logger.warning("Undo: %s", p)
        self.analysis = None
        self._fill_table()
        self._set_busy(False, text)

    def _open_folder(self) -> None:
        from .app import open_in_explorer

        path = self.dest_root()
        if path:
            open_in_explorer(path)

    # ------------------------------------------------------------- settings
    def _restore(self) -> None:
        s = self.settings
        self.folder_picker.set_path(s.value("sort/folder", ""))
        self.subfolders_check.setChecked(s.value("sort/subfolders", "false") == "true")
        self.tolerance_spin.setValue(int(s.value("sort/tolerance", 25)))
        self.min_frames_spin.setValue(int(s.value("sort/min_frames", 5)))
        self.counter_check.setChecked(s.value("sort/counter", "true") == "true")
        self.template_edit.setText(s.value("sort/template", sorter.DEFAULT_TEMPLATE))
        self.dest_picker.set_path(s.value("sort/dest", ""))
        for key, splitter in (("sort/top_splitter", self._top_splitter), ("sort/body_splitter", self._body_splitter)):
            state = s.value(key)
            if state is not None:
                splitter.restoreState(state)
        self._update_buttons()

    def save_settings(self) -> None:
        s = self.settings
        s.setValue("sort/folder", self.folder_picker.path())
        s.setValue("sort/subfolders", "true" if self.subfolders_check.isChecked() else "false")
        s.setValue("sort/tolerance", self.tolerance_spin.value())
        s.setValue("sort/min_frames", self.min_frames_spin.value())
        s.setValue("sort/counter", "true" if self.counter_check.isChecked() else "false")
        s.setValue("sort/template", self._template())
        s.setValue("sort/dest", self.dest_picker.path())
        s.setValue("sort/top_splitter", self._top_splitter.saveState())
        s.setValue("sort/body_splitter", self._body_splitter.saveState())

    def shutdown(self) -> None:
        self.cancel()
        self.tasks.wait(30000)
