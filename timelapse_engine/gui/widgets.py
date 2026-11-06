"""Reusable Qt widgets: zoomable image view, processor settings form, path picker."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QSpinBox, QToolTip, QWidget)

from ..files import to_uint8
from ..jobs import ParamSpec, processor_params


def to_display(img: np.ndarray) -> np.ndarray:
    """Any frame/result (uint8, uint16 or float 0-1; 1 or 3 channels) -> contiguous RGB uint8."""
    img = to_uint8(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    return np.ascontiguousarray(img[:, :, :3])


class ImageView(QGraphicsView):
    """Shows an image fitted to the view. Wheel zooms, drag pans, double-click fits again."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(28, 28, 30))
        self.setFrameShape(QGraphicsView.NoFrame)
        self._fit = True
        self._message = "Choose an image folder or video file to begin."

    def set_message(self, text: str) -> None:
        self._message = text
        self._item.setPixmap(QPixmap())
        self.viewport().update()

    def set_image(self, img: np.ndarray) -> None:
        rgb = to_display(img)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._item.setPixmap(QPixmap.fromImage(qimg))
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self._message = ""
        if self._fit:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def drawForeground(self, painter, rect):
        if self._message:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor(170, 170, 175))
            painter.drawText(self.viewport().rect(), Qt.AlignCenter | Qt.TextWordWrap, self._message)
            painter.restore()

    def wheelEvent(self, event):
        if self._item.pixmap().isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._fit = False

    def mouseDoubleClickEvent(self, event):
        self._fit = True
        self.fitInView(self._item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit and not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.KeepAspectRatio)


class Timeline(QWidget):
    """Frame timeline: a playhead for the current frame plus start/end handles
    that select which frames are processed.

    Drag the white playhead (or click anywhere on the bar) to scrub; drag the
    blue start/end handles to trim. Frames outside the selection are dimmed.
    Keys: Left/Right (Shift = 10 frames), Home/End. Mouse wheel steps frames.
    Frames are 0-based internally; tooltips show them 1-based.
    """

    valueChanged = Signal(int)
    rangeChanged = Signal(int, int)

    PAD = 10        # keeps the end handles grabbable
    GRAB = 7        # pixels either side of a handle that grab it

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max = 0
        self._value = 0
        self._lo = 0
        self._hi = 0
        self._drag: Optional[str] = None
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip("Drag the white playhead to scrub; drag the blue handles to choose the first and last "
                        "frames to process.")

    # -- QSlider-like API ---------------------------------------------------
    def maximum(self) -> int:
        return self._max

    def setMaximum(self, maximum: int) -> None:
        self._max = max(0, int(maximum))
        self._lo, self._hi = 0, self._max
        self._value = min(self._value, self._max)
        self.update()

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        value = min(max(int(value), 0), self._max)
        if value != self._value:
            self._value = value
            self.update()
            self.valueChanged.emit(value)

    def selection(self) -> Tuple[int, int]:
        """(first, last) selected frame, inclusive, 0-based."""
        return self._lo, self._hi

    def setSelection(self, lo: int, hi: int) -> None:
        lo = min(max(int(lo), 0), self._max)
        hi = min(max(int(hi), 0), self._max)
        lo, hi = min(lo, hi), max(lo, hi)
        if (lo, hi) != (self._lo, self._hi):
            self._lo, self._hi = lo, hi
            self.update()
            self.rangeChanged.emit(lo, hi)

    # -- geometry -----------------------------------------------------------
    def _track(self) -> QRectF:
        h = self.height()
        return QRectF(self.PAD, h * 0.42, max(1, self.width() - 2 * self.PAD), 9)

    def _x(self, frame: int) -> float:
        t = self._track()
        return t.left() + (frame / max(self._max, 1)) * t.width()

    def _frame_at(self, x: float) -> int:
        t = self._track()
        return int(round(min(max((x - t.left()) / t.width(), 0.0), 1.0) * self._max))

    def _handle_at(self, x: float, y: float) -> Optional[str]:
        if y < self._track().top() - 4:
            return None  # the top strip always scrubs
        d_lo, d_hi = abs(x - self._x(self._lo)), abs(x - self._x(self._hi))
        if min(d_lo, d_hi) > self.GRAB:
            return None
        if self._lo == self._hi:
            return "lo" if x <= self._x(self._lo) else "hi"
        return "lo" if d_lo <= d_hi else "hi"

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = self._track()
        x_lo, x_hi, x_val = self._x(self._lo), self._x(self._hi), self._x(self._value)
        enabled = self._max > 0

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(52, 52, 57))
        p.drawRoundedRect(t, 3, 3)                                           # excluded frames
        if enabled:
            p.setBrush(QColor(53, 132, 228, 200))
            p.drawRect(QRectF(x_lo, t.top(), max(x_hi - x_lo, 2), t.height()))  # selected frames

            # start / end handles: tall blue bars with a grip below the track
            for x in (x_lo, x_hi):
                p.setBrush(QColor(156, 196, 245))
                p.drawRoundedRect(QRectF(x - 2.5, t.top() - 6, 5, t.height() + 12), 2, 2)
                p.drawPolygon([QPointF(x - 6, t.bottom() + 14), QPointF(x + 6, t.bottom() + 14),
                               QPointF(x, t.bottom() + 7)])

            # playhead: white line with a knob on top
            p.setPen(QPen(QColor(245, 245, 245), 2))
            p.drawLine(QPointF(x_val, 3), QPointF(x_val, t.bottom() + 3))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(245, 245, 245))
            p.drawPolygon([QPointF(x_val - 6, 2), QPointF(x_val + 6, 2), QPointF(x_val, 10)])
        if self.hasFocus():
            p.setPen(QPen(QColor(53, 132, 228, 120), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)

    # -- interaction --------------------------------------------------------
    def mousePressEvent(self, event):
        if self._max <= 0 or event.button() != Qt.LeftButton:
            return
        pos = event.position()
        self._drag = self._handle_at(pos.x(), pos.y()) or "value"
        self._drag_to(pos)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._drag:
            self._drag_to(pos)
        else:
            over = self._handle_at(pos.x(), pos.y())
            self.setCursor(Qt.SizeHorCursor if over else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._drag = None

    def _drag_to(self, pos):
        f = self._frame_at(pos.x())
        if self._drag == "lo":
            self.setSelection(min(f, self._hi), self._hi)
            label = f"Start: frame {self._lo + 1}"
        elif self._drag == "hi":
            self.setSelection(self._lo, max(f, self._lo))
            label = f"End: frame {self._hi + 1}"
        else:
            self.setValue(f)
            label = f"Frame {self._value + 1}"
        QToolTip.showText(self.mapToGlobal(pos.toPoint()) + QPoint(10, -30), label, self)

    def keyPressEvent(self, event):
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Down):
            self.setValue(self._value - step)
        elif key in (Qt.Key_Right, Qt.Key_Up):
            self.setValue(self._value + step)
        elif key == Qt.Key_Home:
            self.setValue(0)
        elif key == Qt.Key_End:
            self.setValue(self._max)
        elif key == Qt.Key_PageUp:
            self.setValue(self._value - max(1, self._max // 10))
        elif key == Qt.Key_PageDown:
            self.setValue(self._value + max(1, self._max // 10))
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1
        self.setValue(self._value + (-step if event.angleDelta().y() > 0 else step))


class ParamForm(QWidget):
    """Settings form generated from a processor's constructor parameters."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._widgets: Dict[str, QWidget] = {}
        self._specs: List[ParamSpec] = []

    def set_processor(self, name: str, values: Optional[dict] = None) -> None:
        values = values or {}
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._widgets.clear()
        self._specs = processor_params(name)
        for spec in self._specs:
            value = values.get(spec.name, spec.default)
            if spec.kind is bool:
                w = QCheckBox()
                w.setChecked(bool(value))
                w.toggled.connect(self.changed)
            elif spec.kind is int:
                w = QSpinBox()
                w.setRange(int(spec.minimum), int(spec.maximum))
                w.setSingleStep(int(spec.step))
                w.setValue(int(value))
                w.valueChanged.connect(self.changed)
            else:
                w = QDoubleSpinBox()
                w.setDecimals(3 if spec.maximum <= 2 else 1)
                w.setRange(float(spec.minimum), float(spec.maximum))
                w.setSingleStep(float(spec.step))
                w.setValue(float(value))
                w.valueChanged.connect(self.changed)
            w.setToolTip(spec.help)
            label = QLabel(spec.label)
            label.setToolTip(spec.help)
            self._layout.addRow(label, w)
            self._widgets[spec.name] = w
        if not self._specs:
            note = QLabel("No settings for this effect.")
            note.setEnabled(False)
            self._layout.addRow(note)

    def values(self) -> dict:
        out = {}
        for spec in self._specs:
            w = self._widgets[spec.name]
            if isinstance(w, QCheckBox):
                v = w.isChecked()
            else:
                v = w.value()
            if v != spec.default:  # keep jobs readable: only store changed settings
                out[spec.name] = v
        return out


class PathPicker(QWidget):
    """Line edit plus browse button(s)."""

    changed = Signal(str)

    def __init__(self, placeholder: str = "", folder_button: str = "Folder…", file_button: str = "",
                 file_filter: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.editingFinished.connect(lambda: self.changed.emit(self.path()))
        lay.addWidget(self.edit, 1)
        self.start_dir = ""
        if folder_button:
            b = QPushButton(folder_button)
            b.clicked.connect(self._pick_folder)
            lay.addWidget(b)
        if file_button:
            b = QPushButton(file_button)
            b.clicked.connect(lambda: self._pick_file(file_filter))
            lay.addWidget(b)

    def path(self) -> str:
        return self.edit.text().strip().strip('"')

    def set_path(self, path: str, emit: bool = False) -> None:
        self.edit.setText(path)
        if emit:
            self.changed.emit(path)

    def _start(self) -> str:
        return self.path() or self.start_dir

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose folder", self._start())
        if path:
            self.set_path(str(path).replace("/", "\\"), emit=True)

    def _pick_file(self, file_filter: str):
        path, _ = QFileDialog.getOpenFileName(self, "Choose file", self._start(), file_filter)
        if path:
            self.set_path(str(path).replace("/", "\\"), emit=True)
