"""Reusable Qt widgets: zoomable image view, processor settings form, path picker."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QWidget)

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
