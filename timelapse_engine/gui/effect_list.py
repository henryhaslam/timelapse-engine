"""Choosing which effects appear in the Effect list.

The full list has over twenty effects. The owner ticks the ones they use; the rest stay
available through "Show all effects" or by ticking them again.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from ..processors import CATEGORY_ORDER, PROCESSORS

EDIT_LIST = "__edit_effect_list__"   # item data of the "Edit this list…" entry at the top of the Effect list


def effect_categories() -> List[tuple]:
    """[(category, [effect names])] in display order, skipping empty categories."""
    categories = CATEGORY_ORDER + sorted({c.category for c in PROCESSORS.values()} - set(CATEGORY_ORDER))
    out = []
    for category in categories:
        names = [n for n, c in PROCESSORS.items() if c.category == category]
        if names:
            out.append((category, names))
    return out


class EffectCombo(QComboBox):
    """QComboBox that knows whether a change of item came from its pop-up list, so picking
    "Edit this list…" there opens the dialog but scrolling the wheel onto it doesn't.

    Qt 6 closes the pop-up *before* it changes the current item, so "just closed" (until the
    next pass of the event loop) counts as well as "open".
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.popup_open = False
        self._just_closed = False

    @property
    def picked_from_popup(self) -> bool:
        return self.popup_open or self._just_closed

    def showPopup(self):
        self.popup_open = True
        super().showPopup()

    def hidePopup(self):
        super().hidePopup()
        self.popup_open = False
        self._just_closed = True
        QTimer.singleShot(0, self._forget_popup)

    def _forget_popup(self):
        self._just_closed = False


class EffectListDialog(QDialog):
    """Tick the effects to show in the Effect list."""

    def __init__(self, shown: Iterable[str], show_all: bool, display_names: Dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Effects in the list")
        self.resize(460, 620)
        shown = set(shown)
        lay = QVBoxLayout(self)
        intro = QLabel("Tick the effects you want in the Effect list. Unticked effects are only hidden: "
                       "tick them again here, or use Show all effects, to bring them back.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self._items: Dict[str, QTreeWidgetItem] = {}
        for category, names in effect_categories():
            parent_item = QTreeWidgetItem(self.tree, [category])
            parent_item.setFlags(parent_item.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
            font = parent_item.font(0)
            font.setBold(True)
            parent_item.setFont(0, font)
            for name in names:
                item = QTreeWidgetItem(parent_item, [display_names.get(name, name)])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if name in shown else Qt.Unchecked)
                item.setToolTip(0, PROCESSORS[name].description)
                item.setData(0, Qt.UserRole, name)
                self._items[name] = item
        self.tree.expandAll()
        self.tree.itemChanged.connect(self._update_ok)
        lay.addWidget(self.tree, 1)

        row = QHBoxLayout()
        tick_all = QPushButton("Tick all")
        tick_all.clicked.connect(lambda: self.set_ticked(PROCESSORS))
        untick_all = QPushButton("Untick all")
        untick_all.clicked.connect(lambda: self.set_ticked(()))
        row.addWidget(tick_all)
        row.addWidget(untick_all)
        row.addStretch(1)
        lay.addLayout(row)

        self.show_all_check = QCheckBox("Show all effects (keep the ticks for later)")
        self.show_all_check.setToolTip("Puts every effect back in the list without losing the ticks above. "
                                       "Also in the Settings menu.")
        self.show_all_check.setChecked(show_all)
        self.show_all_check.toggled.connect(self._update_ok)
        lay.addWidget(self.show_all_check)

        self.hint = QLabel()
        self.hint.setStyleSheet("color: palette(mid);")
        lay.addWidget(self.hint)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._update_ok()

    def set_ticked(self, names: Sequence[str]) -> None:
        names = set(names)
        for name, item in self._items.items():
            item.setCheckState(0, Qt.Checked if name in names else Qt.Unchecked)

    def ticked(self) -> List[str]:
        """Ticked effects, in list order."""
        return [name for name, item in self._items.items() if item.checkState(0) == Qt.Checked]

    def show_all(self) -> bool:
        return self.show_all_check.isChecked()

    def _update_ok(self, *_):
        n = len(self.ticked())
        ok = n > 0 or self.show_all()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        self.hint.setText(f"{n} of {len(self._items)} effects ticked" if ok else "Tick at least one effect.")


def parse_shown(value: Optional[str]) -> Optional[List[str]]:
    """Saved setting -> list of known effect names (None if never set)."""
    if not value:
        return None
    names = [n for n in str(value).split(",") if n in PROCESSORS]
    return names or None
