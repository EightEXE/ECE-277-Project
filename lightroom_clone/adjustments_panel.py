import os
from typing import Optional

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .groups import FXGroupWidget, GeometryGroupWidget, LightGroupWidget, ColorGroupWidget, ToneGroupWidget


class AdjustmentsPanel(QWidget):
    """
    Right-side adjustments panel with:
    - Group icon bar (Light, Color, Tone, Geometry, FX)
    - Group stack (one widget per group)
    """

    def __init__(self, mw, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mw = mw

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        group_bar = QWidget()
        group_bar.setObjectName("adjustIconBar")
        gbox = QHBoxLayout(group_bar)
        gbox.setContentsMargins(4, 2, 4, 2)
        gbox.setSpacing(4)

        self.group_stack = QStackedWidget()
        self._group_buttons: list[QToolButton] = []
        self._group_button_group = QButtonGroup(self)
        self._group_button_group.setExclusive(True)
        self._group_button_group.idClicked.connect(self.group_stack.setCurrentIndex)

        root.addWidget(group_bar, 0)
        root.addWidget(self.group_stack, 1)

        for btn in self._group_buttons:
            btn.setIconSize(QSize(20, 20))
            btn.setFixedSize(28, 28)
        gbox.setContentsMargins(2, 2, 2, 2)
        gbox.setSpacing(2)

        self._init_groups(gbox)

    def _init_groups(self, group_bar_layout: QHBoxLayout):
        def load_icon(name: str, fallback_role=None):
            icons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")
            path = os.path.join(icons_dir, name)
            if os.path.exists(path):
                return QIcon(path)
            if fallback_role is not None:
                return self.style().standardIcon(fallback_role)
            return QIcon()

        self.light_group = LightGroupWidget(self._mw)
        self.color_group = ColorGroupWidget(self._mw)
        self.tone_group = ToneGroupWidget(self._mw)
        self.geometry_group = GeometryGroupWidget(self._mw)
        self.fx_group = FXGroupWidget(self._mw)

        groups = [
            (self.light_group, "Light", "light.png", QStyle.SP_DialogYesButton),
            (self.color_group, "Color", "color.png", QStyle.SP_DialogOpenButton),
            (self.tone_group, "Tone", "detail.png", QStyle.SP_TitleBarShadeButton),
            (self.geometry_group, "Geometry", "geometry.png", QStyle.SP_ArrowUp),
            (self.fx_group, "FX", "fx.png", QStyle.SP_BrowserReload),
        ]

        for idx, (widget, label, icon_name, fallback) in enumerate(groups):
            self.group_stack.addWidget(widget)

            btn = QToolButton()
            btn.setCheckable(True)
            btn.setToolTip(label)
            btn.setAutoRaise(True)
            btn.setIconSize(QSize(24, 24))
            btn.setIcon(load_icon(icon_name, fallback))
            group_bar_layout.addWidget(btn)

            self._group_button_group.addButton(btn, idx)
            self._group_buttons.append(btn)

        if self._group_buttons:
            self._group_buttons[0].setChecked(True)
            self.group_stack.setCurrentIndex(0)


__all__ = ["AdjustmentsPanel"]
