from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from .groups import FXGroupWidget, LightGroupWidget, ColorGroupWidget, ToneGroupWidget, GeometryGroupWidget


class AdjustmentsPanel(QWidget):
    """
    Right-side adjustments panel with:
    - Group icon bar (Light, Color, Tone, Geometry, FX)
    - Group stack (one widget per group)
    """

    def __init__(self, mw, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mw = mw
        self.setObjectName("adjustmentsRoot")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        group_bar = QWidget()
        group_bar.setObjectName("adjustIconBar")
        gbox = QHBoxLayout(group_bar)
        gbox.setContentsMargins(8, 6, 8, 6)
        gbox.setSpacing(8)

        self.group_stack = QStackedWidget()
        self.group_stack.setObjectName("adjustStack")
        self._group_buttons: list[QToolButton] = []
        self._group_button_group = QButtonGroup(self)
        self._group_button_group.setExclusive(True)
        self._group_button_group.idClicked.connect(self.group_stack.setCurrentIndex)

        root.addWidget(group_bar, 0)
        self._gradient_strip = QFrame()
        self._gradient_strip.setObjectName("adjustGradientStrip")
        self._gradient_strip.setFixedHeight(4)
        self._gradient_strip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grad_style = (
            "border-radius: 2px;"
            "margin: 0 12px;"
            "background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0,"
            "stop:0 #1E6DFF, stop:0.25 #5B3CFF, stop:0.5 #7C2CFF, stop:0.75 #C12AFF, stop:1 #FF7A2F"
            ");"
        )
        self._gradient_strip.setStyleSheet(grad_style)
        root.addWidget(self._gradient_strip, 0)
        root.addWidget(self.group_stack, 1)

        self._init_groups(gbox)

    def _init_groups(self, group_bar_layout: QHBoxLayout):
        self.light_group = LightGroupWidget(self._mw)
        self.color_group = ColorGroupWidget(self._mw)
        self.tone_group = ToneGroupWidget(self._mw)
        self.geometry_group = GeometryGroupWidget(self._mw)
        self.fx_group = FXGroupWidget(self._mw)

        groups = [
            (self.light_group, "Light", "icons/light.png"),
            (self.color_group, "Color", "icons/color.png"),
            (self.tone_group, "Detail", "icons/detail.png"),
            (self.geometry_group, "Geometry", "icons/geometry.png"),
            (self.fx_group, "FX", "icons/fx.png"),
        ]

        icon_size = QSize(32, 32)
        for idx, (widget, label, icon_path) in enumerate(groups):
            self.group_stack.addWidget(widget)

            btn = QToolButton()
            btn.setCheckable(True)
            btn.setToolTip(label)
            btn.setAutoRaise(True)
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(icon_size)
            btn.setText("")
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setFixedHeight(32)
            group_bar_layout.addWidget(btn)

            self._group_button_group.addButton(btn, idx)
            self._group_buttons.append(btn)

        if self._group_buttons:
            self._group_buttons[0].setChecked(True)
            self.group_stack.setCurrentIndex(0)


__all__ = ["AdjustmentsPanel"]
