import base64
import copy
from io import BytesIO
import numpy as np
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabBar,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
    QToolButton,
    QButtonGroup,
    QSizePolicy,
    QStyle,
)
from PySide6.QtGui import QIcon, QColor, QPainter, QPen, QBrush, QLinearGradient, QPixmap
from PIL import Image

from .widgets import CollapsibleSection
from .color_tools import (
    COLOR_NAMES,
    default_color_params,
    merge_color_params,
    is_section_default,
)

if TYPE_CHECKING:
    from .main_window import MainWindow


class BaseGroupWidget(QWidget):
    """
    Group widget with its own top tab bar and central stacked widget.
    All tabs share the outer panel header/footer from AdjustmentsPanel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._tab_stack = QStackedWidget()
        self._tab_bar = QTabBar()
        self._tab_bar.setExpanding(False)
        self._tab_bar.setDrawBase(False)

        self._tab_bar.currentChanged.connect(self._tab_stack.setCurrentIndex)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._tab_bar, 0)
        layout.addWidget(self._tab_stack, 1)

    def add_tab(self, widget: QWidget, title: str):
        idx = self._tab_stack.addWidget(widget)
        self._tab_bar.addTab(title)
        if self._tab_bar.currentIndex() == -1:
            self._tab_bar.setCurrentIndex(idx)
            self._tab_stack.setCurrentIndex(idx)


def _gradient_role_from_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    lower = label.lower()
    if "white" in lower and "level" in lower:
        return "white_level"
    if "black" in lower and "level" in lower:
        return "black_level"
    if "shadow" in lower:
        return "shadows"
    if "highlight" in lower:
        return "highlights"
    if "temperature" in lower:
        return "temperature"
    if "tint" in lower:
        return "tint"
    if "saturation" in lower:
        return "saturation"
    if "vibrance" in lower:
        return "vibrance"
    if "exposure" in lower:
        return "exposure"
    if "contrast" in lower:
        return "contrast"
    return None


def _apply_slider_gradient_role(slider: QSlider, label: Optional[str], override: Optional[str] = None):
    role = override or _gradient_role_from_label(label)
    if role:
        slider.setProperty("gradientRole", role)


class LightGroupWidget(QWidget):
    """
    GROUP 1 — LIGHT
    Implemented as vertical collapsible sections:
      - Levels
      - White Balance
      - Brightness / Contrast
      - Exposure
      - Shadows / Highlights
      - Vibrance
      - Posterize
    """

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw

        self._levels_state: dict[tuple[str, str], dict] = {}
        self._levels_target: tuple[str, str] | None = None
        self._mask_name = "Global"
        self._defaults = dict(self._mw._default_params_template)
        self._effect_labels = {
            "exposure": "Exposure",
            "contrast": "Contrast",
            "highlights": "Highlights",
            "shadows": "Shadows",
            "whites": "Whites",
            "blacks": "Blacks",
            "temperature": "White Balance",
            "tint": "Tint",
            "vibrance": "Vibrance",
            "saturation": "Saturation",
        }
        self._levels_channel_options = {
            "RGB": ["Master", "Red", "Green", "Blue", "Alpha"],
            "Red": ["Master", "Red"],
            "Green": ["Master", "Green"],
            "Blue": ["Master", "Blue"],
            "Alpha": ["Master", "Alpha"],
        }
        self._levels_defaults = {
            "levels_black": 0,
            "levels_white": 100,
            "levels_gamma": 1.0,
            "levels_out_black": 0,
            "levels_out_white": 100,
            "levels_linear": False,
        }

        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        levels_section = CollapsibleSection("Levels", self, start_collapsed=False)
        levels_section.content_layout().addWidget(self._build_levels_panel())
        v.addWidget(levels_section)

        wb_section = CollapsibleSection("White Balance", self, start_collapsed=False)
        wb_section.content_layout().addWidget(self._build_white_balance_panel())
        v.addWidget(wb_section)

        bc_section = CollapsibleSection("Brightness / Contrast", self, start_collapsed=True)
        bc_section.content_layout().addWidget(self._build_bright_contrast_panel())
        v.addWidget(bc_section)

        exp_section = CollapsibleSection("Exposure", self, start_collapsed=True)
        exp_section.content_layout().addWidget(self._build_exposure_panel())
        v.addWidget(exp_section)

        sh_section = CollapsibleSection("Shadows / Highlights", self, start_collapsed=True)
        sh_section.content_layout().addWidget(self._build_shadows_highlights_panel())
        v.addWidget(sh_section)

        vib_section = CollapsibleSection("Vibrance", self, start_collapsed=True)
        vib_section.content_layout().addWidget(self._build_vibrance_panel())
        v.addWidget(vib_section)

        post_section = CollapsibleSection("Posterize", self, start_collapsed=True)
        post_section.content_layout().addWidget(self._build_posterize_panel())
        v.addWidget(post_section)

        v.addStretch(1)
        scroll.setWidget(container)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)

    # ---------- shared helpers ----------

    def _emit_render(self):
        """Apply current params to the image whenever a slider moves."""
        if self._mw._preview_base_linear is None:
            return

        try:
            if hasattr(self._mw, "_on_edit_params_changed"):
                self._mw._on_edit_params_changed()
            elif hasattr(self._mw, "_schedule_render"):
                self._mw._schedule_render()
        except Exception as exc:
            print("[LightGroupWidget] render error:", exc)

    def _effect_label(self, key: str) -> str:
        return self._effect_labels.get(key, key.replace("_", " ").title())

    def _compact_slider(self, slider: QSlider):
        """Tighten vertical footprint and set a modest minimum width."""
        slider.setFixedHeight(10)
        slider.setMinimumWidth(120)
        slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px; margin:0 4px;}"
            "QSlider::handle:horizontal{width:10px; margin:-6px 0;}"
        )

    def _record_effect(self, param_key: str, value):
        params = {param_key: value}
        if param_key == "contrast":
            params["bc_linear"] = bool(self._mw._current_params.get("bc_linear", False))

        default_val = self._defaults.get(param_key, 128)
        default_linear = self._defaults.get("bc_linear", False)
        has_change = (value != default_val) or (
            param_key == "contrast" and params.get("bc_linear", False) != default_linear
        )

        effect_key = f"{param_key}:global"
        if not has_change:
            self._mw._remove_effect(effect_key, mask=self._mask_name)
            return

        self._mw._upsert_effect(
            effect_key,
            "mask",
            params,
            mask=self._mask_name,
            label=self._effect_label(param_key),
        )

    def eventFilter(self, obj, event):
        if obj in (getattr(self, "levels_color_model_combo", None), getattr(self, "levels_channel_combo", None)):
            if event.type() == QEvent.FocusOut:
                cm = self.levels_color_model_combo.currentText()
                ch = self.levels_channel_combo.currentText()
                if cm != "RGB" and ch != "Master":
                    self.levels_channel_combo.blockSignals(True)
                    self.levels_channel_combo.setCurrentText("Master")
                    self.levels_channel_combo.blockSignals(False)
                    self._on_levels_target_changed()
            return super().eventFilter(obj, event)
        return super().eventFilter(obj, event)

    def _update_param(self, key: str, slider_value: int):
        """
        Map Light panel slider value to the 0–255-ish internal param,
        update MainWindow._current_params, then trigger render.
        """
        if key in {
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "saturation",
            "temperature",
            "tint",
            "vibrance",
        }:
            val = int(slider_value / 100.0 * 255)
        elif key == "exposure":
            val = int(128 + (slider_value / 400.0) * 128)
        else:
            val = slider_value

        self._mw._current_params[key] = val
        self._record_effect(key, val)
        self._emit_render()

    def _collect_levels_params(self) -> dict:
        return {
            "levels_black": self.black_slider.value(),
            "levels_white": self.white_slider.value(),
            "levels_gamma": self.gamma_spin.value(),
            "levels_out_black": self.out_black_slider.value(),
            "levels_out_white": self.out_white_slider.value(),
            "levels_linear": self.linear_checkbox.isChecked(),
            "levels_color_model": self.levels_color_model_combo.currentText(),
            "levels_channel": self.levels_channel_combo.currentText(),
        }

    def _apply_levels_params_to_ui(self, params: dict):
        widgets = [
            self.black_slider,
            self.white_slider,
            self.gamma_spin,
            self.out_black_slider,
            self.out_white_slider,
            self.linear_checkbox,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.black_slider.setValue(int(params.get("levels_black", 0)))
        self.white_slider.setValue(int(params.get("levels_white", 100)))
        self.gamma_spin.setValue(float(params.get("levels_gamma", 1.0)))
        self.out_black_slider.setValue(int(params.get("levels_out_black", 0)))
        self.out_white_slider.setValue(int(params.get("levels_out_white", 100)))
        self.linear_checkbox.setChecked(bool(params.get("levels_linear", params.get("linear", False))))

        for w in widgets:
            w.blockSignals(False)

    def _push_levels_params(self, params: dict):
        cm = params.get("levels_color_model", self.levels_color_model_combo.currentText())
        ch = params.get("levels_channel", self.levels_channel_combo.currentText())
        target = (cm, ch)
        self._levels_state[target] = params

        # Store into main parameter dict so the render engine reads it
        self._mw._current_params.update(params)
        if hasattr(self._mw, "_on_edit_params_changed"):
            self._mw._on_edit_params_changed()
        elif hasattr(self._mw, "_schedule_render"):
            self._mw._schedule_render()
        else:
            self._mw._apply_edit_params_to_current_image(self._mw._current_params)

    def _sync_levels_channel_options(self):
        """Keep channel options consistent with the chosen color model."""
        cm = self.levels_color_model_combo.currentText()
        options = self._levels_channel_options.get(cm, ["Master"])
        current = self.levels_channel_combo.currentText()

        self.levels_channel_combo.blockSignals(True)
        self.levels_channel_combo.clear()
        for opt in options:
            self.levels_channel_combo.addItem(opt)

        if current in options:
            self.levels_channel_combo.setCurrentText(current)
        else:
            self.levels_channel_combo.setCurrentText("Master")
        self.levels_channel_combo.blockSignals(False)

    def _levels_is_default(self, params: dict) -> bool:
        """Check whether the current levels params match defaults."""
        defaults = dict(self._levels_defaults)
        return all(abs(params.get(k, defaults[k]) - defaults[k]) < 1e-6 for k in defaults)

    def _maybe_set_levels_effect(self, cm: str, ch: str, params: dict):
        """Only add a Levels effect if it differs from defaults; otherwise remove it."""
        effect_key = f"levels:{cm}:{ch}"
        if self._levels_is_default(params):
            self._mw._remove_effect(effect_key, mask=self._mask_name)
            return

        self._mw._upsert_effect(
            effect_key,
            "levels",
            dict(params),
            mask=self._mask_name,
            label=f"Levels ({cm}/{ch})",
        )

    def reset_ui_to_defaults(self):
        """Neutralize sliders without emitting signals when switching images."""
        widgets = [
            getattr(self, "wb_temp_slider", None),
            getattr(self, "wb_tint_slider", None),
            getattr(self, "bc_linear_checkbox", None),
            getattr(self, "black_slider", None),
            getattr(self, "white_slider", None),
            getattr(self, "gamma_slider", None),
            getattr(self, "out_black_slider", None),
            getattr(self, "out_white_slider", None),
            getattr(self, "linear_checkbox", None),
            getattr(self, "levels_color_model_combo", None),
            getattr(self, "levels_channel_combo", None),
            getattr(self, "exposure_slider", None),
            getattr(self, "shadows_slider", None),
            getattr(self, "highlights_slider", None),
            getattr(self, "whites_slider", None),
            getattr(self, "blacks_slider", None),
            getattr(self, "vibrance_slider", None),
            getattr(self, "saturation_slider", None),
            getattr(self, "brightness_slider", None),
            getattr(self, "contrast_slider", None),
            getattr(self, "shadows_spin", None),
            getattr(self, "highlights_spin", None),
            getattr(self, "brightness_spin", None),
            getattr(self, "contrast_spin", None),
            getattr(self, "vibrance_spin", None),
            getattr(self, "saturation_spin", None),
        ]
        for w in widgets:
            if w is not None:
                w.blockSignals(True)

        # White balance
        if getattr(self, "wb_temp_slider", None):
            self.wb_temp_slider.setValue(50)
        if getattr(self, "wb_tint_slider", None):
            self.wb_tint_slider.setValue(50)

        # Brightness / Contrast
        if getattr(self, "bc_linear_checkbox", None):
            self.bc_linear_checkbox.setChecked(False)
        # Sliders are reset below via helper

        # Exposure
        if getattr(self, "exposure_slider", None):
            self.exposure_slider.setValue(0)

        # Shadows / Highlights / Whites / Blacks / Vibrance / Saturation
        defaults = [
            ("shadows_slider", 50),
            ("highlights_slider", 50),
            ("whites_slider", 50),
            ("blacks_slider", 50),
            ("vibrance_slider", 50),
            ("saturation_slider", 50),
            ("contrast_slider", 50),
            ("brightness_slider", 50),
        ]
        for name, val in defaults:
            sl = getattr(self, name, None)
            if sl is not None:
                sl.setValue(val)

        # Levels
        if getattr(self, "black_slider", None):
            self.black_slider.setValue(0)
        if getattr(self, "white_slider", None):
            self.white_slider.setValue(100)
        if getattr(self, "gamma_slider", None):
            self.gamma_slider.setValue(100)
        if getattr(self, "out_black_slider", None):
            self.out_black_slider.setValue(0)
        if getattr(self, "out_white_slider", None):
            self.out_white_slider.setValue(100)
        if getattr(self, "linear_checkbox", None):
            self.linear_checkbox.setChecked(False)
        if getattr(self, "levels_color_model_combo", None):
            self.levels_color_model_combo.setCurrentText("RGB")
        if getattr(self, "levels_channel_combo", None):
            self.levels_channel_combo.setCurrentText("Master")
        self._levels_state.clear()

        for w in widgets:
            if w is not None:
                w.blockSignals(False)

    def capture_ui_state(self) -> dict:
        """Snapshot current slider/checkbox/combo values without changing them."""
        def val(widget, getter):
            return getter(widget) if widget is not None else None

        return {
            "wb_temp_slider": val(getattr(self, "wb_temp_slider", None), lambda w: w.value()),
            "wb_tint_slider": val(getattr(self, "wb_tint_slider", None), lambda w: w.value()),
            "bc_linear_checkbox": val(getattr(self, "bc_linear_checkbox", None), lambda w: w.isChecked()),
            "black_slider": val(getattr(self, "black_slider", None), lambda w: w.value()),
            "white_slider": val(getattr(self, "white_slider", None), lambda w: w.value()),
            "gamma_slider": val(getattr(self, "gamma_slider", None), lambda w: w.value()),
            "out_black_slider": val(getattr(self, "out_black_slider", None), lambda w: w.value()),
            "out_white_slider": val(getattr(self, "out_white_slider", None), lambda w: w.value()),
            "linear_checkbox": val(getattr(self, "linear_checkbox", None), lambda w: w.isChecked()),
            "levels_color_model_combo": val(getattr(self, "levels_color_model_combo", None), lambda w: w.currentText()),
            "levels_channel_combo": val(getattr(self, "levels_channel_combo", None), lambda w: w.currentText()),
            "exposure_slider": val(getattr(self, "exposure_slider", None), lambda w: w.value()),
            "shadows_slider": val(getattr(self, "shadows_slider", None), lambda w: w.value()),
            "highlights_slider": val(getattr(self, "highlights_slider", None), lambda w: w.value()),
            "whites_slider": val(getattr(self, "whites_slider", None), lambda w: w.value()),
            "blacks_slider": val(getattr(self, "blacks_slider", None), lambda w: w.value()),
            "vibrance_slider": val(getattr(self, "vibrance_slider", None), lambda w: w.value()),
            "saturation_slider": val(getattr(self, "saturation_slider", None), lambda w: w.value()),
            "brightness_slider": val(getattr(self, "brightness_slider", None), lambda w: w.value()),
            "contrast_slider": val(getattr(self, "contrast_slider", None), lambda w: w.value()),
            "shadows_spin": val(getattr(self, "shadows_spin", None), lambda w: w.value()),
            "highlights_spin": val(getattr(self, "highlights_spin", None), lambda w: w.value()),
            "brightness_spin": val(getattr(self, "brightness_spin", None), lambda w: w.value()),
            "contrast_spin": val(getattr(self, "contrast_spin", None), lambda w: w.value()),
            "vibrance_spin": val(getattr(self, "vibrance_spin", None), lambda w: w.value()),
            "saturation_spin": val(getattr(self, "saturation_spin", None), lambda w: w.value()),
        }

    def restore_ui_state(self, state: dict | None):
        """Restore slider/checkbox/combo values without emitting signals."""
        if not state:
            return

        widgets = {
            "wb_temp_slider": getattr(self, "wb_temp_slider", None),
            "wb_tint_slider": getattr(self, "wb_tint_slider", None),
            "bc_linear_checkbox": getattr(self, "bc_linear_checkbox", None),
            "black_slider": getattr(self, "black_slider", None),
            "white_slider": getattr(self, "white_slider", None),
            "gamma_slider": getattr(self, "gamma_slider", None),
            "out_black_slider": getattr(self, "out_black_slider", None),
            "out_white_slider": getattr(self, "out_white_slider", None),
            "linear_checkbox": getattr(self, "linear_checkbox", None),
            "levels_color_model_combo": getattr(self, "levels_color_model_combo", None),
            "levels_channel_combo": getattr(self, "levels_channel_combo", None),
            "exposure_slider": getattr(self, "exposure_slider", None),
            "shadows_slider": getattr(self, "shadows_slider", None),
            "highlights_slider": getattr(self, "highlights_slider", None),
            "whites_slider": getattr(self, "whites_slider", None),
            "blacks_slider": getattr(self, "blacks_slider", None),
            "vibrance_slider": getattr(self, "vibrance_slider", None),
            "saturation_slider": getattr(self, "saturation_slider", None),
            "brightness_slider": getattr(self, "brightness_slider", None),
            "contrast_slider": getattr(self, "contrast_slider", None),
            "shadows_spin": getattr(self, "shadows_spin", None),
            "highlights_spin": getattr(self, "highlights_spin", None),
            "brightness_spin": getattr(self, "brightness_spin", None),
            "contrast_spin": getattr(self, "contrast_spin", None),
            "vibrance_spin": getattr(self, "vibrance_spin", None),
            "saturation_spin": getattr(self, "saturation_spin", None),
        }

        for name, widget in widgets.items():
            if widget is None or name not in state:
                continue
            widget.blockSignals(True)
            val = state[name]
            if val is None:
                widget.blockSignals(False)
                continue
            if hasattr(widget, "setValue"):
                widget.setValue(val)
            elif hasattr(widget, "setChecked"):
                widget.setChecked(bool(val))
            elif hasattr(widget, "setCurrentText"):
                widget.setCurrentText(str(val))
            widget.blockSignals(False)

        # Keep paired spin/slider in sync visually even while signals are blocked
        pairs = [
            ("shadows_slider", "shadows_spin"),
            ("highlights_slider", "highlights_spin"),
            ("brightness_slider", "brightness_spin"),
            ("contrast_slider", "contrast_spin"),
            ("vibrance_slider", "vibrance_spin"),
            ("saturation_slider", "saturation_spin"),
        ]
        for slider_name, spin_name in pairs:
            sld = widgets.get(slider_name)
            spn = widgets.get(spin_name)
            if sld is None or spn is None:
                continue
            sld.blockSignals(True)
            spn.blockSignals(True)
            val = state.get(slider_name, state.get(spin_name))
            if val is not None:
                sld.setValue(val)
                spn.setValue(val)
            sld.blockSignals(False)
            spn.blockSignals(False)

    # ---------- Levels panel ----------

    def _on_levels_target_changed(self):
        cm = self.levels_color_model_combo.currentText()
        self._sync_levels_channel_options()
        ch = self.levels_channel_combo.currentText()
        self._levels_target = (cm, ch)

        params = self._levels_state.get(self._levels_target, self._collect_levels_params())
        params["levels_color_model"] = cm
        params["levels_channel"] = ch

        self._apply_levels_params_to_ui(params)
        self._push_levels_params(params)
        self._maybe_set_levels_effect(cm, ch, params)
        self._emit_render()

    def _update_levels_param(self, key: str, value):
        cm = self.levels_color_model_combo.currentText()
        ch = self.levels_channel_combo.currentText()
        target = (cm, ch)
        params = self._levels_state.get(target, self._collect_levels_params())
        if "linear" in params and "levels_linear" not in params:
            params["levels_linear"] = params.pop("linear")

        params[key] = value
        params["levels_color_model"] = cm
        params["levels_channel"] = ch

        self._push_levels_params(params)
        self._maybe_set_levels_effect(cm, ch, params)
        self._emit_render()

    def _build_levels_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        row = QHBoxLayout()
        self.levels_color_model_combo = QComboBox()
        self.levels_color_model_combo.addItems(["RGB", "Red", "Green", "Blue", "Alpha"])
        self.levels_channel_combo = QComboBox()
        self.levels_channel_combo.addItems(["Master", "Red", "Green", "Blue", "Alpha"])
        row.addWidget(QLabel("Color Model"))
        row.addWidget(self.levels_color_model_combo)
        row.addSpacing(8)
        row.addWidget(QLabel("Channel"))
        row.addWidget(self.levels_channel_combo)
        row.addStretch(1)
        layout.addLayout(row)

        self.levels_color_model_combo.currentTextChanged.connect(self._on_levels_target_changed)
        self.levels_channel_combo.currentTextChanged.connect(self._on_levels_target_changed)
        self.levels_color_model_combo.installEventFilter(self)
        self.levels_channel_combo.installEventFilter(self)

        self._sync_levels_channel_options()

        def make_levels_row(label_text: str, param_key: str, is_gamma=False):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            slider = QSlider(Qt.Horizontal)
            _apply_slider_gradient_role(
                slider,
                label_text,
                override="white_level" if is_gamma else None,
            )
            self._compact_slider(slider)

            if is_gamma:
                slider.setRange(10, 300)  # maps to 0.10–3.00
                value_edit = QDoubleSpinBox()
                value_edit.setRange(0.10, 3.00)
                value_edit.setSingleStep(0.05)
                value_edit.setDecimals(2)
                value_edit.setValue(1.00)
                value_edit.setButtonSymbols(QDoubleSpinBox.NoButtons)
                slider.setValue(100)
            else:
                slider.setRange(0, 100)
                value_edit = QSpinBox()
                value_edit.setRange(0, 100)
                if "Black" in label_text and "Output" not in label_text:
                    value_edit.setValue(0)
                    slider.setValue(0)
                elif "White" in label_text and "Output" not in label_text:
                    value_edit.setValue(100)
                    slider.setValue(100)
                elif "Output Black" in label_text:
                    value_edit.setValue(0)
                    slider.setValue(0)
                elif "Output White" in label_text:
                    value_edit.setValue(100)
                    slider.setValue(100)
                else:
                    value_edit.setValue(0)
                    slider.setValue(0)

                value_edit.setFixedWidth(64)
                value_edit.setButtonSymbols(QSpinBox.NoButtons)

            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(value_edit)

            if is_gamma:
                slider.valueChanged.connect(lambda v, ed=value_edit: ed.setValue(v / 100.0))

                def on_gamma_changed(val, key=param_key):
                    self._update_levels_param(key, float(val))

                value_edit.valueChanged.connect(on_gamma_changed)
            else:
                slider.valueChanged.connect(value_edit.setValue)

                def on_slider_changed(v, key=param_key):
                    self._update_levels_param(key, int(v))

                slider.valueChanged.connect(on_slider_changed)

            return slider, value_edit, row

        self.black_slider, self.black_spin, black_row = make_levels_row("Black Level", "levels_black")
        self.white_slider, self.white_spin, white_row = make_levels_row("White Level", "levels_white")
        self.gamma_slider, self.gamma_spin, gamma_row = make_levels_row("Gamma", "levels_gamma", is_gamma=True)
        self.out_black_slider, self.out_black_spin, out_black_row = make_levels_row(
            "Output Black Level", "levels_out_black"
        )
        self.out_white_slider, self.out_white_spin, out_white_row = make_levels_row(
            "Output White Level", "levels_out_white"
        )

        layout.addLayout(black_row)
        layout.addLayout(white_row)
        layout.addLayout(gamma_row)
        layout.addLayout(out_black_row)
        layout.addLayout(out_white_row)

        self.linear_checkbox = QCheckBox("Use linear curve")
        self.linear_checkbox.toggled.connect(
            lambda checked: self._update_levels_param("levels_linear", bool(checked))
        )
        layout.addWidget(self.linear_checkbox, 0, Qt.AlignLeft)

        layout.addStretch(1)
        self._on_levels_target_changed()
        return page

    # ---------- Shadows / Highlights panel ----------

    def _build_shadows_highlights_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        def make_row(name: str, param_key: str):
            row = QHBoxLayout()
            lbl = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            _apply_slider_gradient_role(slider, name)
            slider.setRange(0, 100)
            self._compact_slider(slider)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            spin.setFixedWidth(56)
            spin.setButtonSymbols(QSpinBox.NoButtons)

            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda v, key=param_key: self._update_param(key, v))

            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            return slider, spin, row

        self.shadows_slider, self.shadows_spin, sh_row = make_row("Shadows", "shadows")
        layout.addLayout(sh_row)
        self.highlights_slider, self.highlights_spin, hi_row = make_row("Highlights", "highlights")
        layout.addLayout(hi_row)

        return page

    # ---------- White Balance panel ----------

    def _build_white_balance_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        def make_row(label_text: str, param_key: str, override: Optional[str] = None):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            self._compact_slider(slider)
            _apply_slider_gradient_role(slider, label_text, override=override)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            spin.setFixedWidth(56)
            spin.setButtonSymbols(QSpinBox.NoButtons)

            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda v, key=param_key: self._update_param(key, v))

            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            return slider, row

        self.wb_temp_slider, temp_row = make_row("White Balance", "temperature", override="white_level")
        self.wb_tint_slider, tint_row = make_row("Tint", "tint")

        layout.addLayout(temp_row)
        layout.addLayout(tint_row)

        self.wb_picker_btn = QPushButton("Picker")
        self.wb_picker_btn.clicked.connect(self._on_wb_picker_clicked)
        layout.addWidget(self.wb_picker_btn, 0, Qt.AlignLeft)

        return page

    # ---------- Brightness / Contrast panel ----------

    def _build_bright_contrast_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        def make_row(name: str, param_key: str):
            row = QHBoxLayout()
            lbl = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            _apply_slider_gradient_role(slider, name)
            slider.setRange(0, 100)
            self._compact_slider(slider)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            slider.setValue(50)
            spin.setFixedWidth(56)
            spin.setButtonSymbols(QSpinBox.NoButtons)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)

            def on_slider_changed(v, key=param_key):
                val_255 = int(v / 100.0 * 255)
                self._mw._current_params[key] = val_255
                self._record_effect(key, val_255)
                self._emit_render()

            slider.valueChanged.connect(on_slider_changed)

            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            return slider, spin, row

        self.brightness_slider, self.brightness_spin, b_row = make_row("Brightness", "exposure")
        layout.addLayout(b_row)
        self.contrast_slider, self.contrast_spin, c_row = make_row("Contrast", "contrast")
        layout.addLayout(c_row)

        self.bc_linear_checkbox = QCheckBox("Linear")
        layout.addWidget(self.bc_linear_checkbox, 0, Qt.AlignLeft)

        def on_bc_linear_toggled(checked: bool):
            self._mw._current_params["bc_linear"] = bool(checked)
            self._record_effect("contrast", self._mw._current_params.get("contrast", 128))
            self._emit_render()

        self.bc_linear_checkbox.toggled.connect(on_bc_linear_toggled)

        return page

    # ---------- Exposure panel ----------

    def _build_exposure_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        row = QHBoxLayout()
        lbl = QLabel("Exposure")
        slider = QSlider(Qt.Horizontal)
        _apply_slider_gradient_role(slider, "Exposure")
        slider.setRange(-400, 400)  # -4 to +4 EV
        self._compact_slider(slider)
        spin = QDoubleSpinBox()
        spin.setRange(-4.0, 4.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.10)
        spin.setSuffix(" EV")
        spin.setFixedWidth(80)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)

        slider.valueChanged.connect(lambda v: spin.setValue(v / 100.0))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * 100)))
        slider.valueChanged.connect(lambda v: self._update_param("exposure", v))

        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        layout.addLayout(row)
        self.exposure_slider = slider

        return page

    # ---------- Vibrance panel ----------

    def _build_vibrance_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        def make_row(name: str, param_key: str):
            row = QHBoxLayout()
            lbl = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            _apply_slider_gradient_role(slider, name)
            slider.setRange(0, 100)
            self._compact_slider(slider)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            spin.setFixedWidth(56)
            spin.setButtonSymbols(QSpinBox.NoButtons)

            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda v, key=param_key: self._update_param(key, v))

            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            return slider, spin, row

        self.vibrance_slider, self.vibrance_spin, v_row = make_row("Vibrance", "vibrance")
        layout.addLayout(v_row)
        self.saturation_slider, self.saturation_spin, s_row = make_row("Saturation", "saturation")
        layout.addLayout(s_row)

        return page

    # ---------- Posterize panel ----------

    def _build_posterize_panel(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        row = QHBoxLayout()
        lbl = QLabel("Posterize Levels")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(2, 64)
        spin = QSpinBox()
        spin.setRange(2, 64)
        spin.setValue(4)
        spin.setFixedWidth(60)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)

        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        layout.addLayout(row)

        return page

    # ---------- WB picker ----------

    def _on_wb_picker_clicked(self):
        """Open a color dialog and map chosen color to Temp/Tint sliders."""
        color = self._mw.pick_color(self)
        if color is None:
            return

        r = color.red() / 255.0
        g = color.green() / 255.0
        b = color.blue() / 255.0

        warm = r - b
        tint = g - (r + b) * 0.5

        def clamp_slider(x):
            return max(0, min(100, int(round(x))))

        temp_slider_val = clamp_slider(50 + warm * 40.0)
        tint_slider_val = clamp_slider(50 + tint * 40.0)

        self.wb_temp_slider.setValue(temp_slider_val)
        self.wb_tint_slider.setValue(tint_slider_val)


class HSLTabWidget(QWidget):
    """HSL controls using the same slider style as the Light sections."""

    def __init__(self, mw: "MainWindow", parent=None, on_change=None):
        super().__init__(parent)
        self._mw = mw
        self._colors = list(COLOR_NAMES)
        self._kinds = ["hue", "sat", "lum"]
        self._on_change = on_change

        hsl_defaults = default_color_params()["hsl"]
        self._defaults = {f"hsl_{kind}_{c}": 0 for kind in self._kinds for c in self._colors}
        self._defaults["enabled"] = hsl_defaults.get("enabled", True)
        self._slider_for_key: dict[str, QSlider] = {}

        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        v.addWidget(self._build_panel("hue", "Hue"))
        v.addWidget(self._build_panel("sat", "Saturation"))
        v.addWidget(self._build_panel("lum", "Luminance"))

        v.addStretch(1)
        scroll.setWidget(container)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)

    # ---------- helpers ----------

    def _color_state(self) -> dict:
        color_state = merge_color_params(self._mw._current_params.get("color"), self._mw._current_params)
        self._mw._current_params["color"] = color_state
        return color_state.get("hsl", {})

    def _compact_slider(self, slider: QSlider):
        slider.setFixedHeight(10)
        slider.setMinimumWidth(120)
        slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px; margin:0 4px;}"
            "QSlider::handle:horizontal{width:12px; height:12px; min-width:12px; min-height:12px; max-width:12px; max-height:12px; margin:-7px 0; border-radius:6px;}"
        )

    def _emit_render(self):
        if self._mw._preview_base_linear is None:
            return
        try:
            if hasattr(self._mw, "_on_edit_params_changed"):
                self._mw._on_edit_params_changed()
            elif hasattr(self._mw, "_schedule_render"):
                self._mw._schedule_render()
            else:
                self._mw._apply_edit_params_to_current_image(self._mw._current_params)
        except Exception as exc:
            print("[HSLTabWidget] render error:", exc)

    def _effect_label(self, param_key: str) -> str:
        try:
            _, kind, color = param_key.split("_", 2)
        except ValueError:
            return param_key
        return f"{kind.title()} – {color.replace('_', ' ').title()}"

    def _update_param(self, param_key: str, value: int):
        # Legacy flat param for backwards compatibility
        self._mw._current_params[param_key] = value

        try:
            _, kind, color = param_key.split("_", 2)
        except ValueError:
            kind = ""
            color = ""

        hsl_state = self._color_state()
        hsl_state.setdefault("enabled", True)
        hsl_state.setdefault("hue", {c: 0.0 for c in self._colors})
        hsl_state.setdefault("sat", {c: 0.0 for c in self._colors})
        hsl_state.setdefault("lum", {c: 0.0 for c in self._colors})
        if kind in {"hue", "sat", "lum"} and color:
            try:
                hsl_state[kind][color] = float(value)
            except Exception:
                pass
        color_state = merge_color_params(self._mw._current_params.get("color"), self._mw._current_params)
        color_state["hsl"] = hsl_state
        self._mw._current_params["color"] = color_state

        if callable(self._on_change):
            self._on_change()
        self._emit_render()

    def _build_panel(self, kind: str, title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        header = QLabel(title)
        header.setStyleSheet("font-weight: 600;")
        layout.addWidget(header)

        def make_row(label_text: str, color_key: str):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            self._compact_slider(slider)

            param_key = f"hsl_{kind}_{color_key}"
            slider.setProperty("gradientRole", f"hsl_{kind}_{color_key}")
            slider.valueChanged.connect(lambda v, key=param_key: self._update_param(key, int(v)))

            row.addWidget(lbl)
            row.addWidget(slider, 1)

            self._slider_for_key[param_key] = slider
            return row

        labels = {
            "red": "Red",
            "orange": "Orange",
            "yellow": "Yellow",
            "green": "Green",
            "aqua": "Aqua",
            "blue": "Blue",
            "purple": "Purple",
            "magenta": "Magenta",
        }

        for color in self._colors:
            layout.addLayout(make_row(labels.get(color, color.title()), color))

        return page

    # ---------- UI state helpers ----------

    def reset_ui_to_defaults(self):
        for key, slider in self._slider_for_key.items():
            slider.blockSignals(True)
            slider.setValue(self._defaults.get(key, 0))
            slider.blockSignals(False)

    def capture_ui_state(self) -> dict:
        return {key: slider.value() for key, slider in self._slider_for_key.items()}

    def restore_ui_state(self, state: dict | None):
        if not state:
            return
        for key, value in state.items():
            slider = self._slider_for_key.get(key)
            if slider is None:
                continue
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)

    def set_enabled(self, enabled: bool):
        for slider in self._slider_for_key.values():
            slider.setEnabled(enabled)


class ColorGroupWidget(QWidget):
    """
    GROUP 2 – COLOR
    Full Color panel with toggles for HSL, Recolor, B&W, Selective Color, Color Balance, and Color WB.
    """

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw
        self._defaults = default_color_params()

        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # HSL
        self.hsl_section = HSLTabWidget(self._mw, self, on_change=self._on_hsl_changed)
        hsl_collapsible = CollapsibleSection("HSL", self, start_collapsed=False)
        hsl_layout = hsl_collapsible.content_layout()
        hsl_row = QHBoxLayout()
        self.hsl_enable = QCheckBox("Enable HSL")
        self.hsl_enable.setChecked(self._defaults["hsl"].get("enabled", True))
        self.hsl_enable.toggled.connect(self._on_hsl_toggled)
        hsl_row.addWidget(self.hsl_enable)
        hsl_row.addStretch(1)
        hsl_layout.addLayout(hsl_row)
        hsl_layout.addWidget(self.hsl_section)
        v.addWidget(hsl_collapsible)

        # Other color sections
        v.addWidget(self._build_recolor_section())
        v.addWidget(self._build_bw_section())
        v.addWidget(self._build_selective_color_section())
        v.addWidget(self._build_color_balance_section())
        v.addWidget(self._build_color_wb_section())

        v.addStretch(1)
        scroll.setWidget(container)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)

    # ---------- UI state helpers ----------

    def reset_ui_to_defaults(self):
        widgets = [
            getattr(self, "hsl_enable", None),
            getattr(self, "recolor_enable", None),
            getattr(self, "bw_enable", None),
            getattr(self, "sel_color_enable", None),
            getattr(self, "cb_enable", None),
            getattr(self, "cwb_enable", None),
        ]
        for w in widgets:
            if isinstance(w, QCheckBox):
                w.blockSignals(True)
        if getattr(self, "hsl_section", None):
            self.hsl_section.reset_ui_to_defaults()
            self.hsl_enable.setChecked(self._defaults["hsl"].get("enabled", True))
        self._apply_default_recolor()
        self._apply_default_bw()
        self._apply_default_selective_color()
        self._apply_default_color_balance()
        self._apply_default_color_wb()
        for w in widgets:
            if isinstance(w, QCheckBox):
                w.blockSignals(False)

    def capture_ui_state(self) -> dict:
        state: dict = {}
        if getattr(self, "hsl_section", None):
            state["hsl"] = self.hsl_section.capture_ui_state()
            state["hsl_enabled"] = self.hsl_enable.isChecked()
        state["recolor"] = {
            "enabled": getattr(self, "recolor_enable", QCheckBox()).isChecked(),
            "target": getattr(self, "recolor_hue_slider", QSlider()).value(),
            "strength": getattr(self, "recolor_strength_slider", QSlider()).value(),
            "saturation": getattr(self, "recolor_sat_slider", QSlider()).value(),
            "preserve": getattr(self, "recolor_preserve", QCheckBox()).isChecked(),
            "mode": getattr(self, "recolor_mode", QComboBox()).currentText(),
        }
        state["bw"] = {
            "enabled": getattr(self, "bw_enable", QCheckBox()).isChecked(),
            "mix": {c: sl.value() for c, sl in getattr(self, "_bw_sliders", {}).items()},
            "contrast": getattr(self, "bw_contrast_slider", QSlider()).value() if hasattr(self, "bw_contrast_slider") else 0,
        }
        state["selective"] = {
            "enabled": getattr(self, "sel_color_enable", QCheckBox()).isChecked(),
            "range": getattr(self, "sel_color_range", QComboBox()).currentText() if hasattr(self, "sel_color_range") else "",
            "mode": getattr(self, "sel_color_mode", QComboBox()).currentText() if hasattr(self, "sel_color_mode") else "",
            "adjust": {k: sl.value() for k, sl in getattr(self, "_sel_sliders", {}).items()},
        }
        state["color_balance"] = {
            "enabled": getattr(self, "cb_enable", QCheckBox()).isChecked(),
            "preserve": getattr(self, "cb_preserve", QCheckBox()).isChecked() if hasattr(self, "cb_preserve") else True,
            "tones": {
                tone: {axis: sl.value() for axis, sl in sliders.items()}
                for tone, sliders in getattr(self, "_cb_sliders", {}).items()
            },
        }
        state["color_wb"] = {
            "enabled": getattr(self, "cwb_enable", QCheckBox()).isChecked(),
            "temp": getattr(self, "cwb_temp_slider", QSlider()).value(),
            "tint": getattr(self, "cwb_tint_slider", QSlider()).value(),
            "preset": getattr(self, "cwb_preset_combo", QComboBox()).currentText() if hasattr(self, "cwb_preset_combo") else "",
        }
        return state

    def restore_ui_state(self, state: dict | None):
        if not state:
            return

        def _block(w, value, setter):
            if w is None:
                return
            w.blockSignals(True)
            setter(value)
            w.blockSignals(False)

        if getattr(self, "hsl_section", None):
            self.hsl_section.restore_ui_state(state.get("hsl", {}))
            _block(self.hsl_enable, state.get("hsl_enabled", True), self.hsl_enable.setChecked)

        rc = state.get("recolor", {})
        _block(getattr(self, "recolor_enable", None), rc.get("enabled", False), getattr(self, "recolor_enable").setChecked if hasattr(self, "recolor_enable") else lambda _: None)
        _block(getattr(self, "recolor_hue_slider", None), rc.get("target", 0), getattr(self, "recolor_hue_slider").setValue if hasattr(self, "recolor_hue_slider") else lambda _: None)
        _block(getattr(self, "recolor_strength_slider", None), rc.get("strength", 0), getattr(self, "recolor_strength_slider").setValue if hasattr(self, "recolor_strength_slider") else lambda _: None)
        _block(getattr(self, "recolor_sat_slider", None), rc.get("saturation", 0), getattr(self, "recolor_sat_slider").setValue if hasattr(self, "recolor_sat_slider") else lambda _: None)
        if hasattr(self, "recolor_preserve"):
            _block(self.recolor_preserve, rc.get("preserve", True), self.recolor_preserve.setChecked)
        if hasattr(self, "recolor_mode") and rc.get("mode"):
            _block(self.recolor_mode, rc["mode"], self.recolor_mode.setCurrentText)

        bw = state.get("bw", {})
        _block(getattr(self, "bw_enable", None), bw.get("enabled", False), getattr(self, "bw_enable").setChecked if hasattr(self, "bw_enable") else lambda _: None)
        for c, sl in getattr(self, "_bw_sliders", {}).items():
            _block(sl, bw.get("mix", {}).get(c, 0), sl.setValue)
        if hasattr(self, "bw_contrast_slider"):
            _block(self.bw_contrast_slider, bw.get("contrast", 0), self.bw_contrast_slider.setValue)

        sel = state.get("selective", {})
        _block(getattr(self, "sel_color_enable", None), sel.get("enabled", False), getattr(self, "sel_color_enable").setChecked if hasattr(self, "sel_color_enable") else lambda _: None)
        if hasattr(self, "sel_color_range") and sel.get("range"):
            _block(self.sel_color_range, sel["range"], self.sel_color_range.setCurrentText)
        if hasattr(self, "sel_color_mode") and sel.get("mode"):
            _block(self.sel_color_mode, sel["mode"], self.sel_color_mode.setCurrentText)
        for k, sl in getattr(self, "_sel_sliders", {}).items():
            _block(sl, sel.get("adjust", {}).get(k, 0), sl.setValue)

        cb = state.get("color_balance", {})
        _block(getattr(self, "cb_enable", None), cb.get("enabled", False), getattr(self, "cb_enable").setChecked if hasattr(self, "cb_enable") else lambda _: None)
        if hasattr(self, "cb_preserve"):
            _block(self.cb_preserve, cb.get("preserve", True), self.cb_preserve.setChecked)
        for tone, sliders in getattr(self, "_cb_sliders", {}).items():
            tone_vals = cb.get("tones", {}).get(tone, {})
            for axis, sl in sliders.items():
                _block(sl, tone_vals.get(axis, 0), sl.setValue)

        wb_state = state.get("color_wb", {})
        _block(getattr(self, "cwb_enable", None), wb_state.get("enabled", False), getattr(self, "cwb_enable").setChecked if hasattr(self, "cwb_enable") else lambda _: None)
        _block(getattr(self, "cwb_temp_slider", None), wb_state.get("temp", 0), getattr(self, "cwb_temp_slider").setValue if hasattr(self, "cwb_temp_slider") else lambda _: None)
        _block(getattr(self, "cwb_tint_slider", None), wb_state.get("tint", 0), getattr(self, "cwb_tint_slider").setValue if hasattr(self, "cwb_tint_slider") else lambda _: None)
        if hasattr(self, "cwb_preset_combo") and wb_state.get("preset"):
            _block(self.cwb_preset_combo, wb_state["preset"], self.cwb_preset_combo.setCurrentText)

        self._on_hsl_changed()
        self._update_recolor_from_ui()
        self._update_bw_from_ui()
        self._update_selective_from_ui()
        self._update_color_balance_from_ui()
        self._update_color_wb_from_ui()

    # ---------- helpers ----------

    def _color_params(self) -> dict:
        state = merge_color_params(self._mw._current_params.get("color"), self._mw._current_params)
        self._mw._current_params["color"] = state
        return state

    def _emit_render(self):
        if self._mw._preview_base_linear is None:
            return
        try:
            if hasattr(self._mw, "_on_edit_params_changed"):
                self._mw._on_edit_params_changed()
            elif hasattr(self._mw, "_schedule_render"):
                self._mw._schedule_render()
            else:
                self._mw._apply_edit_params_to_current_image(self._mw._current_params)
        except Exception as exc:
            print("[ColorGroupWidget] render error:", exc)

    def _sync_section_effect(self, section_key: str, label: str):
        color_state = self._color_params()
        section_state = copy.deepcopy(color_state.get(section_key, {}))
        defaults = self._defaults.get(section_key, {})
        enabled = bool(section_state.get("enabled", False))
        if not enabled or is_section_default(section_state, defaults):
            self._mw._remove_effect(f"color:{section_key}", mask="Global")
            return
        self._mw._upsert_effect(
            f"color:{section_key}",
            "color",
            {"color": {section_key: section_state}},
            mask="Global",
            label=label,
        )

    # ---------- HSL handlers ----------

    def _on_hsl_toggled(self, checked: bool):
        state = self._color_params()
        hsl = state.get("hsl", {})
        hsl["enabled"] = bool(checked)
        state["hsl"] = hsl
        self._mw._current_params["color"] = state
        if hasattr(self, "hsl_section"):
            self.hsl_section.set_enabled(checked)
        self._sync_section_effect("hsl", "HSL")
        self._emit_render()

    def _on_hsl_changed(self):
        color_state = self._color_params()
        hsl_state = color_state.get("hsl", {})
        hsl_state["enabled"] = self.hsl_enable.isChecked()
        color_state["hsl"] = hsl_state
        self._mw._current_params["color"] = color_state
        self._sync_section_effect("hsl", "HSL")

    # ---------- Recolor section ----------

    def _build_recolor_section(self) -> CollapsibleSection:
        sect = CollapsibleSection("Recolor", self, start_collapsed=True)
        layout = sect.content_layout()
        self.recolor_enable = QCheckBox("Enable Recolor")
        self.recolor_enable.setChecked(False)
        self.recolor_enable.toggled.connect(self._update_recolor_from_ui)
        layout.addWidget(self.recolor_enable, 0, Qt.AlignLeft)

        self.recolor_hue_slider, self.recolor_hue_spin = self._make_slider_row(
            "Target Hue", 0, 360, 0, layout, unit="°"
        )
        self.recolor_strength_slider, self.recolor_strength_spin = self._make_slider_row(
            "Strength", 0, 100, 0, layout
        )
        self.recolor_sat_slider, self.recolor_sat_spin = self._make_slider_row(
            "Saturation", -100, 100, 0, layout
        )
        self.recolor_preserve = QCheckBox("Preserve Luminance")
        self.recolor_preserve.setChecked(True)
        self.recolor_preserve.toggled.connect(self._update_recolor_from_ui)
        layout.addWidget(self.recolor_preserve, 0, Qt.AlignLeft)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.recolor_mode = QComboBox()
        self.recolor_mode.addItems(["Tint", "Colorize"])
        self.recolor_mode.currentIndexChanged.connect(lambda _: self._update_recolor_from_ui())
        mode_row.addWidget(self.recolor_mode, 1)
        layout.addLayout(mode_row)

        for widget in [
            self.recolor_hue_slider,
            self.recolor_strength_slider,
            self.recolor_sat_slider,
            self.recolor_hue_spin,
            self.recolor_strength_spin,
            self.recolor_sat_spin,
        ]:
            widget.valueChanged.connect(self._update_recolor_from_ui)

        self._apply_default_recolor()
        return sect

    def _apply_default_recolor(self):
        defaults = self._defaults["recolor"]
        self.recolor_enable.blockSignals(True)
        self.recolor_hue_slider.blockSignals(True)
        self.recolor_strength_slider.blockSignals(True)
        self.recolor_sat_slider.blockSignals(True)
        self.recolor_preserve.blockSignals(True)
        self.recolor_mode.blockSignals(True)

        self.recolor_enable.setChecked(defaults.get("enabled", False))
        self.recolor_hue_slider.setValue(int(defaults.get("target_hue", 0)))
        self.recolor_strength_slider.setValue(int(defaults.get("strength", 0)))
        self.recolor_sat_slider.setValue(int(defaults.get("saturation", 0)))
        self.recolor_preserve.setChecked(defaults.get("preserve_luminance", True))
        mode = defaults.get("mode", "Tint")
        if self.recolor_mode.findText(mode) >= 0:
            self.recolor_mode.setCurrentText(mode)

        self.recolor_enable.blockSignals(False)
        self.recolor_hue_slider.blockSignals(False)
        self.recolor_strength_slider.blockSignals(False)
        self.recolor_sat_slider.blockSignals(False)
        self.recolor_preserve.blockSignals(False)
        self.recolor_mode.blockSignals(False)
        self._update_recolor_from_ui()

    def _update_recolor_from_ui(self):
        state = self._color_params()
        recolor = state.get("recolor", {})
        recolor["enabled"] = self.recolor_enable.isChecked()
        recolor["target_hue"] = float(self.recolor_hue_slider.value())
        recolor["strength"] = float(self.recolor_strength_slider.value())
        recolor["saturation"] = float(self.recolor_sat_slider.value())
        recolor["preserve_luminance"] = self.recolor_preserve.isChecked()
        recolor["mode"] = self.recolor_mode.currentText()
        state["recolor"] = recolor
        self._mw._current_params["color"] = state
        self._sync_section_effect("recolor", "Recolor")
        self._emit_render()

    # ---------- Black & White ----------

    def _build_bw_section(self) -> CollapsibleSection:
        sect = CollapsibleSection("Black & White", self, start_collapsed=True)
        layout = sect.content_layout()
        self.bw_enable = QCheckBox("Enable B&W")
        self.bw_enable.setChecked(False)
        self.bw_enable.toggled.connect(self._update_bw_from_ui)
        layout.addWidget(self.bw_enable, 0, Qt.AlignLeft)

        self._bw_sliders: dict[str, QSlider] = {}
        for color in COLOR_NAMES:
            label = color.title()
            slider, spin = self._make_slider_row(label, -100, 100, 0, layout)
            slider.valueChanged.connect(self._update_bw_from_ui)
            spin.valueChanged.connect(self._update_bw_from_ui)
            self._bw_sliders[color] = slider

        self.bw_contrast_slider, self.bw_contrast_spin = self._make_slider_row(
            "Contrast Boost", -100, 100, 0, layout
        )
        self.bw_contrast_slider.valueChanged.connect(self._update_bw_from_ui)
        self.bw_contrast_spin.valueChanged.connect(self._update_bw_from_ui)

        self._apply_default_bw()
        return sect

    def _apply_default_bw(self):
        defaults = self._defaults["black_white"]
        self.bw_enable.blockSignals(True)
        self.bw_enable.setChecked(defaults.get("enabled", False))
        mix = defaults.get("mix", {})
        for c, sl in self._bw_sliders.items():
            sl.blockSignals(True)
            sl.setValue(int(mix.get(c, 0)))
            sl.blockSignals(False)
        self.bw_contrast_slider.blockSignals(True)
        self.bw_contrast_slider.setValue(int(defaults.get("contrast_boost", 0)))
        self.bw_contrast_slider.blockSignals(False)
        self.bw_enable.blockSignals(False)
        self._update_bw_from_ui()

    def _update_bw_from_ui(self):
        state = self._color_params()
        bw = state.get("black_white", {})
        bw["enabled"] = self.bw_enable.isChecked()
        bw["mix"] = {c: float(sl.value()) for c, sl in self._bw_sliders.items()}
        bw["contrast_boost"] = float(self.bw_contrast_slider.value())
        state["black_white"] = bw
        self._mw._current_params["color"] = state
        self._sync_section_effect("black_white", "B&W")
        self._emit_render()

    # ---------- Selective Color ----------

    def _build_selective_color_section(self) -> CollapsibleSection:
        sect = CollapsibleSection("Selective Color", self, start_collapsed=True)
        layout = sect.content_layout()

        self.sel_color_enable = QCheckBox("Enable Selective Color")
        self.sel_color_enable.setChecked(False)
        self.sel_color_enable.toggled.connect(self._update_selective_from_ui)
        layout.addWidget(self.sel_color_enable, 0, Qt.AlignLeft)

        row = QHBoxLayout()
        row.addWidget(QLabel("Range"))
        self.sel_color_range = QComboBox()
        self.sel_color_range.addItems(["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas", "Whites", "Neutrals", "Blacks"])
        self.sel_color_range.currentIndexChanged.connect(lambda _: self._update_selective_from_ui())
        row.addWidget(self.sel_color_range, 1)
        layout.addLayout(row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.sel_color_mode = QComboBox()
        self.sel_color_mode.addItems(["Relative", "Absolute"])
        self.sel_color_mode.currentIndexChanged.connect(lambda _: self._update_selective_from_ui())
        mode_row.addWidget(self.sel_color_mode, 1)
        layout.addLayout(mode_row)

        self._sel_sliders: dict[str, QSlider] = {}
        for label in ["Cyan", "Magenta", "Yellow", "Black"]:
            slider, spin = self._make_slider_row(label, -100, 100, 0, layout)
            slider.valueChanged.connect(self._update_selective_from_ui)
            spin.valueChanged.connect(self._update_selective_from_ui)
            self._sel_sliders[label.lower()] = slider

        self._apply_default_selective_color()
        return sect

    def _apply_default_selective_color(self):
        defaults = self._defaults["selective_color"]
        self.sel_color_enable.blockSignals(True)
        self.sel_color_range.blockSignals(True)
        self.sel_color_mode.blockSignals(True)

        self.sel_color_enable.setChecked(defaults.get("enabled", False))
        self.sel_color_range.setCurrentText(defaults.get("range", "Reds"))
        self.sel_color_mode.setCurrentText(defaults.get("mode", "Relative"))

        for key, sl in self._sel_sliders.items():
            sl.blockSignals(True)
            sl.setValue(int(defaults.get(key, 0)))
            sl.blockSignals(False)

        self.sel_color_enable.blockSignals(False)
        self.sel_color_range.blockSignals(False)
        self.sel_color_mode.blockSignals(False)
        self._update_selective_from_ui()

    def _update_selective_from_ui(self):
        state = self._color_params()
        sc = state.get("selective_color", {})
        sc["enabled"] = self.sel_color_enable.isChecked()
        sc["range"] = self.sel_color_range.currentText()
        sc["mode"] = self.sel_color_mode.currentText()
        for key, sl in self._sel_sliders.items():
            sc[key] = float(sl.value())
        state["selective_color"] = sc
        self._mw._current_params["color"] = state
        self._sync_section_effect("selective_color", "Selective Color")
        self._emit_render()

    # ---------- Color Balance ----------

    def _build_color_balance_section(self) -> CollapsibleSection:
        sect = CollapsibleSection("Color Balance", self, start_collapsed=True)
        layout = sect.content_layout()

        self.cb_enable = QCheckBox("Enable Color Balance")
        self.cb_enable.setChecked(False)
        self.cb_enable.toggled.connect(self._update_color_balance_from_ui)
        layout.addWidget(self.cb_enable, 0, Qt.AlignLeft)

        self._cb_sliders: dict[str, dict[str, QSlider]] = {}
        for tone in ["shadows", "midtones", "highlights"]:
            layout.addWidget(QLabel(tone.title()))
            tone_sliders: dict[str, QSlider] = {}
            for label, axis in [("Cyan / Red", "cr"), ("Magenta / Green", "mg"), ("Yellow / Blue", "yb")]:
                slider, spin = self._make_slider_row(label, -100, 100, 0, layout)
                slider.valueChanged.connect(self._update_color_balance_from_ui)
                spin.valueChanged.connect(self._update_color_balance_from_ui)
                tone_sliders[axis] = slider
            self._cb_sliders[tone] = tone_sliders

        self.cb_preserve = QCheckBox("Preserve Luminance")
        self.cb_preserve.setChecked(True)
        self.cb_preserve.toggled.connect(self._update_color_balance_from_ui)
        layout.addWidget(self.cb_preserve, 0, Qt.AlignLeft)

        self._apply_default_color_balance()
        return sect

    def _apply_default_color_balance(self):
        defaults = self._defaults["color_balance"]
        self.cb_enable.blockSignals(True)
        self.cb_preserve.blockSignals(True)
        self.cb_enable.setChecked(defaults.get("enabled", False))
        self.cb_preserve.setChecked(defaults.get("preserve_luminance", True))
        for tone, sliders in self._cb_sliders.items():
            tone_vals = defaults.get(tone, {})
            for axis, sl in sliders.items():
                sl.blockSignals(True)
                sl.setValue(int(tone_vals.get(axis, 0)))
                sl.blockSignals(False)
        self.cb_enable.blockSignals(False)
        self.cb_preserve.blockSignals(False)
        self._update_color_balance_from_ui()

    def _update_color_balance_from_ui(self):
        state = self._color_params()
        cb = state.get("color_balance", {})
        cb["enabled"] = self.cb_enable.isChecked()
        cb["preserve_luminance"] = self.cb_preserve.isChecked()
        for tone, sliders in self._cb_sliders.items():
            tone_state = cb.get(tone, {})
            for axis, sl in sliders.items():
                tone_state[axis] = float(sl.value())
            cb[tone] = tone_state
        state["color_balance"] = cb
        self._mw._current_params["color"] = state
        self._sync_section_effect("color_balance", "Color Balance")
        self._emit_render()

    # ---------- Color White Balance ----------

    def _build_color_wb_section(self) -> CollapsibleSection:
        sect = CollapsibleSection("White Balance", self, start_collapsed=True)
        layout = sect.content_layout()
        self.cwb_enable = QCheckBox("Enable Color WB")
        self.cwb_enable.setChecked(False)
        self.cwb_enable.toggled.connect(self._update_color_wb_from_ui)
        layout.addWidget(self.cwb_enable, 0, Qt.AlignLeft)

        presets = ["As Shot", "Auto", "Daylight", "Cloudy", "Shade", "Tungsten", "Fluorescent", "Flash"]
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.cwb_preset_combo = QComboBox()
        self.cwb_preset_combo.addItems(presets)
        self.cwb_preset_combo.currentIndexChanged.connect(self._on_color_wb_preset_changed)
        row.addWidget(self.cwb_preset_combo, 1)
        layout.addLayout(row)

        self.cwb_temp_slider, self.cwb_temp_spin = self._make_slider_row("Temperature", -100, 100, 0, layout)
        self.cwb_tint_slider, self.cwb_tint_spin = self._make_slider_row("Tint", -100, 100, 0, layout)
        self.cwb_temp_slider.valueChanged.connect(self._update_color_wb_from_ui)
        self.cwb_temp_spin.valueChanged.connect(self._update_color_wb_from_ui)
        self.cwb_tint_slider.valueChanged.connect(self._update_color_wb_from_ui)
        self.cwb_tint_spin.valueChanged.connect(self._update_color_wb_from_ui)

        picker_btn = QPushButton("WB Picker")
        picker_btn.clicked.connect(self._on_color_wb_picker)
        layout.addWidget(picker_btn, 0, Qt.AlignLeft)

        self._apply_default_color_wb()
        return sect

    def _apply_default_color_wb(self):
        defaults = self._defaults["white_balance"]
        self.cwb_enable.blockSignals(True)
        self.cwb_temp_slider.blockSignals(True)
        self.cwb_tint_slider.blockSignals(True)
        self.cwb_preset_combo.blockSignals(True)

        self.cwb_enable.setChecked(defaults.get("enabled", False))
        self.cwb_temp_slider.setValue(int(defaults.get("temperature", 0)))
        self.cwb_tint_slider.setValue(int(defaults.get("tint", 0)))
        self.cwb_preset_combo.setCurrentText(defaults.get("preset", "As Shot"))

        self.cwb_enable.blockSignals(False)
        self.cwb_temp_slider.blockSignals(False)
        self.cwb_tint_slider.blockSignals(False)
        self.cwb_preset_combo.blockSignals(False)
        self._update_color_wb_from_ui()

    def _on_color_wb_picker(self):
        color = self._mw.pick_color(self)
        if color is None:
            return
        rgb = np.array([color.red(), color.green(), color.blue()], dtype=np.float32) / 255.0
        avg = max(np.mean(rgb), 1e-6)
        scales = avg / np.clip(rgb, 1e-3, None)
        scales = scales / np.max(scales)

        state = self._color_params()
        wb = state.get("white_balance", {})
        wb["multipliers"] = [float(x) for x in scales.tolist()]
        wb["enabled"] = True
        wb["preset"] = "As Shot"
        state["white_balance"] = wb
        self._mw._current_params["color"] = state
        self.cwb_enable.setChecked(True)
        self._sync_section_effect("white_balance", "White Balance")
        self._emit_render()

    def _on_color_wb_preset_changed(self, index: int):
        presets = {
            "Auto": (0, 0),
            "Daylight": (20, 0),
            "Cloudy": (30, 5),
            "Shade": (35, 5),
            "Tungsten": (-40, -5),
            "Fluorescent": (-15, 10),
            "Flash": (25, 5),
            "As Shot": (0, 0),
        }
        name = self.cwb_preset_combo.itemText(index)
        if name in presets:
            t, g = presets[name]
            self.cwb_temp_slider.blockSignals(True)
            self.cwb_tint_slider.blockSignals(True)
            self.cwb_temp_slider.setValue(t)
            self.cwb_tint_slider.setValue(g)
            self.cwb_temp_slider.blockSignals(False)
            self.cwb_tint_slider.blockSignals(False)
        self.cwb_enable.setChecked(True)
        self._update_color_wb_from_ui()

    def _update_color_wb_from_ui(self):
        state = self._color_params()
        wb = state.get("white_balance", {})
        wb["enabled"] = self.cwb_enable.isChecked()
        wb["temperature"] = float(self.cwb_temp_slider.value())
        wb["tint"] = float(self.cwb_tint_slider.value())
        wb["preset"] = self.cwb_preset_combo.currentText()
        state["white_balance"] = wb
        self._mw._current_params["color"] = state
        self._sync_section_effect("white_balance", "White Balance")
        self._emit_render()

    # ---------- shared row builder ----------

    def _make_slider_row(self, label: str, min_val: int, max_val: int, default: int, parent_layout: QVBoxLayout, unit: str = ""):
        row = QHBoxLayout()
        lbl = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        _apply_slider_gradient_role(slider, label)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setFixedHeight(12)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedWidth(64)
        if unit:
            spin.setSuffix(unit)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return slider, spin


# Detail/FX group implementations injected below


class CurveGraphWidget(QFrame):
    """Lightroom-style point curve editor with add/move/remove interactions."""

    pointsChanged = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("curveGraph")
        self._points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._channel = "rgb"
        self._drag_idx: int | None = None
        self._dragging = False
        self._point_radius = 6
        self._padding = 12

    def set_channel(self, channel: str):
        self._channel = channel
        self.update()

    def set_points(self, points: list[tuple[float, float]]):
        self._points = self._normalize_points(points)
        self.update()

    def points(self) -> list[tuple[float, float]]:
        return list(self._points)

    # ---------- math helpers ----------

    def _normalize_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        norm: list[tuple[float, float]] = []
        for pt in points or []:
            try:
                x, y = pt
            except Exception:
                continue
            x = float(max(0.0, min(1.0, x)))
            y = float(max(0.0, min(1.0, y)))
            norm.append((x, y))

        if not norm:
            norm = [(0.0, 0.0), (1.0, 1.0)]

        norm = sorted(norm, key=lambda p: p[0])
        dedup: list[tuple[float, float]] = []
        last_x: float | None = None
        for x, y in norm:
            if last_x is not None and abs(x - last_x) < 1e-4:
                dedup[-1] = (x, y)
            else:
                dedup.append((x, y))
            last_x = x

        if dedup[0][0] > 0.0:
            dedup.insert(0, (0.0, dedup[0][1]))
        if dedup[-1][0] < 1.0:
            dedup.append((1.0, dedup[-1][1]))
        return dedup

    def _curve_samples(self, samples: int = 256) -> np.ndarray:
        pts = self._normalize_points(self._points)
        if len(pts) < 2:
            x = np.linspace(0.0, 1.0, samples, dtype=np.float32)
            return np.stack([x, x], axis=1)

        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        dx = np.diff(xs)
        dy = np.diff(ys)
        slopes = dy / np.where(np.abs(dx) < 1e-6, 1e-6, dx)
        m = np.zeros_like(xs)
        m[0] = slopes[0]
        m[-1] = slopes[-1]
        for i in range(1, len(xs) - 1):
            if slopes[i - 1] * slopes[i] <= 0:
                m[i] = 0.0
            else:
                m[i] = (2 * slopes[i - 1] * slopes[i]) / (slopes[i - 1] + slopes[i])

        t_vals = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        out = np.zeros((samples, 2), dtype=np.float32)
        out[:, 0] = t_vals
        for idx, tx in enumerate(t_vals):
            seg = int(np.clip(np.searchsorted(xs, tx) - 1, 0, len(xs) - 2))
            h = xs[seg + 1] - xs[seg]
            if h <= 0:
                out[idx, 1] = ys[seg]
                continue
            t = (tx - xs[seg]) / h
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            out[idx, 1] = (
                h00 * ys[seg]
                + h10 * h * m[seg]
                + h01 * ys[seg + 1]
                + h11 * h * m[seg + 1]
            )
        out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
        return out

    # ---------- painting ----------

    def _canvas_rect(self):
        return self.rect().adjusted(self._padding, self._padding, -self._padding, -self._padding)

    def _to_canvas(self, pt: tuple[float, float]):
        r = self._canvas_rect()
        x = r.left() + pt[0] * r.width()
        y = r.bottom() - pt[1] * r.height()
        return x, y

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        bg = QColor(26, 26, 26)
        painter.fillRect(self.rect(), bg)

        inner = self._canvas_rect()
        painter.fillRect(inner, QColor(34, 34, 34))

        grid_pen = QPen(QColor(60, 60, 60))
        painter.setPen(grid_pen)
        for i in range(1, 4):
            x = inner.left() + inner.width() * i / 4.0
            y = inner.top() + inner.height() * i / 4.0
            painter.drawLine(int(x), inner.top(), int(x), inner.bottom())
            painter.drawLine(inner.left(), int(y), inner.right(), int(y))

        diag_pen = QPen(QColor(90, 90, 90))
        diag_pen.setStyle(Qt.DashLine)
        painter.setPen(diag_pen)
        painter.drawLine(inner.bottomLeft(), inner.topRight())

        samples = self._curve_samples(256)
        accent = {
            "rgb": QColor(188, 160, 255),
            "r": QColor(255, 120, 120),
            "g": QColor(120, 210, 140),
            "b": QColor(120, 170, 255),
        }.get(self._channel, QColor(188, 160, 255))
        curve_pen = QPen(accent)
        curve_pen.setWidth(2)
        painter.setPen(curve_pen)
        last_pt = None
        for x_norm, y_norm in samples:
            x, y = self._to_canvas((x_norm, y_norm))
            if last_pt is not None:
                painter.drawLine(int(last_pt[0]), int(last_pt[1]), int(x), int(y))
            last_pt = (x, y)

        painter.setBrush(accent)
        painter.setPen(QPen(QColor(18, 18, 18), 1))
        for pt in self._points:
            cx, cy = self._to_canvas(pt)
            r = self._point_radius
            painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

    # ---------- interactions ----------

    def _pos_to_norm(self, pos):
        inner = self._canvas_rect()
        x = (pos.x() - inner.left()) / max(1.0, inner.width())
        y = (inner.bottom() - pos.y()) / max(1.0, inner.height())
        return float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y)))

    def _point_at(self, pos, tol_px: float = 10.0) -> int | None:
        best = None
        best_dist = tol_px * tol_px
        for idx, pt in enumerate(self._points):
            cx, cy = self._to_canvas(pt)
            dx = pos.x() - cx
            dy = pos.y() - cy
            dist2 = dx * dx + dy * dy
            if dist2 <= best_dist:
                best = idx
                best_dist = dist2
        return best

    def mousePressEvent(self, event):  # noqa: N802
        pos = event.position() if hasattr(event, "position") else event.pos()
        idx = self._point_at(pos)
        if event.button() == Qt.RightButton:
            if idx is not None and 0 < idx < len(self._points) - 1 and len(self._points) > 2:
                pts = list(self._points)
                pts.pop(idx)
                self._points = self._normalize_points(pts)
                self.pointsChanged.emit([(float(x), float(y)) for x, y in self._points])
                self.update()
            return

        if idx is not None:
            self._drag_idx = idx
            self._dragging = True
            return

        x, y = self._pos_to_norm(pos)
        pts = list(self._points) + [(x, y)]
        self._points = self._normalize_points(pts)
        self._drag_idx = None
        for i, pt in enumerate(self._points):
            if abs(pt[0] - x) < 1e-4 and abs(pt[1] - y) < 1e-4:
                self._drag_idx = i
                break
        self.pointsChanged.emit([(float(px), float(py)) for px, py in self._points])
        self._dragging = True
        self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self._dragging or self._drag_idx is None:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        x, y = self._pos_to_norm(pos)
        pts = list(self._points)
        idx = int(self._drag_idx)
        if idx == 0:
            x = 0.0
        elif idx == len(pts) - 1:
            x = 1.0
        else:
            min_x = pts[idx - 1][0] + 1e-3
            max_x = pts[idx + 1][0] - 1e-3
            x = max(min_x, min(max_x, x))
        pts[idx] = (x, y)
        self._points = self._normalize_points(pts)
        self.pointsChanged.emit([(float(px), float(py)) for px, py in self._points])
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._dragging = False
        self._drag_idx = None


class GradientStopBar(QFrame):
    """Preview bar + handles for gradient map stops."""

    stopSelected = Signal(int)
    stopsChanged = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._stops: list[dict] = [
            {"pos": 0.0, "color": (0, 0, 0), "opacity": 1.0},
            {"pos": 1.0, "color": (255, 255, 255), "opacity": 1.0},
        ]
        self._selected = 0
        self._dragging = False
        self.setFixedHeight(54)
        self.setFrameShape(QFrame.StyledPanel)

    def stops(self) -> list[dict]:
        return list(self._stops)

    def set_stops(self, stops: list[dict]):
        self._stops = self._normalize_stops(stops)
        self._selected = min(self._selected, len(self._stops) - 1)
        self.update()

    def _normalize_stops(self, stops: list[dict]) -> list[dict]:
        if not stops:
            stops = [
                {"pos": 0.0, "color": (0, 0, 0), "opacity": 1.0},
                {"pos": 1.0, "color": (255, 255, 255), "opacity": 1.0},
            ]
        norm = []
        for st in stops:
            pos = float(st.get("pos", 0.0))
            col = st.get("color", (0, 0, 0))
            opacity = float(st.get("opacity", 1.0))
            try:
                r, g, b = col
            except Exception:
                r, g, b = 0, 0, 0
            norm.append({
                "pos": max(0.0, min(1.0, pos)),
                "color": (int(r), int(g), int(b)),
                "opacity": max(0.0, min(1.0, opacity)),
            })
        norm = sorted(norm, key=lambda s: s["pos"])
        if norm[0]["pos"] > 0.0:
            norm.insert(0, {"pos": 0.0, "color": norm[0]["color"], "opacity": norm[0]["opacity"]})
        if norm[-1]["pos"] < 1.0:
            norm.append({"pos": 1.0, "color": norm[-1]["color"], "opacity": norm[-1]["opacity"]})
        return norm

    def _gradient_rect(self):
        return self.rect().adjusted(12, 14, -12, -14)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        r = self._gradient_rect()
        grad = QLinearGradient(r.left(), 0, r.right(), 0)
        for st in self._stops:
            col = QColor(*st["color"])
            col.setAlphaF(st.get("opacity", 1.0))
            grad.setColorAt(st["pos"], col)
        painter.fillRect(r, grad)
        painter.setPen(QPen(QColor(60, 60, 60)))
        painter.drawRect(r)

        for idx, st in enumerate(self._stops):
            x = r.left() + st["pos"] * r.width()
            top = r.bottom() + 2
            color = QColor(*st["color"])
            painter.setPen(QPen(QColor(255, 255, 255) if idx == self._selected else QColor(80, 80, 80)))
            painter.setBrush(color)
            painter.drawEllipse(int(x - 6), int(top), 12, 12)

    def _pos_to_norm(self, pos) -> float:
        r = self._gradient_rect()
        return float(max(0.0, min(1.0, (pos.x() - r.left()) / max(1.0, r.width()))))

    def _stop_at(self, pos, tol_px: float = 10.0) -> int | None:
        r = self._gradient_rect()
        best = None
        best_dist = tol_px * tol_px
        for idx, st in enumerate(self._stops):
            x = r.left() + st["pos"] * r.width()
            y = r.bottom() + 8
            dx = pos.x() - x
            dy = pos.y() - y
            dist2 = dx * dx + dy * dy
            if dist2 <= best_dist:
                best = idx
                best_dist = dist2
        return best

    def mousePressEvent(self, event):  # noqa: N802
        pos = event.position() if hasattr(event, "position") else event.pos()
        idx = self._stop_at(pos)
        if event.button() == Qt.RightButton:
            if idx is not None and len(self._stops) > 2 and idx not in (0, len(self._stops) - 1):
                self._stops.pop(idx)
                self._selected = max(0, min(self._selected, len(self._stops) - 1))
                self.stopsChanged.emit(self._stops)
                self.update()
            return

        if idx is None:
            pos_norm = self._pos_to_norm(pos)
            color = self._color_at(pos_norm)
            self._stops.append({"pos": pos_norm, "color": color, "opacity": 1.0})
            self._stops = self._normalize_stops(self._stops)
            idx = min(range(len(self._stops)), key=lambda i: abs(self._stops[i]["pos"] - pos_norm))

        self._selected = idx
        self._dragging = True
        self.stopSelected.emit(idx)
        self.stopsChanged.emit(self._stops)
        self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self._dragging:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        idx = int(self._selected)
        pos_norm = self._pos_to_norm(pos)
        min_pos = self._stops[idx - 1]["pos"] + 1e-3 if idx > 0 else 0.0
        max_pos = self._stops[idx + 1]["pos"] - 1e-3 if idx < len(self._stops) - 1 else 1.0
        pos_norm = max(min_pos, min(max_pos, pos_norm))
        self._stops[idx]["pos"] = pos_norm
        self._stops = self._normalize_stops(self._stops)
        self.stopsChanged.emit(self._stops)
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._dragging = False

    def _color_at(self, pos_norm: float) -> tuple[int, int, int]:
        stops = self._normalize_stops(self._stops)
        left = stops[0]
        right = stops[-1]
        for i in range(len(stops) - 1):
            if stops[i]["pos"] <= pos_norm <= stops[i + 1]["pos"]:
                left = stops[i]
                right = stops[i + 1]
                break
        t = 0.0 if right["pos"] == left["pos"] else (pos_norm - left["pos"]) / (right["pos"] - left["pos"])
        lr, lg, lb = left["color"]
        rr, rg, rb = right["color"]
        r = int(lr + (rr - lr) * t)
        g = int(lg + (rg - lg) * t)
        b = int(lb + (rb - lb) * t)
        return (r, g, b)


class GradientEditorWidget(QWidget):
    """Gradient map editor: preview bar + stop controls."""

    stopsChanged = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._stop_bar = GradientStopBar(self)
        self._stop_bar.stopsChanged.connect(self._on_stops_changed)
        self._stop_bar.stopSelected.connect(self._on_stop_selected)
        self._selected_idx = 0

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(self._stop_bar)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 100)
        self.pos_slider.valueChanged.connect(self._on_position_changed)
        self.pos_spin = QDoubleSpinBox()
        self.pos_spin.setRange(0.0, 1.0)
        self.pos_spin.setSingleStep(0.01)
        self.pos_spin.setDecimals(3)
        self.pos_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.pos_spin.valueChanged.connect(self._on_position_spin_changed)
        controls.addWidget(QLabel("Position"))
        controls.addWidget(self.pos_slider, 1)
        controls.addWidget(self.pos_spin)
        v.addLayout(controls)

        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        self.color_btn = QPushButton("Pick Color")
        self.color_btn.clicked.connect(self._on_pick_color)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.opacity_spin.valueChanged.connect(self.opacity_slider.setValue)
        self.opacity_slider.valueChanged.connect(self.opacity_spin.setValue)

        color_row.addWidget(self.color_btn)
        color_row.addSpacing(8)
        color_row.addWidget(QLabel("Opacity"))
        color_row.addWidget(self.opacity_slider, 1)
        color_row.addWidget(self.opacity_spin)
        v.addLayout(color_row)

        self._sync_controls_from_stop()

    def set_stops(self, stops: list[dict]):
        self._stop_bar.set_stops(stops)
        self._selected_idx = min(self._selected_idx, len(self._stop_bar.stops()) - 1)
        self._sync_controls_from_stop()

    def stops(self) -> list[dict]:
        return self._stop_bar.stops()

    def _on_stops_changed(self, stops: list[dict]):
        self._sync_controls_from_stop()
        self.stopsChanged.emit(stops)

    def _on_stop_selected(self, idx: int):
        self._selected_idx = idx
        self._sync_controls_from_stop()

    def _sync_controls_from_stop(self):
        stops = self._stop_bar.stops()
        if not stops:
            return
        self._selected_idx = max(0, min(self._selected_idx, len(stops) - 1))
        st = stops[self._selected_idx]
        self.pos_slider.blockSignals(True)
        self.pos_spin.blockSignals(True)
        self.opacity_slider.blockSignals(True)
        self.opacity_spin.blockSignals(True)

        self.pos_slider.setValue(int(round(st["pos"] * 100)))
        self.pos_spin.setValue(float(st["pos"]))
        opacity_pct = int(round(st.get("opacity", 1.0) * 100))
        self.opacity_slider.setValue(opacity_pct)
        self.opacity_spin.setValue(opacity_pct)
        self._set_color_button_color(QColor(*st["color"]))

        self.pos_slider.blockSignals(False)
        self.pos_spin.blockSignals(False)
        self.opacity_slider.blockSignals(False)
        self.opacity_spin.blockSignals(False)

    def _set_color_button_color(self, color: QColor):
        pm = QPixmap(16, 16)
        pm.fill(color)
        self.color_btn.setIcon(QIcon(pm))

    def _update_stop(self, updater):
        stops = self._stop_bar.stops()
        if not stops:
            return
        idx = max(0, min(self._selected_idx, len(stops) - 1))
        updater(stops[idx])
        self._stop_bar.set_stops(stops)
        self._stop_bar.stopSelected.emit(idx)
        self.stopsChanged.emit(self._stop_bar.stops())
        self._sync_controls_from_stop()

    def _on_position_changed(self, value: int):
        val = float(value) / 100.0
        self.pos_spin.blockSignals(True)
        self.pos_spin.setValue(val)
        self.pos_spin.blockSignals(False)
        self._update_stop(lambda st: st.__setitem__("pos", val))

    def _on_position_spin_changed(self, value: float):
        self.pos_slider.blockSignals(True)
        self.pos_slider.setValue(int(round(value * 100)))
        self.pos_slider.blockSignals(False)
        self._update_stop(lambda st: st.__setitem__("pos", float(value)))

    def _on_pick_color(self):
        stops = self._stop_bar.stops()
        if not stops:
            return
        idx = max(0, min(self._selected_idx, len(stops) - 1))
        current = stops[idx]["color"]
        color = QColorDialog.getColor(QColor(*current), parent=self)
        if color.isValid():
            self._set_color_button_color(color)
            self._update_stop(lambda st, c=color: st.__setitem__("color", (c.red(), c.green(), c.blue())))

    def _on_opacity_changed(self, value: int):
        val = max(0.0, min(1.0, value / 100.0))
        self.opacity_spin.blockSignals(True)
        self.opacity_spin.setValue(int(round(val * 100)))
        self.opacity_spin.blockSignals(False)
        self._update_stop(lambda st: st.__setitem__("opacity", val))


# Tone/geometry/FX classes follow


class ToneGroupWidget(BaseGroupWidget):
    """GROUP 3 - DETAIL (Curves, Mixer, Gradient Map, Split Toning, Normals)."""

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw
        self._defaults = self._detail_defaults()
        self._param_sliders: dict[str, QSlider] = {}
        self._chmix_sliders: dict[str, QSlider] = {}
        self._split_sliders: dict[str, QSlider] = {}
        self._normal_sliders: dict[str, QSlider] = {}
        self._curve_points: dict[str, list[tuple[float, float]]] = {}
        self._current_curve_channel = "rgb"

        self._curve_canvas = CurveGraphWidget(self)
        self._curve_canvas.pointsChanged.connect(self._on_curve_points_changed)

        self._gradient_editor = GradientEditorWidget(self)
        self._gradient_editor.stopsChanged.connect(self._on_gradient_stops_changed)

        self.add_tab(self._build_curves_tab(), "Curves")
        self.add_tab(self._build_channel_mixer_tab(), "Channel Mixer")
        self.add_tab(self._build_gradient_map_tab(), "Gradient Map")
        self.add_tab(self._build_split_toning_tab(), "Split Toning")
        self.add_tab(self._build_normals_tab(), "Normals")

    # ---------- shared helpers ----------

    def _detail_defaults(self) -> dict:
        return {
            "curve_param_highlights": 0,
            "curve_param_lights": 0,
            "curve_param_darks": 0,
            "curve_param_shadows": 0,
            "curve_points_rgb": [(0.0, 0.0), (1.0, 1.0)],
            "curve_points_r": [(0.0, 0.0), (1.0, 1.0)],
            "curve_points_g": [(0.0, 0.0), (1.0, 1.0)],
            "curve_points_b": [(0.0, 0.0), (1.0, 1.0)],
            "chmix_red_r": 100,
            "chmix_red_g": 0,
            "chmix_red_b": 0,
            "chmix_green_r": 0,
            "chmix_green_g": 100,
            "chmix_green_b": 0,
            "chmix_blue_r": 0,
            "chmix_blue_g": 0,
            "chmix_blue_b": 100,
            "grad_stops": [
                {"pos": 0.0, "color": (0, 0, 0), "opacity": 1.0},
                {"pos": 1.0, "color": (255, 255, 255), "opacity": 1.0},
            ],
            "grad_blend_mode": "Normal",
            "grad_opacity": 0,
            "split_shadow_hue": 0,
            "split_shadow_sat": 0,
            "split_mid_hue": 0,
            "split_mid_sat": 0,
            "split_high_hue": 0,
            "split_high_sat": 0,
            "split_balance": 0,
            "normal_light_x": 0,
            "normal_light_y": 0,
            "normal_light_elev": 50,
            "normal_intensity": 100,
            "normal_specular": 0,
            "normal_diffuse": 100,
            "normal_map": None,
        }

    def _emit_render(self):
        if self._mw._preview_base_linear is None:
            return
        try:
            if hasattr(self._mw, "_on_edit_params_changed"):
                self._mw._on_edit_params_changed(immediate_preview=True)
            elif hasattr(self._mw, "_render_preview"):
                self._mw._render_preview()
            if hasattr(self._mw, "_schedule_full_render"):
                self._mw._schedule_full_render()
        except Exception as exc:
            print("[ToneGroupWidget] render error:", exc)

    def _update_param(self, key: str, value):
        self._mw._current_params[key] = value
        self._emit_render()

    def _curve_key_for_channel(self, channel: str) -> str:
        mapping = {"rgb": "curve_points_rgb", "r": "curve_points_r", "g": "curve_points_g", "b": "curve_points_b"}
        return mapping.get(channel, "curve_points_rgb")

    # ---------- Curves ----------

    def _build_parametric_row(self, label: str, key: str, gradient_role: str | None = None):
        row = QHBoxLayout()
        lbl = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-100, 100)
        slider.setValue(self._mw._current_params.get(key, self._defaults[key]))
        slider.setFixedHeight(12)
        if gradient_role:
            slider.setProperty("gradientRole", gradient_role)
        spin = QSpinBox()
        spin.setRange(-100, 100)
        spin.setValue(slider.value())
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedWidth(64)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda v, k=key: self._update_param(k, int(v)))
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        self._param_sliders[key] = slider
        return row

    def _on_curve_channel_selected(self, channel: str):
        self._current_curve_channel = channel
        key = self._curve_key_for_channel(channel)
        pts = self._mw._current_params.get(key, self._defaults[key])
        self._curve_canvas.set_channel(channel)
        self._curve_canvas.set_points(pts)

    def _on_curve_points_changed(self, pts: list[tuple[float, float]]):
        key = self._curve_key_for_channel(self._current_curve_channel)
        clean = [(float(x), float(y)) for x, y in pts]
        self._mw._current_params[key] = clean
        self._curve_points[key] = clean
        self._emit_render()

    def _reset_curve_points(self):
        key = self._curve_key_for_channel(self._current_curve_channel)
        default = self._defaults.get(key, [(0.0, 0.0), (1.0, 1.0)])
        self._mw._current_params[key] = list(default)
        self._curve_canvas.set_points(default)
        self._emit_render()

    def _build_curves_tab(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        param_section = CollapsibleSection("Parametric Curve", self, start_collapsed=False)
        pl = param_section.content_layout()
        pl.addLayout(self._build_parametric_row("Highlights", "curve_param_highlights", "highlights"))
        pl.addLayout(self._build_parametric_row("Lights", "curve_param_lights", "highlights"))
        pl.addLayout(self._build_parametric_row("Darks", "curve_param_darks", "shadows"))
        pl.addLayout(self._build_parametric_row("Shadows", "curve_param_shadows", "shadows"))
        v.addWidget(param_section)

        point_section = CollapsibleSection("Point Curve Editor", self, start_collapsed=False)
        pcl = point_section.content_layout()
        btn_row = QHBoxLayout()
        self._curve_channel_buttons: dict[str, QToolButton] = {}
        for key, text in [("rgb", "RGB Master"), ("r", "Red"), ("g", "Green"), ("b", "Blue")]:
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setAutoRaise(True)
            btn.setText(text)
            btn.setMinimumWidth(70)
            btn.clicked.connect(lambda checked, ch=key: self._on_curve_channel_selected(ch))
            self._curve_channel_buttons[key] = btn
            btn_row.addWidget(btn)
        self._curve_channel_buttons["rgb"].setChecked(True)
        reset_btn = QPushButton("Reset Curve")
        reset_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reset_btn.clicked.connect(self._reset_curve_points)
        btn_row.addStretch(1)
        btn_row.addWidget(reset_btn)
        pcl.addLayout(btn_row)
        pcl.addWidget(self._curve_canvas)
        hint = QLabel("Left-click to add/move points. Right-click to remove.")
        hint.setStyleSheet("color: #bbbbbb; font-size: 11px;")
        pcl.addWidget(hint)
        v.addWidget(point_section)

        v.addStretch(1)
        scroll.setWidget(container)

        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)

        self._on_curve_channel_selected("rgb")
        return page

    # ---------- Channel Mixer ----------

    def _add_chmix_row(self, label: str, key: str, parent_layout: QVBoxLayout):
        row = QHBoxLayout()
        lbl = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-200, 200)
        slider.setValue(int(self._mw._current_params.get(key, self._defaults[key])))
        slider.setFixedHeight(12)
        spin = QSpinBox()
        spin.setRange(-200, 200)
        spin.setValue(slider.value())
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedWidth(64)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda v, k=key: self._update_param(k, int(v)))
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        self._chmix_sliders[key] = slider

    def _build_channel_group(self, title: str, keys: dict[str, str]) -> CollapsibleSection:
        section = CollapsibleSection(title, self, start_collapsed=False)
        cl = section.content_layout()
        self._add_chmix_row("Red → " + title.split()[0], keys["r"], cl)
        self._add_chmix_row("Green → " + title.split()[0], keys["g"], cl)
        self._add_chmix_row("Blue → " + title.split()[0], keys["b"], cl)
        return section

    def _build_channel_mixer_tab(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        v.addWidget(self._build_channel_group("Red Output Channel", {
            "r": "chmix_red_r",
            "g": "chmix_red_g",
            "b": "chmix_red_b",
        }))
        v.addWidget(self._build_channel_group("Green Output Channel", {
            "r": "chmix_green_r",
            "g": "chmix_green_g",
            "b": "chmix_green_b",
        }))
        v.addWidget(self._build_channel_group("Blue Output Channel", {
            "r": "chmix_blue_r",
            "g": "chmix_blue_g",
            "b": "chmix_blue_b",
        }))

        v.addStretch(1)
        scroll.setWidget(container)
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)
        return page

    # ---------- Gradient Map ----------

    def _on_gradient_stops_changed(self, stops: list[dict]):
        self._mw._current_params["grad_stops"] = stops
        self._emit_render()

    def _build_gradient_map_tab(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        grad_section = CollapsibleSection("Gradient Editor", self, start_collapsed=False)
        gl = grad_section.content_layout()
        self._gradient_editor.set_stops(self._mw._current_params.get("grad_stops", self._defaults["grad_stops"]))
        gl.addWidget(self._gradient_editor)
        v.addWidget(grad_section)

        blend_section = CollapsibleSection("Blend Options", self, start_collapsed=False)
        bl = blend_section.content_layout()
        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Blend Mode"))
        self.grad_mode_combo = QComboBox()
        modes = ["Normal", "Soft Light", "Overlay", "Multiply", "Screen"]
        self.grad_mode_combo.addItems(modes)
        current_mode = self._mw._current_params.get("grad_blend_mode", self._defaults["grad_blend_mode"])
        if current_mode in modes:
            self.grad_mode_combo.setCurrentText(current_mode)
        self.grad_mode_combo.currentTextChanged.connect(lambda text: self._update_param("grad_blend_mode", text))
        blend_row.addWidget(self.grad_mode_combo)
        blend_row.addStretch(1)
        bl.addLayout(blend_row)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        self.grad_opacity_slider = QSlider(Qt.Horizontal)
        self.grad_opacity_slider.setRange(0, 100)
        self.grad_opacity_slider.setValue(int(self._mw._current_params.get("grad_opacity", 100)))
        self.grad_opacity_slider.setFixedHeight(12)
        self.grad_opacity_spin = QSpinBox()
        self.grad_opacity_spin.setRange(0, 100)
        self.grad_opacity_spin.setValue(self.grad_opacity_slider.value())
        self.grad_opacity_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.grad_opacity_spin.setFixedWidth(64)
        self.grad_opacity_slider.valueChanged.connect(self.grad_opacity_spin.setValue)
        self.grad_opacity_spin.valueChanged.connect(self.grad_opacity_slider.setValue)
        self.grad_opacity_slider.valueChanged.connect(lambda v: self._update_param("grad_opacity", int(v)))
        op_row.addWidget(self.grad_opacity_slider, 1)
        op_row.addWidget(self.grad_opacity_spin)
        bl.addLayout(op_row)

        v.addWidget(blend_section)
        v.addStretch(1)
        scroll.setWidget(container)
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)
        return page

    # ---------- Split Toning ----------

    def _add_split_row(self, layout: QVBoxLayout, label: str, key: str, min_val: int, max_val: int, suffix: str = ""):
        row = QHBoxLayout()
        lbl = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(int(self._mw._current_params.get(key, self._defaults[key])))
        slider.setFixedHeight(12)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(slider.value())
        if suffix:
            spin.setSuffix(suffix)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedWidth(64)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda v, k=key: self._update_param(k, int(v)))
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        layout.addLayout(row)
        self._split_sliders[key] = slider

    def _build_split_section(self, title: str, hue_key: str, sat_key: str) -> CollapsibleSection:
        section = CollapsibleSection(title, self, start_collapsed=False)
        cl = section.content_layout()
        self._add_split_row(cl, "Hue", hue_key, 0, 360, "°")
        self._add_split_row(cl, "Saturation", sat_key, 0, 100, "")
        return section

    def _build_split_toning_tab(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        v.addWidget(self._build_split_section("Shadows", "split_shadow_hue", "split_shadow_sat"))
        v.addWidget(self._build_split_section("Midtones", "split_mid_hue", "split_mid_sat"))
        v.addWidget(self._build_split_section("Highlights", "split_high_hue", "split_high_sat"))

        balance_section = CollapsibleSection("Balance", self, start_collapsed=False)
        bl = balance_section.content_layout()
        self._add_split_row(bl, "Balance", "split_balance", -100, 100)
        v.addWidget(balance_section)

        v.addStretch(1)
        scroll.setWidget(container)
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)
        return page

    # ---------- Normals ----------

    def _encode_normal_map(self, normal: np.ndarray) -> dict:
        normal01 = np.clip(normal, 0.0, 1.0)
        u8 = (normal01 * 255.0).round().astype(np.uint8)
        h, w = u8.shape[:2]
        img = Image.fromarray(u8, mode="RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"data": encoded, "width": w, "height": h, "encoding": "base64_png"}

    def _set_normal_map(self, normal: np.ndarray):
        payload = self._encode_normal_map(normal)
        self._mw._current_params["normal_map"] = payload
        self._emit_render()

    def _generate_normals_from_image(self):
        base = self._mw._preview_base_linear or self._mw._base_image
        if base is None:
            return
        src = np.clip(np.asarray(base, dtype=np.float32), 0.0, 1.0)
        lum = (
            0.2126 * src[..., 0] +
            0.7152 * src[..., 1] +
            0.0722 * src[..., 2]
        )
        lum_small = lum
        max_dim = 768
        h, w = lum.shape
        scale = min(1.0, max_dim / max(h, w))
        if scale < 0.999:
            small_w = max(1, int(w * scale))
            small_h = max(1, int(h * scale))
            img = Image.fromarray((lum * 255.0).astype(np.uint8))
            img = img.resize((small_w, small_h), Image.BILINEAR)
            lum_small = np.asarray(img, dtype=np.float32) / 255.0

        gx = np.gradient(lum_small, axis=1)
        gy = np.gradient(lum_small, axis=0)
        nz = np.ones_like(lum_small)
        normal = np.stack((-gx, -gy, nz), axis=-1)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-6, 1.0, norm)
        normal = normal / norm
        normal01 = normal * 0.5 + 0.5
        self._set_normal_map(normal01)

    def _load_normal_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Normal Map", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            normal = np.asarray(img, dtype=np.float32) / 255.0
            self._set_normal_map(normal)
        except Exception as exc:
            print("[ToneGroupWidget] Failed to load normal map:", exc)

    def _build_normal_row(self, label: str, key: str, min_val: int, max_val: int, default: int):
        row = QHBoxLayout()
        lbl = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(int(self._mw._current_params.get(key, default)))
        slider.setFixedHeight(12)
        spin = QDoubleSpinBox() if (max_val - min_val) <= 200 else QSpinBox()
        if isinstance(spin, QDoubleSpinBox):
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setRange(min_val / 100.0, max_val / 100.0)
            spin.setValue(slider.value() / 100.0)
            slider.valueChanged.connect(lambda v, s=spin: s.setValue(v / 100.0))
            spin.valueChanged.connect(lambda v, sld=slider: sld.setValue(int(round(v * 100.0))))
        else:
            spin.setRange(min_val, max_val)
            spin.setValue(slider.value())
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedWidth(70)
        slider.valueChanged.connect(lambda v, k=key: self._update_param(k, int(v)))
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        self._normal_sliders[key] = slider
        return row

    def _build_normals_tab(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setObjectName("adjustScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container.setObjectName("adjustScrollContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        recon = CollapsibleSection("Normals Reconstruction", self, start_collapsed=False)
        rl = recon.content_layout()
        btn_row = QHBoxLayout()
        gen_btn = QPushButton("Generate Normals from Image")
        gen_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        gen_btn.clicked.connect(self._generate_normals_from_image)
        load_btn = QPushButton("Load Normal Map…")
        load_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        load_btn.clicked.connect(self._load_normal_map)
        btn_row.addWidget(gen_btn)
        btn_row.addWidget(load_btn)
        btn_row.addStretch(1)
        rl.addLayout(btn_row)
        v.addWidget(recon)

        relight = CollapsibleSection("Relight Controls", self, start_collapsed=False)
        rl2 = relight.content_layout()
        rl2.addLayout(self._build_normal_row("Light Direction X", "normal_light_x", -100, 100, 0))
        rl2.addLayout(self._build_normal_row("Light Direction Y", "normal_light_y", -100, 100, 0))
        rl2.addLayout(self._build_normal_row("Light Elevation", "normal_light_elev", 0, 100, 50))
        rl2.addLayout(self._build_normal_row("Intensity", "normal_intensity", 0, 200, 100))
        rl2.addLayout(self._build_normal_row("Specular Boost", "normal_specular", 0, 200, 0))
        rl2.addLayout(self._build_normal_row("Diffuse Strength", "normal_diffuse", 0, 200, 100))
        v.addWidget(relight)

        v.addStretch(1)
        scroll.setWidget(container)
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)
        return page

    # ---------- UI helpers ----------

    def reset_ui_to_defaults(self):
        for key, slider in self._param_sliders.items():
            slider.blockSignals(True)
            slider.setValue(self._defaults.get(key, 0))
            slider.blockSignals(False)

        for key, slider in self._chmix_sliders.items():
            slider.blockSignals(True)
            slider.setValue(self._defaults.get(key, 0))
            slider.blockSignals(False)

        for key, slider in self._split_sliders.items():
            slider.blockSignals(True)
            slider.setValue(self._defaults.get(key, 0))
            slider.blockSignals(False)

        for key, slider in self._normal_sliders.items():
            slider.blockSignals(True)
            slider.setValue(self._defaults.get(key, slider.value()))
            slider.blockSignals(False)

        self._gradient_editor.set_stops(self._defaults["grad_stops"])
        self.grad_mode_combo.setCurrentText(self._defaults["grad_blend_mode"])
        self.grad_opacity_slider.setValue(self._defaults["grad_opacity"])
        for ch in ["rgb", "r", "g", "b"]:
            key = self._curve_key_for_channel(ch)
            self._mw._current_params[key] = list(self._defaults[key])
        self._on_curve_channel_selected(self._current_curve_channel)
        self._emit_render()


class GeometryGroupWidget(BaseGroupWidget):
    """GROUP 4 - GEOMETRY stub (kept for compatibility)."""

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("Normals tab is available under Detail -> Normals."))
        v.addStretch(1)
        self.add_tab(page, "Normals")


class FXGroupWidget(BaseGroupWidget):
    """GROUP 5 - FX stub."""

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("Lens Filter tab coming soon..."))
        v.addStretch(1)
        self.add_tab(page, "Lens Filter")


__all__ = [
    "BaseGroupWidget",
    "LightGroupWidget",
    "ColorGroupWidget",
    "ToneGroupWidget",
    "GeometryGroupWidget",
    "FXGroupWidget",
    "HSLTabWidget",
]
