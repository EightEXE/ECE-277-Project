from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
)
from PySide6.QtGui import QIcon

from .widgets import CollapsibleSection

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
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
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
        if self._mw._preview_base_image is None:
            return

        if hasattr(self._mw, "_render_timer"):
            try:
                self._mw._schedule_render()
            except Exception as exc:
                print("[LightGroupWidget] schedule_render error:", exc)

        try:
            self._mw._apply_edit_params_to_current_image(self._mw._current_params)
        except Exception as exc:
            print("[LightGroupWidget] direct render error:", exc)

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
        if hasattr(self._mw, "_schedule_render"):
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
            self._compact_slider(slider)

            if is_gamma:
                slider.setRange(10, 300)  # maps to 0.10–3.00
                value_edit = QDoubleSpinBox()
                value_edit.setRange(0.10, 3.00)
                value_edit.setSingleStep(0.05)
                value_edit.setDecimals(2)
                value_edit.setValue(1.00)
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
            slider.setRange(0, 100)
            self._compact_slider(slider)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            spin.setFixedWidth(56)

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

        def make_row(label_text: str, gradient_name: str, param_key: str):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            self._compact_slider(slider)
            slider.setProperty("gradientRole", gradient_name)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            spin.setFixedWidth(56)

            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda v, key=param_key: self._update_param(key, v))

            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            return slider, row

        self.wb_temp_slider, temp_row = make_row("White Balance", "wbTemp", "temperature")
        self.wb_tint_slider, tint_row = make_row("Tint", "wbTint", "tint")

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
            slider.setRange(0, 100)
            self._compact_slider(slider)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            slider.setValue(50)
            spin.setFixedWidth(56)
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
        slider.setRange(-400, 400)  # -4 to +4 EV
        self._compact_slider(slider)
        spin = QDoubleSpinBox()
        spin.setRange(-4.0, 4.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.10)
        spin.setSuffix(" EV")
        spin.setFixedWidth(80)

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
            slider.setRange(0, 100)
            self._compact_slider(slider)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(50)
            spin.setFixedWidth(56)

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


class ColorGroupWidget(BaseGroupWidget):
    """
    GROUP 2 — COLOR
    Tabs: HSL, Recolor, Black & White, Selective Color, Color Balance, (White Balance)
    For now this is a stub with simple placeholder pages.
    """

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw

        for name in ["HSL", "Recolor", "Black & White", "Selective Color", "Color Balance", "White Balance"]:
            page = QWidget()
            v = QVBoxLayout(page)
            v.addWidget(QLabel(f"{name} tab coming soon…"))
            v.addStretch(1)
            self.add_tab(page, name)


class ToneGroupWidget(BaseGroupWidget):
    """GROUP 3 — TONE stub."""

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw
        for name in ["Curves", "Channel Mixer", "Gradient Map", "Split Toning"]:
            page = QWidget()
            v = QVBoxLayout(page)
            v.addWidget(QLabel(f"{name} tab coming soon…"))
            v.addStretch(1)
            self.add_tab(page, name)


class GeometryGroupWidget(BaseGroupWidget):
    """GROUP 4 — GEOMETRY stub."""

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("Normals tab coming soon…"))
        v.addStretch(1)
        self.add_tab(page, "Normals")


class FXGroupWidget(BaseGroupWidget):
    """GROUP 5 — FX stub."""

    def __init__(self, mw: "MainWindow", parent=None):
        super().__init__(parent)
        self._mw = mw
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("Lens Filter tab coming soon…"))
        v.addStretch(1)
        self.add_tab(page, "Lens Filter")


__all__ = [
    "BaseGroupWidget",
    "LightGroupWidget",
    "ColorGroupWidget",
    "ToneGroupWidget",
    "GeometryGroupWidget",
    "FXGroupWidget",
]
