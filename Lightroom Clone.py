import os
import sys
import json
from datetime import datetime

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QFileDialog, QStatusBar, QListView, QSizePolicy, QLayout,
    QAbstractItemView, QStyle, QDockWidget, QSlider, QHBoxLayout, QMessageBox,
    QTreeWidget, QTreeWidgetItem
)
from PySide6.QtCore import Qt, QSize, QRunnable, QThreadPool, Signal, QObject, QPoint
from PySide6.QtGui import (
    QPixmap, QIcon, QImageReader, QFontMetrics, QImage
)

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
LRC_VERSION = "1.0"


def _relpath_or_same(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path


def _abspath_from_base(maybe_rel: str, base: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.abspath(os.path.join(base, maybe_rel))


class _WorkerSignals(QObject):
    ready = Signal(str, QIcon)


class _ThumbTask(QRunnable):
    def __init__(self, path: str, size: QSize):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = _WorkerSignals()

    def run(self):
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)
        reader.setScaledSize(self.size)
        img = reader.read()
        if img.isNull():
            icon = QIcon()
        else:
            icon = QIcon(QPixmap.fromImage(img))
        self.signals.ready.emit(self.path, icon)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---------- Window basics ----------
        self.setWindowTitle("LightRoom Clone")
        self.setWindowIcon(QIcon("icon.ico"))
        self.setIconSize(QSize(500, 500))
        self.resize(1200, 800)
        self.setMinimumSize(600, 400)

        central = QWidget(self)
        self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central)
        self.vbox.setContentsMargins(8, 8, 8, 8)
        self.vbox.setSpacing(8)

        # project / image state
        self._current_folder = None
        self._current_path = None            # active image path
        self._current_pixmap = None          # current preview pixmap
        self._preview_base_image = None      # QImage preview base for active image

        # image caches
        self._image_cache = {}               # path -> {"full": QImage, "preview": QImage}
        self._image_params = {}              # path -> dict of combined params
        self._image_parents = {}             # path -> QTreeWidgetItem parent in history tree
        self._edits = {}                     # used for .lrc export (combined params)

        # current parameters (add new ones here)
        self._current_params = {
            "saturation": 128,
            "contrast": 128,
        }

        # ---------- Central image area ----------
        self.image_display = QLabel("No Image Loaded")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumHeight(400)
        self.image_display.setObjectName("imageDisplay")
        self.vbox.addWidget(self.image_display, 1)
        self.vbox.setSizeConstraint(QLayout.SetDefaultConstraint)

        # ---------- Bottom filmstrip (now as dock) ----------
        self.thumbs = QListWidget()
        self.thumbs.setObjectName("bottomFilmstrip")
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(100, 100))
        self.thumbs.setResizeMode(QListWidget.Adjust)
        self.thumbs.setFlow(QListView.LeftToRight)
        self.thumbs.setWrapping(False)
        self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setSpacing(10)
        self.thumbs.setUniformItemSizes(True)
        self.thumbs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.thumbs.itemClicked.connect(self._on_thumbnail_clicked)

        # Let the user resize this dock vertically
        self.thumbs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.thumbs.horizontalScrollBar().valueChanged.connect(
            lambda _: self._ensure_visible_thumbs()
        )

        icon_w = self.thumbs.iconSize().width()
        icon_h = self.thumbs.iconSize().height()
        fm = QFontMetrics(self.thumbs.font())
        text_h = fm.height()
        pad_h = 8
        cell_w = icon_w + 16
        cell_h = icon_h + text_h + pad_h

        self.thumbs.setGridSize(QSize(cell_w, cell_h))

        sb_h = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        # Minimum height that still lets the user shrink/expand it
        self.thumbs.setMinimumHeight(cell_h + sb_h + 2)

        self.thumbs.setResizeMode(QListView.Fixed)
        self.thumbs.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.setStatusBar(QStatusBar(self))

        # ---------- Thread pool for thumbs ----------
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(min(8, (os.cpu_count() or 4)))
        self._icon_cache = {}
        self._loading = set()
        self._item_for_path = {}

        # ---------- Menu ----------
        file_menu = self.menuBar().addMenu("&File")

        # New project
        self.act_new_project = file_menu.addAction("&New Project")
        self.act_new_project.setShortcut("Ctrl+N")
        self.act_new_project.triggered.connect(self.new_project)


        # Open PROJECT (.lrc)
        self.act_open_project = file_menu.addAction("&Open Project...")
        self.act_open_project.setShortcut("Ctrl+O")
        self.act_open_project.triggered.connect(self.import_project)

        # Import FOLDER (images)
        self.act_import_folder = file_menu.addAction("&Import Folder...")
        self.act_import_folder.setShortcut("Ctrl+I")
        self.act_import_folder.triggered.connect(self.open_folder)

        file_menu.addSeparator()

        # Save Project (.lrc)
        self.act_save_project = file_menu.addAction("&Save Project")
        self.act_save_project.setShortcut("Ctrl+S")
        self.act_save_project.triggered.connect(self.export_lrc)

        # placeholder icon for thumbnails
        self._placeholder_icon = QIcon(QPixmap(icon_w, icon_h))



        # ---------- Right sidebar controls ----------
        # Saturation
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 255)
        self.saturation_slider.setValue(128)
        self.saturation_value = QLabel("Saturation: 128")

        sat_row = QHBoxLayout()
        sat_row.addWidget(QLabel("Saturation"))
        sat_row.addWidget(self.saturation_slider, 1)
        sat_row.addWidget(self.saturation_value)

        # Contrast
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 255)
        self.contrast_slider.setValue(128)
        self.contrast_value = QLabel("Contrast: 128")

        con_row = QHBoxLayout()
        con_row.addWidget(QLabel("Contrast"))
        con_row.addWidget(self.contrast_slider, 1)
        con_row.addWidget(self.contrast_value)

        # connect signals
        self.saturation_slider.valueChanged.connect(self._on_saturation_changed)
        self.saturation_slider.sliderReleased.connect(self._on_edit_committed)

        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.contrast_slider.sliderReleased.connect(self._on_edit_committed)

        # Right dock (Adjustments)
        self.right_dock = QDockWidget("Adjustments", self)
        self.right_dock.setObjectName("rightSidebar")
        self.right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.right_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.addLayout(sat_row)
        sidebar_layout.addLayout(con_row)
        sidebar_layout.addStretch(1)
        self.right_dock.setWidget(sidebar_content)

        # ---------- Left dock: Edit tree ----------
        self.left_dock = QDockWidget("Edit History", self)
        self.left_dock.setObjectName("leftSidebar")
        self.left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.left_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        lSidebar_content = QWidget()
        lSidebar_layout = QVBoxLayout(lSidebar_content)

        history_label = QLabel("Edit History")
        history_label.setObjectName("historyLabel")

        self.history_tree = QTreeWidget()
        self.history_tree.setObjectName("editHistoryTree")
        self.history_tree.setHeaderHidden(True)
        self.history_tree.setIconSize(QSize(64, 64))
        self.history_tree.itemClicked.connect(self._on_history_item_clicked)

        lSidebar_layout.addWidget(history_label)
        lSidebar_layout.addWidget(self.history_tree, 1)
        self.left_dock.setWidget(lSidebar_content)

        # ---------- Filmstrip dock ----------
        self.filmstrip_dock = QDockWidget("Filmstrip", self)
        self.filmstrip_dock.setObjectName("filmstripDock")
        self.filmstrip_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.filmstrip_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.filmstrip_dock.setWidget(self.thumbs)

        # Add docks to the main window
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.filmstrip_dock)

        # ---------- Window menu: toggles & layout ----------
        window_menu = self.menuBar().addMenu("&Window")

        self.act_restore_layout = window_menu.addAction("Restore Window Layout")
        self.act_restore_layout.triggered.connect(self.restore_default_layout)

        window_menu.addSeparator()
        window_menu.addAction(self.left_dock.toggleViewAction())
        window_menu.addAction(self.right_dock.toggleViewAction())
        window_menu.addAction(self.filmstrip_dock.toggleViewAction())

        window_menu.addSeparator()
        self.act_toggle_statusbar = window_menu.addAction("Show Status Bar")
        self.act_toggle_statusbar.setCheckable(True)
        self.act_toggle_statusbar.setChecked(True)
        self.act_toggle_statusbar.triggered.connect(self._on_toggle_statusbar)

        # --- Full screen toggle (F11) ---
        window_menu.addSeparator()
        self.act_fullscreen = window_menu.addAction("Full Screen")
        self.act_fullscreen.setCheckable(True)
        self.act_fullscreen.setShortcut("F11")
        self.act_fullscreen.triggered.connect(self.toggle_fullscreen)

        # fullscreen state
        self._is_fullscreen_mode = False
        self._fs_prev_geometry = None
        self._fs_prev_menubar_visible = True
        self._fs_prev_statusbar_visible = True
        self._fs_prev_left_visible = True
        self._fs_prev_right_visible = True
        self._fs_prev_filmstrip_visible = True

        # ---------- Apply theme ----------
        self._apply_styles()

        # Save the default layout so we can restore it later
        self._default_layout_state = self.saveState()

    def new_project(self):
        """Clear everything to start a new blank project."""
        # Reset project-related containers
        self._current_folder = None
        self._current_path = None
        self._preview_base_image = None
        self._current_pixmap = None

        self._image_cache.clear()
        self._image_params.clear()
        self._image_parents.clear()
        self._edits.clear()

        # Clear UI elements
        self.thumbs.clear()
        self.history_tree.clear()
        self.image_display.clear()
        self.image_display.setText("No Image Loaded")

        # Reset sliders
        self.saturation_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)

        self.saturation_slider.setValue(128)
        self.contrast_slider.setValue(128)

        self.saturation_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)

        self.saturation_value.setText("Saturation: 128")
        self.contrast_value.setText("Contrast: 128")
        self._current_params["saturation"] = 128
        self._current_params["contrast"] = 128

        self.statusBar().showMessage("New blank project created.")



    def toggle_fullscreen(self, checked: bool = False):
        """Toggle a clean full-screen image-only mode."""
        # Ignore 'checked' and use our own flag so it always works
        if not self._is_fullscreen_mode:
            # ENTER fullscreen
            self._is_fullscreen_mode = True

            # keep menu action in sync
            self.act_fullscreen.blockSignals(True)
            self.act_fullscreen.setChecked(True)
            self.act_fullscreen.blockSignals(False)

            # save current geometry & visibilities
            self._fs_prev_geometry = self.saveGeometry()

            mb = self.menuBar()
            sb = self.statusBar()

            self._fs_prev_menubar_visible = mb.isVisible() if mb else True
            self._fs_prev_statusbar_visible = sb.isVisible() if sb else True

            self._fs_prev_left_visible = self.left_dock.isVisible()
            self._fs_prev_right_visible = self.right_dock.isVisible()
            self._fs_prev_filmstrip_visible = self.filmstrip_dock.isVisible()

            # hide chrome
            if mb:
                mb.hide()
            if sb:
                sb.hide()

            self.left_dock.hide()
            self.right_dock.hide()
            self.filmstrip_dock.hide()

            # go fullscreen
            self.showFullScreen()

        else:
            # EXIT fullscreen
            self._is_fullscreen_mode = False

            # keep menu action in sync
            self.act_fullscreen.blockSignals(True)
            self.act_fullscreen.setChecked(False)
            self.act_fullscreen.blockSignals(False)

            # leave fullscreen
            self.showNormal()

            # restore geometry
            if self._fs_prev_geometry is not None:
                self.restoreGeometry(self._fs_prev_geometry)

            mb = self.menuBar()
            sb = self.statusBar()

            if mb:
                mb.setVisible(self._fs_prev_menubar_visible)
            if sb:
                sb.setVisible(self._fs_prev_statusbar_visible)
                # sync the "Show Status Bar" action
                self.act_toggle_statusbar.blockSignals(True)
                self.act_toggle_statusbar.setChecked(self._fs_prev_statusbar_visible)
                self.act_toggle_statusbar.blockSignals(False)

            if self._fs_prev_left_visible:
                self.left_dock.show()
            if self._fs_prev_right_visible:
                self.right_dock.show()
            if self._fs_prev_filmstrip_visible:
                self.filmstrip_dock.show()

    def keyPressEvent(self, event):
        # F11: toggle fullscreen no matter what
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return

        # Esc: exit fullscreen if we are in it
        if event.key() == Qt.Key_Escape and self._is_fullscreen_mode:
            self.toggle_fullscreen()
            event.accept()
            return

        # otherwise, normal behavior
        super().keyPressEvent(event)


    # ---------- Window layout helpers ----------

    def restore_default_layout(self):
        """Restore docks/filmstrip to the original layout."""
        if hasattr(self, "_default_layout_state"):
            self.restoreState(self._default_layout_state)

    def _on_toggle_statusbar(self, checked: bool):
        sb = self.statusBar()
        if sb is not None:
            sb.setVisible(checked)

    # ---------- Slider callbacks ----------

    def _on_saturation_changed(self, value: int):
        """Realtime update for saturation slider."""
        self.saturation_value.setText(f"Saturation: {value}")
        self._current_params["saturation"] = value
        if self._preview_base_image is not None:
            self._apply_edit_params_to_current_image(self._current_params)

    def _on_contrast_changed(self, value: int):
        """Realtime update for contrast slider."""
        self.contrast_value.setText(f"Contrast: {value}")
        self._current_params["contrast"] = value
        if self._preview_base_image is not None:
            self._apply_edit_params_to_current_image(self._current_params)

    def _on_edit_committed(self):
        """Called when any slider is released."""
        if not self._current_path or self._preview_base_image is None:
            return

        # Save combined params for this image = its "Active Edit"
        self._image_params[self._current_path] = dict(self._current_params)
        self._edits[self._current_path] = dict(self._current_params)

        # Update the tree for this image
        self._update_history_for_current_image()

    # ---------- History tree logic ----------

    def _update_history_for_current_image(self):
        """
        Ensure the left tree has for this image:
        - a parent node named after the file
        - children: Original, Saturation only, Contrast only (if present)
        """
        path = self._current_path
        if not path or self._current_pixmap is None:
            return

        if path not in self._image_parents:
            parent = QTreeWidgetItem(self.history_tree)
            parent.setText(0, os.path.basename(path))
            self._image_parents[path] = parent
        else:
            parent = self._image_parents[path]
            parent.takeChildren()  # clear children

        # Icons
        cache = self._image_cache.get(path)
        if cache:
            orig_icon = QIcon(QPixmap.fromImage(cache["preview"]))
        else:
            orig_icon = QIcon()

        edited_icon = QIcon(
            self._current_pixmap.scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        combined_params = self._image_params.get(path, {})

        # Parent = combined edit
        parent.setIcon(0, edited_icon)
        parent.setData(0, Qt.UserRole, {"path": path, "params": dict(combined_params)})

        # Child: Original
        child_orig = QTreeWidgetItem(parent)
        child_orig.setText(0, "Original")
        child_orig.setIcon(0, orig_icon)
        child_orig.setData(0, Qt.UserRole, {"path": path, "params": {}})

        # Children: single-parameter views
        for name in ("saturation", "contrast"):
            if name in combined_params:
                val = combined_params[name]
                child = QTreeWidgetItem(parent)
                child.setText(0, f"{name.capitalize()} only ({val})")
                child.setIcon(0, edited_icon)
                child.setData(
                    0,
                    Qt.UserRole,
                    {"path": path, "params": {name: val}},
                )

        parent.setExpanded(True)

    def _on_history_item_clicked(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path")
        params = data.get("params", {})

        if not path:
            return

        self._show_image_version(path, params, update_sliders=True)

    # ---------- Styling ----------

    def _apply_styles(self):
        accent = "#2f8fff"
        self.setStyleSheet(f"""
        QMainWindow {{
            background-color: #2b2b2b;
        }}

        /* Top menu bar */
        QMenuBar {{
            background-color: #3a3a3a;
            color: #f0f0f0;
        }}
        QMenuBar::item {{
            padding: 4px 10px;
            background: transparent;
        }}
        QMenuBar::item:selected {{
            background-color: #4a4a4a;
        }}

        /* Status bar */
        QStatusBar {{
            background-color: #3a3a3a;
            color: #c0c0c0;
            border-top: 1px solid #444444;
        }}

        /* Central image area */
        #imageDisplay {{
            background-color: #262626;
            border-top: 1px solid #444444;
            border-bottom: 1px solid #444444;
            border-radius: 0;
            color: #dddddd;
        }}

        /* Dock panels */
        QDockWidget {{
            background-color: #262626;
            border: 1px solid #333333;
        }}
        QDockWidget::title {{
            padding: 4px 8px;
            background-color: #303030;
            color: #e0e0e0;
            border-bottom: 1px solid #444444;
        }}

        QDockWidget#leftSidebar {{
            border-right: 1px solid #444444;
        }}
        QDockWidget#rightSidebar {{
            border-left: 1px solid #444444;
        }}

        /* Bottom filmstrip */
        QListWidget#bottomFilmstrip {{
            background-color: #262626;
            border-top: 1px solid #444444;
            border-radius: 0;
        }}
        QListWidget#bottomFilmstrip::item {{
            border: none;
            padding: 2px;
        }}
        QListWidget#bottomFilmstrip::item:selected {{
            border: 2px solid {accent};
            background-color: #303030;
        }}

        QLabel {{
            color: #dddddd;
            background: transparent;
        }}

        QLabel#historyLabel {{
            font-weight: bold;
            color: #f0f0f0;
            padding: 2px 4px;
        }}

        QTreeWidget#editHistoryTree {{
            background-color: #262626;
            border-top: 1px solid #444444;
            border-radius: 0;
        }}
        QTreeWidget#editHistoryTree::item {{
            padding: 2px;
            margin: 1px 0;
        }}
        QTreeWidget#editHistoryTree::item:selected {{
            background-color: #404040;
            border: 1px solid {accent};
        }}

        QSlider::groove:horizontal {{
            height: 4px;
            background-color: #444444;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
            background-color: {accent};
        }}
        QSlider::sub-page:horizontal {{
            background-color: {accent};
            border-radius: 2px;
        }}
        QSlider::add-page:horizontal {{
            background-color: #444444;
            border-radius: 2px;
        }}

        QScrollBar:horizontal, QScrollBar:vertical {{
            background: #2b2b2b;
            border: none;
        }}
        QScrollBar::handle:horizontal, QScrollBar::handle:vertical {{
            background: #555555;
            min-width: 20px;
            min-height: 20px;
            border-radius: 3px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            background: none;
            border: none;
        }}
        """)

    # ---------- Image pipeline ----------

    def _ensure_image_cached(self, path: str):
        """Load full + preview images into cache if not already present."""
        if path in self._image_cache:
            return

        reader = QImageReader(path)
        reader.setAutoTransform(True)
        img = reader.read()
        if img.isNull():
            return

        # preview image for editing (downscaled for speed)
        max_w, max_h = 1200, 800
        preview = img.scaled(
            max_w, max_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ).convertToFormat(QImage.Format_RGBA8888)

        self._image_cache[path] = {"full": img, "preview": preview}

    def _show_image_version(self, path: str, params: dict, update_sliders: bool):
        """Show 'path' image with 'params' in the big preview."""
        self._ensure_image_cached(path)
        cache = self._image_cache.get(path)
        if not cache:
            return

        self._current_path = path
        self._preview_base_image = cache["preview"]

        if update_sliders:
            sat_val = params.get("saturation", 128)
            con_val = params.get("contrast", 128)

            self.saturation_slider.blockSignals(True)
            self.saturation_slider.setValue(sat_val)
            self.saturation_slider.blockSignals(False)
            self.saturation_value.setText(f"Saturation: {sat_val}")

            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(con_val)
            self.contrast_slider.blockSignals(False)
            self.contrast_value.setText(f"Contrast: {con_val}")

            self._current_params["saturation"] = sat_val
            self._current_params["contrast"] = con_val

        if not params:
            # Original
            self._current_pixmap = QPixmap.fromImage(self._preview_base_image)
            self._rescale_preview()
        else:
            self._apply_edit_params_to_current_image(params)

    def _apply_edit_params_to_current_image(self, params: dict):
        """Apply saturation + contrast to the preview image using NumPy."""
        if self._preview_base_image is None:
            return

        img = self._preview_base_image.copy()
        img = img.convertToFormat(QImage.Format_RGBA8888)

        w = img.width()
        h = img.height()
        bpl = img.bytesPerLine()

        ptr = img.bits()  # memoryview
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl // 4, 4))

        rgb = arr[:, :w, :3].astype(np.float32)

        # ---- Saturation ----
        sat_val = params.get("saturation", 128)
        sat_factor = 1.0 if sat_val == 128 else sat_val / 128.0

        lum = (
            0.299 * rgb[..., 0] +
            0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2]
        )[..., None]

        rgb = lum + (rgb - lum) * sat_factor

        # ---- Contrast ----
        con_val = params.get("contrast", 128)
        con_factor = 1.0 if con_val == 128 else con_val / 128.0
        mid = 128.0
        rgb = (rgb - mid) * con_factor + mid

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        arr[:, :w, :3] = rgb

        self._current_pixmap = QPixmap.fromImage(img)
        self._rescale_preview()

    def _rescale_preview(self):
        """Resize current pixmap to fit the display label."""
        if self._current_pixmap is None:
            return

        scaled = self._current_pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_display.setPixmap(scaled)
        self.image_display.setText("")

    # ---------- Thumbnail / folder handling ----------

    def _on_thumbnail_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        params = self._image_params.get(path, {})
        self._show_image_version(path, params, update_sliders=True)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", os.path.expanduser("~")
        )
        if folder:
            self.load_folder(folder)

    def _queue_thumb(self, path: str):
        if path in self._icon_cache or path in self._loading:
            return
        self._loading.add(path)
        task = _ThumbTask(path, self.thumbs.iconSize())
        task.signals.ready.connect(self._on_thumb_ready)
        self.pool.start(task)

    def _on_thumb_ready(self, path: str, icon: QIcon):
        self._icon_cache[path] = icon
        self._loading.discard(path)
        item = self._item_for_path.get(path)
        if item:
            item.setIcon(icon)

    def _ensure_visible_thumbs(self):
        vp = self.thumbs.viewport()
        if self.thumbs.count() == 0:
            return
        row_y = vp.height() // 2
        first = self.thumbs.indexAt(QPoint(0, row_y)).row()
        last = self.thumbs.indexAt(QPoint(vp.width() - 1, row_y)).row()

        if first < 0:
            first = 0
        if last < 0:
            approx_per_view = max(
                1,
                vp.width() // (self.thumbs.iconSize().width() + self.thumbs.spacing())
            )
            last = min(self.thumbs.count() - 1, first + approx_per_view + 4)

        first = max(0, first - 4)
        last = min(self.thumbs.count() - 1, last + 8)

        for i in range(first, last + 1):
            it = self.thumbs.item(i)
            self._queue_thumb(it.data(Qt.UserRole))

    def load_folder(self, folder_path: str):
        self.thumbs.clear()
        self._icon_cache.clear()
        self._loading.clear()
        self._item_for_path.clear()
        self._current_folder = folder_path

        # clear per-image state
        self._image_cache.clear()
        self._image_params.clear()
        self._image_parents.clear()
        self.history_tree.clear()
        self._current_path = None
        self._preview_base_image = None
        self._current_pixmap = None

        count = 0
        for name in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            item = QListWidgetItem(self._placeholder_icon, name)
            item.setData(Qt.UserRole, path)
            self.thumbs.addItem(item)
            self._item_for_path[path] = item
            count += 1

        self.statusBar().showMessage(f"Loading {count} images from {folder_path}")

        if count:
            self.thumbs.setCurrentRow(0)
            self._on_thumbnail_clicked(self.thumbs.item(0))
        self._ensure_visible_thumbs()

        self.statusBar().showMessage(f"Loaded {count} images from {folder_path}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_pixmap is not None:
            self._rescale_preview()
        self._ensure_visible_thumbs()

    # ---------- Export / import (.lrc) ----------

    def _get_active_folder(self):
        folder = getattr(self, "_current_folder", None)
        if folder and os.path.isdir(folder):
            return folder
        p = getattr(self, "_current_path", None)
        if p and os.path.isfile(p):
            return os.path.dirname(p)
        it = self.thumbs.currentItem()
        if it:
            p = it.data(Qt.UserRole)
            if p and os.path.isfile(p):
                return os.path.dirname(p)
        return None

    def _rebuild_history_from_params(self):
        """Rebuild the left tree from self._image_params (Active Edits)."""
        self.history_tree.clear()
        self._image_parents.clear()

        for path, params in self._image_params.items():
            if not os.path.isfile(path):
                continue

            self._ensure_image_cached(path)
            cache = self._image_cache.get(path)
            if not cache:
                continue

            # Set up context for _update_history_for_current_image
            self._current_path = path
            self._preview_base_image = cache["preview"]

            # Build _current_pixmap for this image
            if params:
                self._apply_edit_params_to_current_image(params)
            else:
                self._current_pixmap = QPixmap.fromImage(self._preview_base_image)

            # This creates the parent (Active Edit) + children
            self._update_history_for_current_image()

    def export_lrc(self):
        folder = self._get_active_folder()
        if not folder:
            QMessageBox.warning(self, "No Folder Loaded", "Please load a folder before exporting.")
            return

        folder = os.path.abspath(folder)
        current_rel = _relpath_or_same(self._current_path or "", folder)

        # Use _image_params as the canonical "active edit" mapping.
        # (Fall back to _edits if for some reason _image_params is empty.)
        source_edits = self._image_params if self._image_params else self._edits

        edits_rel = {}
        for abs_path, params in source_edits.items():
            try:
                if os.path.commonpath(
                    [os.path.abspath(abs_path), folder]
                ) != folder:
                    continue
            except Exception:
                continue
            rel = _relpath_or_same(abs_path, folder)
            edits_rel[rel] = dict(params)

        data = {
            "schema": "ECE 277 LightRoom Project",
            "version": LRC_VERSION,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "folder_path": folder,
            "current_image": current_rel,        # which image is currently active
            "edits": edits_rel,                  # per-image Active Edit params
        }

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Project",
            os.path.join(folder, "project.lrc"),
            "Lightroom Clone Project (*.lrc)",
        )
        if not path:
            return
        if not path.lower().endswith('.lrc'):
            path += '.lrc'
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Exported project to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export project: {e}")

    def import_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Project",
            os.path.expanduser("~"),
            "Lightroom Clone Project (*.lrc)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to import project: {e}")
            return

        if data.get("schema") != "ECE 277 LightRoom Project":
            QMessageBox.critical(self, "Import Failed", "Invalid project file.")
            return

        folder_abs = data.get("folder_path")
        if not folder_abs or not os.path.isdir(folder_abs):
            QMessageBox.critical(self, "Import Failed", "Project folder does not exist.")
            return

        # Load the folder (thumbnails, etc.)
        self.load_folder(folder_abs)

        # Restore per-image Active Edit params
        self._image_params.clear()
        self._edits.clear()

        imported_edits = data.get("edits", {})
        for rel_path, params in imported_edits.items():
            abs_path = _abspath_from_base(rel_path, folder_abs)
            self._image_params[abs_path] = dict(params)
            self._edits[abs_path] = dict(params)

        # Rebuild the left "Active Edit" tree
        self._rebuild_history_from_params()

        # Restore current image/view
        current_rel = data.get("current_image", "")
        current_abs = _abspath_from_base(current_rel, folder_abs) if current_rel else None

        if current_abs and os.path.isfile(current_abs):
            # select thumbnail if present
            thumb_item = self._item_for_path.get(current_abs)
            if thumb_item is not None:
                self.thumbs.setCurrentItem(thumb_item)

            params = self._image_params.get(current_abs, {})
            self._show_image_version(current_abs, params, update_sliders=True)

            # select parent node (Active Edit) in the tree
            parent = self._image_parents.get(current_abs)
            if parent is not None:
                self.history_tree.setCurrentItem(parent)
        else:
            # no current image in file – just leave whatever load_folder showed
            pass

        self.statusBar().showMessage(f"Imported project from {path}")

    def import_project_from_path(self, path: str):
        """Load .lrc project without a QFileDialog (used when opening from OS)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to import project: {e}")
            return

        if data.get("schema") != "ECE 277 LightRoom Project":
            QMessageBox.critical(self, "Import Failed", "Invalid project file.")
            return

        folder_abs = data.get("folder_path")
        if not folder_abs or not os.path.isdir(folder_abs):
            QMessageBox.critical(self, "Import Failed", "Project folder does not exist.")
            return

        self.load_folder(folder_abs)

        imported_edits = data.get("edits", {})
        self._image_params.clear()
        self._edits.clear()

        for rel_path, params in imported_edits.items():
            abs_path = _abspath_from_base(rel_path, folder_abs)
            self._image_params[abs_path] = dict(params)
            self._edits[abs_path] = dict(params)

        self._rebuild_history_from_params()

        current_rel = data.get("current_image", "")
        current_abs = _abspath_from_base(current_rel, folder_abs)

        if os.path.isfile(current_abs):
            it = self._item_for_path.get(current_abs)
            if it:
                self.thumbs.setCurrentItem(it)
                self._on_thumbnail_clicked(it)

        self.statusBar().showMessage(f"Imported project from {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # If app was launched by double-clicking a .lrc file:
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]

        if arg_path.lower().endswith(".lrc") and os.path.isfile(arg_path):
            try:
                window.import_project_from_path(arg_path)
            except Exception as e:
                print("Failed to auto-load project:", e)

    sys.exit(app.exec())
