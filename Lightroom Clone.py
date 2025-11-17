import os
import sys
import json
from datetime import datetime

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QFileDialog, QStatusBar, QListView, QSizePolicy, QLayout,
    QAbstractItemView, QStyle, QDockWidget, QSlider, QHBoxLayout, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QDialog, QFormLayout, QSpinBox, QDialogButtonBox,
    QMenu, QTabWidget, QTreeView, QFileSystemModel, QScrollArea, QPushButton, QFrame, QTabWidget
)

from PySide6.QtCore import Qt, QSize, QRunnable, QThreadPool, Signal, QObject, QPoint, QTimer
from PySide6.QtCore import QDir
from PySide6.QtGui import (
    QPixmap, QIcon, QImageReader, QFontMetrics, QImage, QPainter, QColor, QPen, QPolygon
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

class HistogramWidget(QWidget):
    """
    RGB overlaid histogram (0–255) of a QImage,
    drawn as filled colored shapes on a dark background.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist_r = None
        self._hist_g = None
        self._hist_b = None
        self.setMinimumHeight(120)

    def clear_histogram(self):
        self._hist_r = self._hist_g = self._hist_b = None
        self.update()

    def set_image(self, qimage: QImage):
        """Compute RGB histograms from the given image."""
        if qimage.isNull():
            self.clear_histogram()
            return

        img = qimage.convertToFormat(QImage.Format_RGBA8888)
        w = img.width()
        h = img.height()
        bpl = img.bytesPerLine()

        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl // 4, 4))
        rgb = arr[:, :w, :3].astype(np.uint8)

        # Separate channels
        r = rgb[..., 0].ravel()
        g = rgb[..., 1].ravel()
        b = rgb[..., 2].ravel()

        self._hist_r, _ = np.histogram(r, bins=256, range=(0, 255))
        self._hist_g, _ = np.histogram(g, bins=256, range=(0, 255))
        self._hist_b, _ = np.histogram(b, bins=256, range=(0, 255))

        self.update()

    def _draw_channel(self, painter: QPainter, hist, color: QColor, rect):
        """Draw one channel as a filled 'mountain' polygon."""
        if hist is None or hist.max() == 0:
            return

        w = rect.width()
        h = rect.height()
        max_count = float(hist.max())
        if max_count <= 0:
            return

        # Scale factors
        sx = w / 255.0
        sy = h / max_count

        # Build polygon: start bottom-left, go along bins, back down
        poly = QPolygon()
        # start at left-bottom
        poly.append(rect.bottomLeft())
        for i, count in enumerate(hist):
            x = rect.left() + i * sx
            y = rect.bottom() - count * sy
            poly.append(QPoint(int(x), int(y)))
        # end at right-bottom
        poly.append(rect.bottomRight())

        painter.setPen(QPen(color.darker(120)))
        painter.setBrush(color)
        painter.drawPolygon(poly)

class ThumbnailFileSystemModel(QFileSystemModel):
    """QFileSystemModel that shows thumbnails for supported image files,
    but falls back to default icons for very large folders so it doesn't
    blow up on thousands of images.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumb_cache = {}
        self._max_thumbs_per_folder = 500  # tweak if you want

    def _make_thumb_icon(self, path: str) -> QIcon | None:
        try:
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            reader.setScaledSize(QSize(48, 48))  # small preview
            img = reader.read()
            if img.isNull():
                return None
            return QIcon(QPixmap.fromImage(img))
        except Exception:
            return None

    def data(self, index, role=Qt.DecorationRole):
        # For everything except DecorationRole, use the default implementation.
        if role != Qt.DecorationRole:
            return super().data(index, role)

        if not index.isValid():
            return super().data(index, role)

        # Only draw an icon in the first column.
        if index.column() != 0:
            return super().data(index, role)

        # If this folder has a ton of items, don't generate thumbnails
        # (use default icons so the UI doesn't die).
        parent_index = index.parent()
        if parent_index.isValid():
            try:
                if self.rowCount(parent_index) > self._max_thumbs_per_folder:
                    return super().data(index, role)
            except Exception:
                # If rowCount blows up for some reason, just bail out to default.
                return super().data(index, role)

        path = self.filePath(index)
        if not path:
            return super().data(index, role)

        # Directories keep normal folder icons.
        if self.isDir(index):
            return super().data(index, role)

        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            # Non-image files (most are filtered out anyway)
            return super().data(index, role)

        # Cached thumbnail?
        icon = self._thumb_cache.get(path)
        if icon is None:
            icon = self._make_thumb_icon(path)
            if icon is None:
                # Fallback to default file icon if thumbnail failed
                icon = super().data(index, role)
            self._thumb_cache[path] = icon

        return icon

class CollapsibleSection(QWidget):
    """
    Simple DxO-style collapsible panel:
    [▼ Title]  (click header to expand/collapse)
    """
    def __init__(self, title: str, parent=None, start_collapsed=False):
        super().__init__(parent)

        self._header_btn = QPushButton(title)
        self._header_btn.setObjectName("sectionHeaderButton")
        self._header_btn.setCheckable(True)
        self._header_btn.setChecked(not start_collapsed)
        self._header_btn.clicked.connect(self._on_toggled)

        self._content = QWidget()
        self._content.setObjectName("sectionContent")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 4, 10, 8)
        self._content_layout.setSpacing(4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header_btn)
        layout.addWidget(self._content)

        if start_collapsed:
            self._content.setVisible(False)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)


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

            # project / image state
        self._current_folder = None
        self._current_path = None            # active image path
        self._current_pixmap = None          # current preview pixmap
        self._preview_base_image = None      # QImage preview base for active image

        # project file path (for autosave)
        self._project_path = None            # path to .lrc if saved, else None
        self._project_root = None  
        # project / image state
        self._current_folder = None
        self._current_path = None            # active image path
        self._current_pixmap = None          # current preview pixmap
        self._preview_base_image = None      # QImage preview base for active image

        # project file path (for autosave)
        self._project_path = None            # path to .lrc if saved, else None

        # autosave preferences
        self._autosave_interval_min = 5      # default: 5 minutes (0 = off)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._update_autosave_timer()


        # ---------- Bottom filmstrip (now as dock) ----------
        self.thumbs = QListWidget()
        self.thumbs.setObjectName("bottomFilmstrip")
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(100, 100))

        # KEY SETTINGS:
        self.thumbs.setFlow(QListView.LeftToRight)        # row-major grid
        self.thumbs.setWrapping(True)                     # wrap to next row
        self.thumbs.setResizeMode(QListWidget.Adjust)
        self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setSpacing(10)
        self.thumbs.setUniformItemSizes(True)

        # Vertical scroll only
        self.thumbs.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Central layout: under the image
        self.thumbs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.vbox.addWidget(self.thumbs, 0)

        # Use vertical scrollbar for lazy loading
        self.thumbs.verticalScrollBar().valueChanged.connect(
            lambda _: self._ensure_visible_thumbs()
        )

        icon_w = self.thumbs.iconSize().width()
        icon_h = self.thumbs.iconSize().height()
        fm = QFontMetrics(self.thumbs.font())
        text_h = fm.height()
        pad_h = 8

        cell_w = icon_w + 4     # some side padding
        cell_h = icon_h + text_h + pad_h
        self.thumbs.setGridSize(QSize(cell_w, cell_h))

        # reasonable minimum so it doesn't collapse
        self.thumbs.setMinimumHeight(cell_h * 2)

        self.thumbs.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)


        # Let the user resize this dock vertically
        self.thumbs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Track vertical scrolling for lazy thumbnail loading
        self.thumbs.verticalScrollBar().valueChanged.connect(
            lambda _: self._ensure_visible_thumbs()
        )
        self.thumbs.verticalScrollBar().valueChanged.connect(
            lambda _: self._ensure_visible_thumbs())

        icon_w = self.thumbs.iconSize().width()
        icon_h = self.thumbs.iconSize().height()
        fm = QFontMetrics(self.thumbs.font())
        text_h = fm.height()
        pad_h = 8
        cell_w = icon_w + 32   # a bit of left/right padding
        cell_h = icon_h + text_h + pad_h

        self.thumbs.setGridSize(QSize(cell_w, cell_h))

        # reasonable minimum height so it doesn't collapse to nothing
        self.thumbs.setMinimumHeight(cell_h * 2)

        self.thumbs.setResizeMode(QListView.Adjust)
        self.thumbs.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

                # --- connect signals for selecting images ---
        # click or double-click will load the image into the main view
        self.thumbs.itemClicked.connect(self._on_thumbnail_clicked)
        self.thumbs.itemActivated.connect(self._on_thumbnail_clicked)

        # also react when selection changes (keyboard navigation, etc.)
        self.thumbs.itemSelectionChanged.connect(self._on_thumbnail_selection_changed)


        self.setStatusBar(QStatusBar(self))

        # ---------- Thread pool for thumbs ----------
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(min(8, (os.cpu_count() or 4)))
        self._icon_cache = {}
        self._loading = set()
        self._item_for_path = {}

        # ---------- App menu (Lightroom Clone) ----------
        app_menu = self.menuBar().addMenu("&Lightroom Clone")

        self.act_preferences = app_menu.addAction("&Preferences...")
        # common on many apps; you can change to Ctrl+P if you prefer
        self.act_preferences.setShortcut("Ctrl+,")
        self.act_preferences.triggered.connect(self.show_preferences_dialog)

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
        # Sliders (we'll drop them into sections below)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 255)
        self.saturation_slider.setValue(128)
        self.saturation_value = QLabel("Saturation: 128")

        sat_row = QHBoxLayout()
        sat_row.addWidget(QLabel("Saturation"))
        sat_row.addWidget(self.saturation_slider, 1)
        sat_row.addWidget(self.saturation_value)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 255)
        self.contrast_slider.setValue(128)
        self.contrast_value = QLabel("Contrast: 128")

        con_row = QHBoxLayout()
        con_row.addWidget(QLabel("Contrast"))
        con_row.addWidget(self.contrast_slider, 1)
        con_row.addWidget(self.contrast_value)

        # Connect slider signals
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

        # Scroll area so we can stack many sections like DxO
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container.setObjectName("adjustmentsContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # --- Section 1: Histogram ---
        hist_section = CollapsibleSection("Histogram", self)
        hist_layout = hist_section.content_layout()
        # assuming you already have self.hist_widget; if not, we can add later
        if hasattr(self, "hist_widget"):
            hist_layout.addWidget(self.hist_widget)
        v.addWidget(hist_section)

        # --- Section 2: Basic Adjustments (your current sliders) ---
        basic_section = CollapsibleSection("Basic Adjustments", self)
        basic_layout = basic_section.content_layout()
        basic_layout.addLayout(sat_row)
        basic_layout.addLayout(con_row)
        v.addWidget(basic_section)

        # --- Placeholder sections for future DxO-style controls ---
        exposure_section = CollapsibleSection("Exposure Compensation", self, start_collapsed=True)
        exposure_layout = exposure_section.content_layout()
        exposure_layout.addWidget(QLabel("Exposure slider goes here"))
        v.addWidget(exposure_section)

        tone_section = CollapsibleSection("Tone Curve", self, start_collapsed=True)
        tone_layout = tone_section.content_layout()
        tone_layout.addWidget(QLabel("Tone curve widget goes here"))
        v.addWidget(tone_section)

        v.addStretch(1)

        scroll.setWidget(container)
        self.right_dock.setWidget(scroll)


        # ---------- Left dock: Edit tree ----------
        self.left_dock = QDockWidget("Active Edit", self)
        self.left_dock.setObjectName("leftSidebar")
        self.left_dock.setAttribute(Qt.WA_StyledBackground, True)

        self.left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.left_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

                # ----- Left dock content: tab widget with "Active Edit" + "Folders" -----
        self.left_tabs = QTabWidget()
        self.left_tabs.setObjectName("leftTabs")
        self.left_tabs.setTabPosition(QTabWidget.North)
        self.left_tabs.setDocumentMode(True)

        # --- Tab 1: Active Edit ---
        active_tab = QWidget()
        active_tab.setObjectName("glassPanelLeft")
        active_layout = QVBoxLayout(active_tab)
        active_layout.setContentsMargins(4, 4, 4, 4)
        active_layout.setSpacing(2)

        self.history_tree = QTreeWidget()
        self.history_tree.setObjectName("editHistoryTree")
        self.history_tree.setHeaderHidden(True)
        self.history_tree.setIconSize(QSize(32, 32))
        self.history_tree.itemClicked.connect(self._on_history_item_clicked)

        active_layout.addWidget(self.history_tree, 1)
        self.left_tabs.addTab(active_tab, "Active Edit")

        # --- Tab 2: Folders (file browser) ---
        folders_tab = QWidget()
        folders_tab.setObjectName("glassPanelLeft")
        folders_layout = QVBoxLayout(folders_tab)
        folders_layout.setContentsMargins(4, 4, 4, 4)
        folders_layout.setSpacing(2)

        self.fs_model = ThumbnailFileSystemModel(self)


        # Show all directories, but only files with supported image extensions
        filters = QDir.AllDirs | QDir.Drives | QDir.NoDotAndDotDot | QDir.Files
        self.fs_model.setFilter(filters)

        # Build name filters from IMAGE_EXTENSIONS, e.g. ["*.png", "*.jpg", ...]
        name_filters = [f"*{ext}" for ext in IMAGE_EXTENSIONS]
        self.fs_model.setNameFilters(name_filters)
        self.fs_model.setNameFilterDisables(False)  # hide non-matching files

        self.fs_model.setRootPath(QDir.homePath())


        self.fs_view = QTreeView()
        self.fs_view.setIconSize(QSize(16, 16))  # or 48, 48 if you want bigger

        self.fs_view.setObjectName("folderTree")
        self.fs_view.setModel(self.fs_model)
        self.fs_view.setRootIndex(self.fs_model.index(QDir.homePath()))
        self.fs_view.setHeaderHidden(False)
        self.fs_view.setSortingEnabled(True)
        self.fs_view.doubleClicked.connect(self._on_fs_double_clicked)

        folders_layout.addWidget(self.fs_view, 1)
        self.left_tabs.addTab(folders_tab, "Folders")

        self.left_dock.setWidget(self.left_tabs)


                # Right-click menu on Active Edit items
        self.history_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_tree.customContextMenuRequested.connect(
            self._on_history_context_menu
        )

        # Add docks to the main window
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

                # Set initial orientation based on starting dock area

        # Update orientation whenever the dock is moved

        # ---------- Window menu: toggles & layout ----------
        window_menu = self.menuBar().addMenu("&Window")

        self.act_restore_layout = window_menu.addAction("Restore Window Layout")
        self.act_restore_layout.triggered.connect(self.restore_default_layout)

        window_menu.addSeparator()
        window_menu.addAction(self.left_dock.toggleViewAction())
        window_menu.addAction(self.right_dock.toggleViewAction())
        self.act_toggle_filmstrip = window_menu.addAction("Show Filmstrip")
        self.act_toggle_filmstrip.setCheckable(True)
        self.act_toggle_filmstrip.setChecked(True)
        self.act_toggle_filmstrip.triggered.connect(self._on_toggle_filmstrip)


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

        self._copied_params = None

    def _on_toggle_filmstrip(self, checked: bool):
        self.thumbs.setVisible(checked)


    def _on_thumbnail_selection_changed(self):
        item = self.thumbs.currentItem()
        if item is not None:
            self._on_thumbnail_clicked(item)


    def _on_fs_double_clicked(self, index):
        """When a folder or image is double-clicked in the Folders tab."""
        path = self.fs_model.filePath(index)
        if not path:
            return

        if os.path.isdir(path):
            # Double-clicked a folder -> load it into the filmstrip
            self.load_folder(path)
        else:
            # Double-clicked a file -> load its folder,
            # and if it's an image, select it.
            folder = os.path.dirname(path)
            self.load_folder(folder)

            ext = os.path.splitext(path)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                item = self._item_for_path.get(path)
                if item:
                    self.thumbs.setCurrentItem(item)
                    self._on_thumbnail_clicked(item)



    def _update_histogram(self):
        """Refresh histogram based on the current displayed pixmap."""
        if not hasattr(self, "hist_widget"):
            return

        if self._current_pixmap is None:
            self.hist_widget.clear_histogram()
        else:
            self.hist_widget.set_image(self._current_pixmap.toImage())


    def _on_history_context_menu(self, pos: QPoint):
        """Show context menu for Active Edit items."""
        item = self.history_tree.itemAt(pos)
        if not item:
            return

        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path")
        params = data.get("params", {})

        if not path:
            return

        # Is this the top-level parent (Active Edit) entry?
        is_parent = (self._image_parents.get(path) is item)

        menu = QMenu(self)
        act_copy = menu.addAction("Copy Values")
        act_paste = menu.addAction("Paste Values")

        act_remove = None
        if is_parent:
            menu.addSeparator()
            act_remove = menu.addAction("Remove From Active Edit")

        chosen = menu.exec(self.history_tree.viewport().mapToGlobal(pos))
        if chosen is act_copy:
            # Copy whatever params are on this specific item (parent or child)
            self._copy_active_edit(path, params)
        elif chosen is act_paste:
            self._paste_active_edit(path)
        elif act_remove is not None and chosen is act_remove:
            self._remove_active_edit_image(path)


    def _copy_active_edit(self, path: str, params: dict):
        """Copy the params for this tree item (parent or child) into a buffer."""
        if not params:
            self._copied_params = {}
        else:
            self._copied_params = dict(params)

        self.statusBar().showMessage(
            f"Copied values for {os.path.basename(path)}", 2000
        )

    def _paste_active_edit(self, path: str):
        """Paste previously copied params onto this image."""
        if self._copied_params is None:
            self.statusBar().showMessage("Nothing to paste (no Active Edit copied yet).", 2000)
            return

        self._image_params[path] = dict(self._copied_params)
        self._edits[path] = dict(self._copied_params)

        # Update preview and tree
        self._show_image_version(path, self._copied_params, update_sliders=True)
        self._update_history_for_current_image()

        # autosave if project already has a file
        self.export_lrc(autosave=True)

        self.statusBar().showMessage(
            f"Pasted Active Edit values to {os.path.basename(path)}", 2000
        )

    def _remove_active_edit_image(self, path: str):
        """Remove this image from the Active Edit list entirely."""
        self._image_params.pop(path, None)
        self._edits.pop(path, None)

        parent_item = self._image_parents.pop(path, None)
        if parent_item is not None:
            idx = self.history_tree.indexOfTopLevelItem(parent_item)
            if idx >= 0:
                self.history_tree.takeTopLevelItem(idx)

        # Optionally revert preview to original if it's the current image
        if self._current_path == path:
            self._show_image_version(path, {}, update_sliders=True)

        # autosave updated project if we have a file
        self.export_lrc(autosave=True)

        self.statusBar().showMessage(
            f"Removed {os.path.basename(path)} from Active Edit.", 2000
        )



    def _update_autosave_timer(self):
        """Start/stop the autosave QTimer based on the current interval."""
        if self._autosave_interval_min is None or self._autosave_interval_min <= 0:
            self._autosave_timer.stop()
            return

        interval_ms = int(self._autosave_interval_min * 60_000)
        self._autosave_timer.start(interval_ms)

    def _on_filmstrip_location_changed(self, area: Qt.DockWidgetArea):
        """Called when the filmstrip dock is moved to a new area."""
        self._update_filmstrip_orientation(area)
        self._ensure_visible_thumbs()

    def _on_autosave_timer(self):
        """Timer callback: autosave only if project has already been saved."""
        if self._project_path is None:
            # user hasn't saved this project yet -> don't autosave
            return
        # use the existing export logic in "autosave" mode
        self.export_lrc(autosave=True)

    def show_preferences_dialog(self):
        """Show Preferences dialog for app settings (currently: autosave interval)."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Preferences - Lightroom Clone")

        layout = QFormLayout(dlg)

        # Autosave interval spinbox
        spin = QSpinBox(dlg)
        spin.setRange(0, 120)  # 0 = off, up to 2 hours
        spin.setSuffix(" min")
        spin.setValue(self._autosave_interval_min)
        layout.addRow("Autosave interval:", spin)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dlg
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.Accepted:
            new_val = spin.value()
            self._autosave_interval_min = new_val
            self._update_autosave_timer()
            if new_val <= 0:
                self.statusBar().showMessage("Autosave disabled.")
            else:
                self.statusBar().showMessage(
                    f"Autosave every {new_val} minute(s)."
                )


    def new_project(self):
        """Clear everything to start a new blank project."""
        # Reset project-related containers
        self._current_folder = None
        self._current_path = None
        self._preview_base_image = None
        self._current_pixmap = None
        self._project_path = None
        self._project_root = None

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

        if hasattr(self, "hist_widget"):
            self.hist_widget.clear_histogram()

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
            self._fs_prev_filmstrip_visible = self.thumbs.isVisible()

            # hide chrome
            if mb:
                mb.hide()
            if sb:
                sb.hide()

            self.left_dock.hide()
            self.right_dock.hide()
            self.thumbs.hide()

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
            self.thumbs.setVisible(self._fs_prev_filmstrip_visible)

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
            print("[EDIT COMMIT] Skipped: no current image.")
            return

        # Save combined params for this image = its "Active Edit"
        self._image_params[self._current_path] = dict(self._current_params)
        self._edits[self._current_path] = dict(self._current_params)

        print(f"[EDIT COMMIT] Params for {self._current_path}: {self._current_params}")

        # Update the tree for this image
        self._update_history_for_current_image()

        # Try autosave (only works if project has been saved once)
        self.export_lrc(autosave=True)

        self._update_histogram()


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
        # DxO-style cyan accent
        accent = "#00b4ff"
        accent_soft = "rgba(0, 180, 255, 130)"

        self.setStyleSheet(f"""
        /* ========== GLOBAL ==========: */

        QMainWindow {{
            background-color: #1c1d1f;
            color: #f0f0f0;
        }}

        * {{
            font-family: "Segoe UI", "Roboto", sans-serif;
            font-size: 9pt;
        }}

        QLabel {{
            color: #f0f0f0;
            background: transparent;
        }}

        /* ========== MENUBAR & MENUS ========== */

        QMenuBar {{
            background-color: #2b2c2f;
            color: #f5f5f5;
            padding: 0 6px;
            border-bottom: 1px solid #17181a;
        }}
        QMenuBar::item {{
            padding: 4px 10px;
            background: transparent;
        }}
        QMenuBar::item:selected {{
            background-color: #3a3b3f;
            border-radius: 2px;
        }}

        QMenu {{
            background-color: #2b2c2f;
            border: 1px solid #3c3d42;
            padding: 4px 0;
        }}
        QMenu::item {{
            padding: 4px 20px;
            color: #e6e6e8;
        }}
        QMenu::item:selected {{
            background-color: #3a3b41;
            color: {accent};
        }}
        
                /* Collapsible sections in Adjustments dock */
        QWidget#adjustmentsContainer {{
            background-color: #242528;
        }}

        QPushButton#sectionHeaderButton {{
            text-align: left;
            padding: 4px 10px;
            border: none;
            background-color: #303236;
            color: #f0f0f0;
            font-weight: 500;
        }}
        QPushButton#sectionHeaderButton::checked {{
            /* same bg; we could add an arrow icon later */
        }}
        QPushButton#sectionHeaderButton:hover {{
            background-color: #3a3c40;
        }}

        QWidget#sectionContent {{
            background-color: #26272b;
            border-bottom: 1px solid #303236;
        }}


        /* ========== STATUS BAR ========== */

        QStatusBar {{
            background-color: #222326;
            color: #b4b6ba;
            border-top: 1px solid #17181a;
        }}

        /* ========== CENTRAL IMAGE AREA ========== */

        #imageDisplay {{
            background-color: #141516;
            border: 1px solid #303236;
            border-radius: 0;
            color: #c3c5c8;
        }}

        /* ========== DOCKS (FRAMES + TITLE BARS) ========== */

        QDockWidget {{
            background-color: #242528;
            border: 1px solid #303236;
            border-radius: 0;
        }}

        QDockWidget::title {{
            padding: 4px 10px;
            background-color: #303236;
            color: #f4f4f5;
            border-bottom: 1px solid #101114;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        QDockWidget::close-button, QDockWidget::float-button {{
            background: transparent;
            border: none;
        }}
        QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
            background-color: #3c3e43;
            border-radius: 3px;
        }}

        QDockWidget#leftSidebar {{
            border-right: 1px solid #151618;
        }}
        QDockWidget#rightSidebar {{
            border-left: 1px solid #151618;
        }}
        QDockWidget#filmstripDock {{
            border-top: 1px solid #151618;
        }}

        /* INNER PANELS (SIDEBARS) */

        QWidget#glassPanelLeft,
        QWidget#glassPanelRight {{
            background-color: #242528;
            border: none;
            margin: 4px;
        }}

        /* ========== ACTIVE EDIT TREE ========== */

        QTreeWidget#editHistoryTree {{
            background-color: #242528;
            border: none;
            padding: 4px 2px;
        }}
        QTreeWidget#editHistoryTree::item {{
            padding: 2px 4px;
            margin: 1px 0;
        }}
        QTreeWidget#editHistoryTree::item:selected {{
            background-color: rgba(0, 180, 255, 40);
            border: 1px solid {accent};
            border-radius: 2px;
        }}
        QTreeWidget#editHistoryTree::item:hover:!selected {{
            background-color: #33353a;
            border-radius: 2px;
        }}

        /* ========== HISTOGRAM PANEL ========== */

        #histogramPanel {{
            background-color: #18191c;
            border: 1px solid #3a3c42;
            border-radius: 2px;
            min-height: 80px;
        }}

        /* Small label above histogram ("Histogram") */
        QDockWidget#rightSidebar QLabel {{
            color: #e0e2e6;
        }}

        /* ========== SLIDERS ========== */

        QSlider::groove:horizontal {{
            height: 4px;
            background-color: #34363c;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
            background-color: {accent};
            border: 1px solid #f5f5f5;
        }}
        QSlider::sub-page:horizontal {{
            background-color: {accent_soft};
            border-radius: 2px;
        }}
        QSlider::add-page:horizontal {{
            background-color: #242528;
            border-radius: 2px;
        }}

        /* ========== FILMSTRIP ========== */

        QListWidget#bottomFilmstrip {{
            background-color: #18191b;
            border-top: 1px solid #303236;
            border-radius: 0;
            padding: 3px;
        }}
        QListWidget#bottomFilmstrip::item {{
            border: 1px solid transparent;
            padding: 2px;
            border-radius: 2px;
        }}
        QListWidget#bottomFilmstrip::item:selected {{
            border: 1px solid {accent};
            background-color: rgba(0, 180, 255, 30);
        }}
        QListWidget#bottomFilmstrip::item:hover:!selected {{
            border: 1px solid #50535a;
            background-color: #26282d;
        }}

        /* ========== SCROLLBARS ========== */

        QScrollBar:horizontal, QScrollBar:vertical {{
            background-color: #18191b;
            border: none;
            margin: 0;
        }}
        QScrollBar::handle:horizontal, QScrollBar::handle:vertical {{
            background-color: #3b3d43;
            min-width: 20px;
            min-height: 20px;
            border-radius: 3px;
        }}
        QScrollBar::handle:hover {{
            background-color: #4a4d55;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            background: none;
            border: none;
        }}

        /* ========== PUSH BUTTONS (if/when you add them) ========== */

        QPushButton {{
            background-color: #2f3137;
            color: #f0f0f0;
            border-radius: 2px;
            border: 1px solid #3d4046;
            padding: 3px 10px;
        }}
        QPushButton:hover {{
            background-color: #3a3c42;
            border-color: {accent};
        }}
        QPushButton:pressed {{
            background-color: #25272b;
        }}

                /* Left dock tabs */
    QTabWidget#leftTabs::pane {{
            border: none;
            background-color: #242528;
        }}
        QTabBar::tab {{
            background-color: #2b2c2f;
            color: #d7d8dd;
            padding: 4px 10px;
            margin-right: 1px;
        }}
        QTabBar::tab:selected {{
            background-color: #383a3f;
            color: {accent};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: #33353a;
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

        # update histogram for whatever is now displayed
        self._update_histogram()

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

        col_x = vp.width() // 2
        first = self.thumbs.indexAt(QPoint(col_x, 0)).row()
        last = self.thumbs.indexAt(QPoint(col_x, vp.height() - 1)).row()

        if first < 0:
            first = 0
        if last < 0:
            approx_per_view = max(
                1,
                vp.height() // (self.thumbs.iconSize().height() + self.thumbs.spacing())
            )
            last = min(self.thumbs.count() - 1, first + approx_per_view + 4)

        first = max(0, first - 4)
        last = min(self.thumbs.count() - 1, last + 8)

        for i in range(first, last + 1):
            it = self.thumbs.item(i)
            self._queue_thumb(it.data(Qt.UserRole))


    def _ensure_project_root(self):
        """Set _project_root if it is not set yet.

        Uses the common parent of all edited images if possible,
        otherwise falls back to the current folder.
        """
        if self._project_root:
            return

        # Prefer paths from _image_params / _edits
        all_paths = list(self._image_params.keys() or self._edits.keys())
        if all_paths:
            abs_paths = [os.path.abspath(p) for p in all_paths]
            try:
                self._project_root = os.path.commonpath(abs_paths)
            except Exception:
                # If commonpath fails, just use the first image's folder
                self._project_root = os.path.dirname(abs_paths[0])
        else:
            # No images with edits yet: fall back to active folder if any
            folder = self._get_active_folder()
            if folder:
                self._project_root = os.path.abspath(folder)

    def _encode_project_path(self, abs_path: str) -> str:
        """Store a path in the .lrc file relative to project_root when possible."""
        abs_path = os.path.abspath(abs_path)
        root = self._project_root
        if root:
            try:
                common = os.path.commonpath([abs_path, root])
                if common == root:
                    return _relpath_or_same(abs_path, root)
            except Exception:
                pass
        # Fallback: store absolute path
        return abs_path

    def _decode_project_path(self, stored: str, root: str) -> str:
        """Convert stored relative/absolute path from .lrc back to absolute."""
        if os.path.isabs(stored):
            return os.path.abspath(stored)
        if root:
            return _abspath_from_base(stored, root)
        return os.path.abspath(stored)



    def load_folder(self, folder_path: str):
        self.thumbs.clear()
        self._icon_cache.clear()
        self._loading.clear()
        self._item_for_path.clear()
        self._current_folder = folder_path

        # Only reset the currently shown image; keep _image_params/history.
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

    def export_lrc(self, autosave: bool = False):
        """
        Save the current project to .lrc.

        - If autosave=True and we don't have a project file yet, we do nothing.
        - If we already have a project file, save directly to it (no dialog).
        - If no project file and autosave=False, show 'Save Project' dialog
          and remember chosen path for future autosaves.
        """
        # Use _image_params as the canonical "active edit" mapping.
        source_edits = self._image_params if self._image_params else self._edits

        if not source_edits and not self._current_path:
            if autosave:
                print("[AUTOSAVE] Skipped: no images to save.")
            else:
                QMessageBox.warning(
                    self,
                    "Nothing to Save",
                    "There are no images with edits in this project yet."
                )
            return

        # Make sure we have a project root so we can store relative paths
        self._ensure_project_root()

        # Use project root (or active folder) as default location for Save dialog
        default_folder = self._project_root or self._get_active_folder() or os.path.expanduser("~")

        # Build edits mapping with project-encoded paths (relative or absolute)
        edits_rel = {}
        for abs_path, params in source_edits.items():
            key = self._encode_project_path(abs_path)
            edits_rel[key] = dict(params)

        current_key = ""
        if self._current_path:
            current_key = self._encode_project_path(self._current_path)

        data = {
            "schema": "ECE 277 LightRoom Project",
            "version": LRC_VERSION,
           "created_utc": datetime.utcnow().isoformat() + "Z",
            "project_root": self._project_root,
            "current_image": current_key,  # which image is currently active
            "edits": edits_rel,            # per-image Active Edit params
        }
        # Decide where to save
        path = None

        if self._project_path is None:
            # No project file yet
            if autosave:
                # Autosave for unsaved project = do nothing
                print("[AUTOSAVE] Skipped: project has never been saved.")
                return

            # Manual save: ask user where to save
            print("[SAVE] No project_path yet; showing Save dialog...")
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Project",
                os.path.join(default_folder, "project.lrc"),
                "Lightroom Clone Project (*.lrc)",
            )
            if not path:
                print("[SAVE] User cancelled save dialog.")
                return
            if not path.lower().endswith(".lrc"):
                path += ".lrc"

            # remember for future autosaves
            self._project_path = path
            print(f"[SAVE] New project path set: {self._project_path}")
        else:
            # Project already has a file; save directly
            path = self._project_path
            if autosave:
                print(f"[AUTOSAVE] Saving to existing project: {path}")
            else:
                print(f"[SAVE] Saving to existing project: {path}")

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if autosave:
                self.statusBar().showMessage(f"Autosaved project to {path}", 2000)
            else:
                self.statusBar().showMessage(f"Saved project to {path}")
        except Exception as e:
            if autosave:
                print(f"[AUTOSAVE] ERROR while saving: {e}")
            else:
                QMessageBox.critical(self, "Save Failed", f"Failed to save project: {e}")

    def import_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
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

        # New format: project_root; fall back to legacy folder_path if needed
        project_root = data.get("project_root") or data.get("folder_path")
        if not project_root or not os.path.isdir(project_root):
            QMessageBox.critical(self, "Import Failed", "Project root folder does not exist.")
            return

        self._project_root = os.path.abspath(project_root)

        # Load some folder into the filmstrip – start at project_root by default
        self.load_folder(self._project_root)

        # Restore per-image Active Edit params
        self._image_params.clear()
        self._edits.clear()

        imported_edits = data.get("edits", {})
        for stored_path, params in imported_edits.items():
            abs_path = self._decode_project_path(stored_path, self._project_root)
            self._image_params[abs_path] = dict(params)
            self._edits[abs_path] = dict(params)

        # Rebuild the left "Active Edit" tree
        self._rebuild_history_from_params()

        # Restore current image/view
        current_rel = data.get("current_image", "")
        current_abs = _abspath_from_base(current_rel, folder_abs) if current_rel else None

        if current_abs and os.path.isfile(current_abs):
            thumb_item = self._item_for_path.get(current_abs)
            if thumb_item is not None:
                self.thumbs.setCurrentItem(thumb_item)
            params = self._image_params.get(current_abs, {})
            self._show_image_version(current_abs, params, update_sliders=True)
            parent = self._image_parents.get(current_abs)
            if parent is not None:
                self.history_tree.setCurrentItem(parent)

        # THIS is what autosave depends on
        self._project_path = os.path.abspath(path)
        print(f"[IMPORT] Project path set to: {self._project_path}")

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

        # NEW: use project_root if present, else fall back to old folder_path
        project_root = data.get("project_root") or data.get("folder_path")
        if not project_root or not os.path.isdir(project_root):
            QMessageBox.critical(self, "Import Failed", "Project root folder does not exist.")
            return

        # Remember project root for path encoding/decoding
        self._project_root = os.path.abspath(project_root)

        # Show something in the UI: load the root into the filmstrip
        self.load_folder(self._project_root)

        # Restore per-image Active Edit params
        imported_edits = data.get("edits", {})
        self._image_params.clear()
        self._edits.clear()

        for stored_path, params in imported_edits.items():
            # stored_path may be relative to project_root or absolute
            abs_path = self._decode_project_path(stored_path, self._project_root)
            self._image_params[abs_path] = dict(params)
            self._edits[abs_path] = dict(params)

        # Rebuild the left "Active Edit" tree for all images
        self._rebuild_history_from_params()

        # Restore current image/view
        current_rel = data.get("current_image", "")
        current_abs = (
            self._decode_project_path(current_rel, self._project_root)
            if current_rel
            else None
        )

        if current_abs and os.path.isfile(current_abs):
            it = self._item_for_path.get(current_abs)
            if it:
                self.thumbs.setCurrentItem(it)
                self._on_thumbnail_clicked(it)

            parent = self._image_parents.get(current_abs)
            if parent is not None:
                self.history_tree.setCurrentItem(parent)

        # Tell autosave what file to use
        self._project_path = os.path.abspath(path)
        print(f"[IMPORT FROM PATH] Project path set to: {self._project_path}")

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
