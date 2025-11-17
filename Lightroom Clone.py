import os
import sys
import json
from datetime import datetime

import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QFileDialog, QStatusBar, QListView, QSizePolicy, QLayout,
    QAbstractItemView, QStyle, QDockWidget, QSlider, QHBoxLayout, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QDialog, QFormLayout, QSpinBox, QDialogButtonBox,
    QMenu, QTabWidget, QTreeView, QFileSystemModel, QScrollArea, QPushButton, QFrame, QComboBox
)
from PySide6.QtCore import (
    Qt, QSize, QRunnable, QThreadPool, Signal, QObject, QPoint, QTimer, QDir, QEvent
)
from PySide6.QtGui import (
    QPixmap, QIcon, QImageReader, QFontMetrics, QImage, QPainter,
    QColor, QPen, QPolygon
)
from PIL import Image, ExifTags

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

# NEW: common RAW formats
RAW_EXTENSIONS = [
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".orf", ".rw2", ".raf",
    ".dng", ".pef", ".srw", ".rwl", ".3fr", ".erf", ".kdc",
    ".mrw", ".sr2", ".srf"
]

IMAGE_EXTENSIONS = IMAGE_EXTENSIONS + RAW_EXTENSIONS

LRC_VERSION = "1.0"


# ----------------- path helpers ----------------- #

def _relpath_or_same(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path


def _abspath_from_base(maybe_rel: str, base: str) -> str:
    if os.path.isabs(maybe_rel):
        return os.path.abspath(maybe_rel)
    return os.path.abspath(os.path.join(base, maybe_rel))

def _load_qimage_any(path: str) -> QImage | None:
    """
    Load a QImage from path.
    Uses rawpy for RAW files when available, otherwise QImageReader.
    Returns None on failure.
    """
    ext = os.path.splitext(path)[1].lower()

    # RAW path: use rawpy if present
    if ext in RAW_EXTENSIONS and rawpy is not None:
        try:
            with rawpy.imread(path) as raw:
                # 8-bit sRGB-ish output, no crazy auto-bright
                rgb = raw.postprocess(
                    output_bps=8,
                    no_auto_bright=True,
                    gamma=(2.2, 4.5),
                )
            h, w, ch = rgb.shape  # ch should be 3
            # stride = bytes per line = 3 * w
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            # make a deep copy so the numpy buffer can be freed
            return img.copy()
        except Exception as e:
            print(f"[RAW] Failed to decode {path}: {e}")
            return None

    # Fallback: normal image via Qt
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    img = reader.read()
    if img.isNull():
        return None
    return img



# ----------------- thumbnail worker ----------------- #

class _WorkerSignals(QObject):
    ready = Signal(str, QIcon)


class _ThumbTask(QRunnable):
    """Background thumbnail generator for the bottom filmstrip."""

    def __init__(self, path: str, size: QSize):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = _WorkerSignals()

    def run(self):
        img = _load_qimage_any(self.path)
        if img is None:
            icon = QIcon()
        else:
            img = img.scaled(self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon = QIcon(QPixmap.fromImage(img))
        self.signals.ready.emit(self.path, icon)



# ----------------- histogram widget ----------------- #

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
        self.setMinimumHeight(100)
        self.setObjectName("histogramPanel")

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

        sx = w / 255.0
        sy = h / max_count

        poly = QPolygon()
        poly.append(rect.bottomLeft())
        for i, count in enumerate(hist):
            x = rect.left() + i * sx
            y = rect.bottom() - count * sy
            poly.append(QPoint(int(x), int(y)))
        poly.append(rect.bottomRight())

        painter.setPen(QPen(color.darker(130)))
        painter.setBrush(color)
        painter.drawPolygon(poly)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        r = self.rect().adjusted(6, 6, -6, -6)
        painter.fillRect(r, QColor(24, 25, 28))

        # subtle grid lines
        painter.setPen(QPen(QColor(60, 62, 68)))
        for i in range(1, 4):
            y = r.top() + int(r.height() * i / 4)
            painter.drawLine(r.left(), y, r.right(), y)

        if self._hist_r is None:
            return

        painter.setOpacity(0.40)
        self._draw_channel(painter, self._hist_r, QColor(255, 80, 80, 180), r)
        painter.setOpacity(0.40)
        self._draw_channel(painter, self._hist_g, QColor(80, 255, 80, 180), r)
        painter.setOpacity(0.40)
        self._draw_channel(painter, self._hist_b, QColor(80, 160, 255, 180), r)


# ----------------- QFileSystemModel with thumbs ----------------- #

class ThumbnailFileSystemModel(QFileSystemModel):
    """
    QFileSystemModel that shows thumbnails for supported image files,
    but falls back to default icons for very large folders so it doesn't
    blow up on thousands of images.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumb_cache: dict[str, QIcon] = {}
        self._max_thumbs_per_folder = 500

    def _make_thumb_icon(self, path: str) -> QIcon | None:
        try:
            img = _load_qimage_any(path)
            if img is None:
                return None
            img = img.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return QIcon(QPixmap.fromImage(img))
        except Exception:
            return None



    def data(self, index, role=Qt.DecorationRole):
        if role != Qt.DecorationRole:
            return super().data(index, role)

        if not index.isValid() or index.column() != 0:
            return super().data(index, role)

        # Avoid thumbnailing gigantic folders
        parent_index = index.parent()
        if parent_index.isValid():
            try:
                if self.rowCount(parent_index) > self._max_thumbs_per_folder:
                    return super().data(index, role)
            except Exception:
                return super().data(index, role)

        path = self.filePath(index)
        if not path:
            return super().data(index, role)

        if self.isDir(index):
            return super().data(index, role)

        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return super().data(index, role)

        icon = self._thumb_cache.get(path)
        if icon is None:
            icon = self._make_thumb_icon(path)
            if icon is None:
                icon = super().data(index, role)
            self._thumb_cache[path] = icon

        return icon


# ----------------- Collapsible section ----------------- #

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
        self.resize(1200, 800)
        self.setMinimumSize(600, 400)

        central = QWidget(self)
        self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central)
        self.vbox.setContentsMargins(8, 8, 8, 8)
        self.vbox.setSpacing(8)

        # ---------- State ----------
        self._current_folder: str | None = None
        self._current_path: str | None = None
        self._current_pixmap: QPixmap | None = None
        self._preview_base_image: QImage | None = None

        self._image_cache: dict[str, dict] = {}   # path -> {"full": QImage, "preview": QImage}
        self._image_params: dict[str, dict] = {}  # path -> params
        self._image_parents: dict[str, QTreeWidgetItem] = {}
        self._edits: dict[str, dict] = {}         # for .lrc export
        self._metadata_labels: dict[str, QLabel] = {}

        self._project_path: str | None = None
        self._project_root: str | None = None

        # current parameters (Light + Color tabs)
        self._current_params = {
            "exposure": 128,
            "contrast": 128,
            "highlights": 128,
            "shadows": 128,
            "whites": 128,
            "blacks": 128,
            "saturation": 128,
            "temperature": 128,
            "tint": 128,
            "vibrance": 128,
        }

        # autosave
        self._autosave_interval_min = 5
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._update_autosave_timer()
                # --- zoom / pan state ---
        self._zoom_mode = "fit"   # "fit" or "manual"
        self._zoom_factor = 1.0   # 1.0 = 100%
        self._dragging = False
        self._drag_last_pos = QPoint()

        # ---------- Central image area ----------
                # ---------- Central image area + zoom toolbar ----------
        self.image_display = QLabel("No Image Loaded")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(200, 200)
        self.image_display.setObjectName("imageDisplay")

        # Scroll area to allow panning
        self.image_scroll = QScrollArea()
        self.image_scroll.setFrameShape(QFrame.NoFrame)
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setWidget(self.image_display)
        self.image_scroll.setAlignment(Qt.AlignCenter)

        # DxO-style zoom toolbar
        self.image_toolbar = QWidget()
        self.image_toolbar.setObjectName("imageToolBar")
        tb = QHBoxLayout(self.image_toolbar)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(6)

        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_100_btn = QPushButton("100%")
        self.zoom_out_btn = QPushButton("−")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)

        tb.addStretch(1)
        tb.addWidget(self.zoom_fit_btn)
        tb.addWidget(self.zoom_100_btn)
        tb.addSpacing(12)
        tb.addWidget(self.zoom_out_btn)
        tb.addWidget(self.zoom_label)
        tb.addWidget(self.zoom_in_btn)
        tb.addStretch(1)

        self.vbox.addWidget(self.image_toolbar, 0)
        self.vbox.addWidget(self.image_scroll, 1)
                # zoom button signals
        self.zoom_fit_btn.clicked.connect(self._on_zoom_fit)
        self.zoom_100_btn.clicked.connect(self._on_zoom_100)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)

        # drag-to-pan on the image
        self.image_display.installEventFilter(self)


        # ---------- Bottom filmstrip ----------
        self.thumbs = QListWidget()
        self.thumbs.setObjectName("bottomFilmstrip")
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(100, 100))
        self.thumbs.setFlow(QListView.LeftToRight)
        self.thumbs.setWrapping(True)
        self.thumbs.setResizeMode(QListWidget.Adjust)
        self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setSpacing(10)
        self.thumbs.setUniformItemSizes(True)
        self.thumbs.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        icon_w = self.thumbs.iconSize().width()
        icon_h = self.thumbs.iconSize().height()
        fm = QFontMetrics(self.thumbs.font())
        text_h = fm.height()
        pad_h = 8
        cell_w = icon_w + 32
        cell_h = icon_h + text_h + pad_h
        self.thumbs.setGridSize(QSize(cell_w, cell_h))
        self.thumbs.setMinimumHeight(cell_h * 2)
        self.thumbs.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

        self.vbox.addWidget(self.thumbs, 0)

        # thumbnail selection
        self.thumbs.itemClicked.connect(self._on_thumbnail_clicked)
        self.thumbs.itemActivated.connect(self._on_thumbnail_clicked)
        self.thumbs.itemSelectionChanged.connect(self._on_thumbnail_selection_changed)
        self.thumbs.verticalScrollBar().valueChanged.connect(
            lambda _: self._ensure_visible_thumbs()
        )

        self.setStatusBar(QStatusBar(self))

        # ---------- Thread pool for thumbs ----------
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(min(8, (os.cpu_count() or 4)))
        self._icon_cache: dict[str, QIcon] = {}
        self._loading: set[str] = set()
        self._item_for_path: dict[str, QListWidgetItem] = {}

        # placeholder icon
        self._placeholder_icon = QIcon(QPixmap(icon_w, icon_h))

        # ---------- Menus ----------
        self._build_menus()

        # ---------- Right dock: adjustments & metadata ----------
        self.hist_widget = HistogramWidget(self)

        self.right_dock = QDockWidget("Adjustments", self)
        self.right_dock.setObjectName("rightSidebar")
        self.right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.right_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self.adjust_tabs = QTabWidget()
        self.adjust_tabs.setObjectName("rightTabs")
        self.adjust_tabs.setTabPosition(QTabWidget.North)
        self.adjust_tabs.setIconSize(QSize(32, 32))
        self.adjust_tabs.setDocumentMode(True)

        self._build_light_tab()
        self._build_color_tab()
        self._build_detail_tab()
        self._build_geometry_tab()
        self._build_fx_tab()
        self._build_metadata_tab()

        self.right_dock.setWidget(self.adjust_tabs)

        # ---------- Left dock: active edit + folders ----------
        self.left_dock = QDockWidget("Panels", self)
        self.left_dock.setObjectName("leftSidebar")
        self.left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.left_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self.left_tabs = QTabWidget()
        self.left_tabs.setObjectName("leftTabs")
        self.left_tabs.setTabPosition(QTabWidget.North)
        self.left_tabs.setDocumentMode(True)

        self._build_active_edit_tab()
        self._build_folders_tab()

        self.left_dock.setWidget(self.left_tabs)

        # history context menu
        self.history_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_tree.customContextMenuRequested.connect(
            self._on_history_context_menu
        )

        # ---------- Add docks ----------
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        # ---------- Window menu: layout & visibility ----------
        self._build_window_menu()

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

        # Save default layout for restore
        self._default_layout_state = self.saveState()

        self._copied_params: dict | None = None

    # -------- menu builders -------- #

        # ---------- zoom helpers ----------

    def _set_zoom(self, factor: float, mode: str = "manual"):
        if self._current_pixmap is None:
            return

        factor = max(0.1, min(8.0, factor))  # clamp 10%–800%
        self._zoom_factor = factor
        self._zoom_mode = mode
        self._update_zoom_label()
        self._rescale_preview()

    def _update_zoom_label(self):
        if self._current_pixmap is None:
            self.zoom_label.setText("—")
            return

        # compute % from displayed pixmap size vs. original preview
        pm = self.image_display.pixmap()
        if pm is not None and not pm.isNull():
            ratio = pm.width() / self._current_pixmap.width()
        else:
            ratio = self._zoom_factor

        pct = int(round(ratio * 100))
        if self._zoom_mode == "fit":
            self.zoom_label.setText(f"{pct}%")
        else:
            self.zoom_label.setText(f"{pct}%")

    def _on_zoom_in(self):
        if self._current_pixmap is None:
            return
        self._set_zoom(self._zoom_factor * 1.25, mode="manual")

    def _on_zoom_out(self):
        if self._current_pixmap is None:
            return
        self._set_zoom(self._zoom_factor / 1.25, mode="manual")

    def _on_zoom_fit(self):
        if self._current_pixmap is None:
            return
        self._zoom_mode = "fit"
        self._rescale_preview()
        self._update_zoom_label()

    def _on_zoom_100(self):
        if self._current_pixmap is None:
            return
        self._set_zoom(1.0, mode="manual")


    def _build_menus(self):
        # Lightroom Clone (app) menu
        app_menu = self.menuBar().addMenu("&Lightroom Clone")

        self.act_preferences = app_menu.addAction("&Preferences...")
        self.act_preferences.setShortcut("Ctrl+,")
        self.act_preferences.triggered.connect(self.show_preferences_dialog)

        # File menu
        file_menu = self.menuBar().addMenu("&File")

        self.act_new_project = file_menu.addAction("&New Project")
        self.act_new_project.setShortcut("Ctrl+N")
        self.act_new_project.triggered.connect(self.new_project)

        self.act_open_project = file_menu.addAction("&Open Project...")
        self.act_open_project.setShortcut("Ctrl+O")
        self.act_open_project.triggered.connect(self.import_project)

        self.act_import_folder = file_menu.addAction("&Import Folder...")
        self.act_import_folder.setShortcut("Ctrl+I")
        self.act_import_folder.triggered.connect(self.open_folder)

        file_menu.addSeparator()

        self.act_save_project = file_menu.addAction("&Save Project")
        self.act_save_project.setShortcut("Ctrl+S")
        self.act_save_project.triggered.connect(self.export_lrc)
        

    # ---- right dock tabs ---- #

    def _make_slider_row(self, title: str):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(128)
        value_label = QLabel(f"{title}: 128")
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return slider, value_label, row

    def _build_light_tab(self):
        # sliders
        self.exposure_slider, self.exposure_value, exp_row = self._make_slider_row("Exposure")
        self.contrast_slider, self.contrast_value, con_row = self._make_slider_row("Contrast")
        self.highlights_slider, self.highlights_value, hi_row = self._make_slider_row("Highlights")
        self.shadows_slider, self.shadows_value, sh_row = self._make_slider_row("Shadows")
        self.whites_slider, self.whites_value, wh_row = self._make_slider_row("Whites")
        self.blacks_slider, self.blacks_value, bl_row = self._make_slider_row("Blacks")
        self.saturation_slider, self.saturation_value, sat_row = self._make_slider_row("Saturation")



        # connect signals
        self.exposure_slider.valueChanged.connect(self._on_exposure_changed)
        self.exposure_slider.sliderReleased.connect(self._on_edit_committed)

        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.contrast_slider.sliderReleased.connect(self._on_edit_committed)

        self.highlights_slider.valueChanged.connect(self._on_highlights_changed)
        self.highlights_slider.sliderReleased.connect(self._on_edit_committed)

        self.shadows_slider.valueChanged.connect(self._on_shadows_changed)
        self.shadows_slider.sliderReleased.connect(self._on_edit_committed)

        self.whites_slider.valueChanged.connect(self._on_whites_changed)
        self.whites_slider.sliderReleased.connect(self._on_edit_committed)

        self.blacks_slider.valueChanged.connect(self._on_blacks_changed)
        self.blacks_slider.sliderReleased.connect(self._on_edit_committed)

        self.saturation_slider.valueChanged.connect(self._on_saturation_changed)
        self.saturation_slider.sliderReleased.connect(self._on_edit_committed)

        # scrollable panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container.setObjectName("adjustmentsContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # Histogram section
        hist_section = CollapsibleSection("Histogram", self)
        hist_layout = hist_section.content_layout()
        hist_layout.addWidget(self.hist_widget)
        v.addWidget(hist_section)

        # Basic adjustments
        basic_section = CollapsibleSection("Basic Adjustments", self)
        basic_layout = basic_section.content_layout()
        basic_layout.addLayout(exp_row)
        basic_layout.addLayout(con_row)
        basic_layout.addLayout(hi_row)
        basic_layout.addLayout(sh_row)
        basic_layout.addLayout(wh_row)
        basic_layout.addLayout(bl_row)
        basic_layout.addSpacing(8)
        basic_layout.addLayout(sat_row)
        v.addWidget(basic_section)

        # placeholders for later
        exposure_section = CollapsibleSection("Exposure Compensation", self, start_collapsed=True)
        exposure_section.content_layout().addWidget(QLabel("Exposure slider goes here"))
        v.addWidget(exposure_section)

        tone_section = CollapsibleSection("Tone Curve", self, start_collapsed=True)
        tone_section.content_layout().addWidget(QLabel("Tone curve widget goes here"))
        v.addWidget(tone_section)

        v.addStretch(1)
        scroll.setWidget(container)

        icons_dir = os.path.join(os.path.dirname(__file__), "icons")

        def load_icon(name: str, fallback_role=None) -> QIcon:
            path = os.path.join(icons_dir, name)
            if os.path.exists(path):
                return QIcon(path)
            if fallback_role is not None:
                return self.style().standardIcon(fallback_role)
            return QIcon()

        light_icon = load_icon("light.png", QStyle.SP_DialogYesButton)
        self.adjust_tabs.addTab(scroll, light_icon, "")

        # keep loader for other tabs
        self._load_icon = load_icon

    def _build_color_tab(self):
        # --- create sliders ---
        self.temperature_slider, self.temperature_value, _ = self._make_slider_row("Temp")
        self.tint_slider, self.tint_value, _ = self._make_slider_row("Tint")
        self.vibrance_slider, self.vibrance_value, _ = self._make_slider_row("Vibrance")

        # object names for special QSS
        self.temperature_slider.setObjectName("tempSlider")
        self.tint_slider.setObjectName("tintSlider")
        self.vibrance_slider.setObjectName("vibranceSlider")

        # start labels at 0 like DxO HSL
        self.temperature_value.setText("0")
        self.tint_value.setText("0")
        self.vibrance_value.setText("0")

        # connect signals
        self.temperature_slider.valueChanged.connect(self._on_temperature_changed)
        self.temperature_slider.sliderReleased.connect(self._on_edit_committed)

        self.tint_slider.valueChanged.connect(self._on_tint_changed)
        self.tint_slider.sliderReleased.connect(self._on_edit_committed)

        self.vibrance_slider.valueChanged.connect(self._on_vibrance_changed)
        self.vibrance_slider.sliderReleased.connect(self._on_edit_committed)

        # --- layout: DxO-style COLOR panel ---
        color_page = QWidget()
        color_layout = QVBoxLayout(color_page)
        color_layout.setContentsMargins(6, 6, 6, 6)
        color_layout.setSpacing(8)

        # top bar: "Channel  [Master ▼]           [Reset]"
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        lbl_channel = QLabel("Channel")
        self.color_channel_combo = QComboBox()
        self.color_channel_combo.addItem("Master")
        self.color_channel_combo.setFixedWidth(110)

        self.color_reset_btn = QPushButton("Reset")
        self.color_reset_btn.setFixedWidth(70)
        self.color_reset_btn.clicked.connect(self._reset_color_sliders)

        top_bar.addWidget(lbl_channel)
        top_bar.addWidget(self.color_channel_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(self.color_reset_btn)

        color_layout.addLayout(top_bar)

        # collapsible "Hue / Saturation / Lightness" section
        hsl_section = CollapsibleSection("Hue / Saturation / Lightness", self)
        hsl_layout = hsl_section.content_layout()
        hsl_layout.setSpacing(6)

        def _make_color_row(label_text: str, slider: QSlider, value_label: QLabel):
            row = QHBoxLayout()
            row.setSpacing(6)

            lbl = QLabel(label_text)
            lbl.setMinimumWidth(80)
            lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            value_label.setMinimumWidth(40)
            value_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            return row

        # map our params to DxO-like names
        hsl_layout.addLayout(_make_color_row("Hue",        self.temperature_slider, self.temperature_value))
        hsl_layout.addLayout(_make_color_row("Saturation", self.tint_slider,        self.tint_value))
        hsl_layout.addLayout(_make_color_row("Luminance",  self.vibrance_slider,    self.vibrance_value))

        color_layout.addWidget(hsl_section)
        color_layout.addStretch(1)

        color_icon = self._load_icon("color.png", QStyle.SP_DialogOpenButton)
        self.adjust_tabs.addTab(color_page, color_icon, "")

    def _reset_color_sliders(self):
        """Reset only the Color tab sliders to neutral and re-apply edits."""
        for slider, label, key in [
            (self.temperature_slider, self.temperature_value, "temperature"),
            (self.tint_slider,        self.tint_value,        "tint"),
            (self.vibrance_slider,    self.vibrance_value,    "vibrance"),
        ]:
            slider.blockSignals(True)
            slider.setValue(128)
            slider.blockSignals(False)
            label.setText(f"{key.capitalize()}: 128")
            self._current_params[key] = 128

        if self._preview_base_image is not None:
            self._apply_edit_params_to_current_image(self._current_params)
            self._on_edit_committed()



    def _build_detail_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Detail / sharpening tools coming soon…"))
        layout.addStretch(1)

        icon = self._load_icon("detail.png", QStyle.SP_FileDialogDetailedView)
        self.adjust_tabs.addTab(page, icon, "")

    def _build_geometry_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Geometry / crop / perspective coming soon…"))
        layout.addStretch(1)

        icon = self._load_icon("geometry.png", QStyle.SP_ArrowUp)
        self.adjust_tabs.addTab(page, icon, "")

    def _build_fx_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Effects / vignettes / film look coming soon…"))
        layout.addStretch(1)

        icon = self._load_icon("fx.png", QStyle.SP_BrowserReload)
        self.adjust_tabs.addTab(page, icon, "")

    def _build_metadata_tab(self):
        info_page = QWidget()
        info_layout = QVBoxLayout(info_page)
        info_layout.setContentsMargins(4, 4, 4, 4)
        info_layout.setSpacing(6)

        meta_section = CollapsibleSection("Metadata", self)
        meta_layout = meta_section.content_layout()

        def _make_meta_row(label_text: str, key: str):
            row = QHBoxLayout()
            name_label = QLabel(label_text)
            name_label.setMinimumWidth(100)
            value_label = QLabel("—")
            value_label.setObjectName(f"meta_{key}")
            row.addWidget(name_label)
            row.addWidget(value_label, 1)
            self._metadata_labels[key] = value_label
            return row

        capture_row = QHBoxLayout()
        capture_row.addLayout(_make_meta_row("ISO", "iso"))
        capture_row.addLayout(_make_meta_row("f /", "aperture"))
        meta_layout.addLayout(capture_row)

        capture_row2 = QHBoxLayout()
        capture_row2.addLayout(_make_meta_row("Shutter", "shutter"))
        capture_row2.addLayout(_make_meta_row("EV", "ev"))
        meta_layout.addLayout(capture_row2)

        meta_layout.addLayout(_make_meta_row("Focal length", "focal"))
        meta_layout.addSpacing(8)
        meta_layout.addLayout(_make_meta_row("File", "file_name"))
        meta_layout.addLayout(_make_meta_row("Folder", "folder"))
        meta_layout.addLayout(_make_meta_row("Format", "format"))
        meta_layout.addLayout(_make_meta_row("Dimensions", "dimensions"))
        meta_layout.addLayout(_make_meta_row("Bit depth", "depth"))
        meta_layout.addLayout(_make_meta_row("File size", "filesize"))
        meta_layout.addLayout(_make_meta_row("Modified", "modified"))

        info_layout.addWidget(meta_section)
        info_layout.addStretch(1)

        info_icon = self._load_icon("info.png", QStyle.SP_MessageBoxInformation)
        self.adjust_tabs.addTab(info_page, info_icon, "")

    # ---- left dock ---- #

    def _build_active_edit_tab(self):
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

    def _build_folders_tab(self):
        folders_tab = QWidget()
        folders_tab.setObjectName("glassPanelLeft")
        folders_layout = QVBoxLayout(folders_tab)
        folders_layout.setContentsMargins(4, 4, 4, 4)
        folders_layout.setSpacing(2)

        self.fs_model = ThumbnailFileSystemModel(self)
        filters = QDir.AllDirs | QDir.Drives | QDir.NoDotAndDotDot | QDir.Files
        self.fs_model.setFilter(filters)
        name_filters = [f"*{ext}" for ext in IMAGE_EXTENSIONS]
        self.fs_model.setNameFilters(name_filters)
        self.fs_model.setNameFilterDisables(False)
        self.fs_model.setRootPath(QDir.homePath())

        self.fs_view = QTreeView()
        self.fs_view.setIconSize(QSize(16, 16))
        self.fs_view.setObjectName("folderTree")
        self.fs_view.setModel(self.fs_model)
        self.fs_view.setRootIndex(self.fs_model.index(QDir.homePath()))
        self.fs_view.setHeaderHidden(False)
        self.fs_view.setSortingEnabled(True)
        self.fs_view.doubleClicked.connect(self._on_fs_double_clicked)

        folders_layout.addWidget(self.fs_view, 1)
        self.left_tabs.addTab(folders_tab, "Folders")

    # ---- window menu ---- #

    def _build_window_menu(self):
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

        window_menu.addSeparator()
        self.act_fullscreen = window_menu.addAction("Full Screen")
        self.act_fullscreen.setCheckable(True)
        self.act_fullscreen.setShortcut("F11")
        self.act_fullscreen.triggered.connect(self.toggle_fullscreen)

    # ------------- styling ------------- #

    def _apply_styles(self):
        accent = "#00b4ff"

        self.setStyleSheet(f"""
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

        QStatusBar {{
            background-color: #222326;
            color: #b4b6ba;
            border-top: 1px solid #17181a;
        }}

        #imageDisplay {{
            background-color: #141516;
            border: 1px solid #303236;
            color: #c3c5c8;
        }}

        QDockWidget {{
            background-color: #242528;
            border: 1px solid #303236;
        }}
        QDockWidget::title {{
            padding: 4px 10px;
            background-color: #303236;
            color: #f4f4f5;
            border-bottom: 1px solid #101114;
        }}
        QDockWidget#leftSidebar {{
            border-right: 1px solid #151618;
        }}
        QDockWidget#rightSidebar {{
            border-left: 1px solid #151618;
        }}

        QWidget#glassPanelLeft {{
            background-color: #242528;
            border: none;
        }}

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

        #histogramPanel {{
            background-color: #18191c;
            border: 1px solid #3a3c42;
            border-radius: 2px;
            min-height: 80px;
        }}

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
        QPushButton#sectionHeaderButton:hover {{
            background-color: #3a3c40;
        }}
        QWidget#sectionContent {{
            background-color: #26272b;
            border-bottom: 1px solid #303236;
        }}

        /* Right dock icon tabs */
        QTabWidget#rightTabs::pane {{
            border: none;
            background-color: #25272b;
        }}
        QTabWidget#rightTabs::tab-bar {{
            alignment: center;
        }}
        QTabWidget#rightTabs > QTabBar::tab {{
            min-width: 32px;
            max-width: 32px;
            min-height: 32px;
            max-height: 32px;
            margin: 0;
            padding: 0;
            background-color: #2b2d31;
            border: 1px solid #33363a;
        }}
        QTabWidget#rightTabs > QTabBar::tab:selected {{
            background-color: #2f8fff;
            border-color: #2f8fff;
        }}
        QTabWidget#rightTabs > QTabBar::tab:hover:!selected {{
            background-color: #34373d;
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

                /* ========== SLIDERS (DxO-style) ========== */

        /* Groove: black → grey → white, rounded bar */
        QSlider::groove:horizontal {{
            border: 1px solid #111111;
            height: 10px;
            margin: 6px 10px;      /* spacing from labels */
            border-radius: 5px;
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0   #000000,
                stop: 0.5 #777777,
                stop: 1   #f5f5f5
            );
        }}

        /* Let the groove gradient show through – no colored fill */
        QSlider::sub-page:horizontal {{
            border: none;
            background: transparent;
        }}
        QSlider::add-page:horizontal {{
            border: none;
            background: transparent;
        }}

        /* Handle: small light-grey pill on top of the bar */
        QSlider::handle:horizontal {{
            background: #e8e8e8;
            border: 1px solid #3a3b3f;
            width: 14px;
            height: 14px;
            margin: -3px 0;        /* lets the knob overlap the bar a bit */
            border-radius: 7px;
        }}

        QSlider::handle:horizontal:hover {{
            background: #ffffff;
            border-color: #f0f0f0;
        }}

        QSlider::handle:horizontal:disabled {{
            background: #777777;
            border-color: #555555;
        }}

                /* ---- Color tab sliders (Hue / Sat / Lum) ---- */

        /* Hue bar (rainbow gradient) */
        QSlider#tempSlider::groove:horizontal {{
            border: 1px solid #111111;
            height: 10px;
            margin: 6px 10px;
            border-radius: 5px;
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0.0  #ff0000,
                stop: 0.16 #ffff00,
                stop: 0.33 #00ff00,
                stop: 0.50 #00ffff,
                stop: 0.66 #0000ff,
                stop: 0.83 #ff00ff,
                stop: 1.0 #ff0000
            );
        }}
        QSlider#tempSlider::sub-page:horizontal {{
            border: 1px solid rgba(0,0,0,80);
            border-radius: 5px;
            background: transparent;
        }}

        /* Sat / Lum: black→white ramp */
        QSlider#tintSlider::groove:horizontal,
        QSlider#vibranceSlider::groove:horizontal {{
            border: 1px solid #111111;
            height: 10px;
            margin: 6px 10px;
            border-radius: 5px;
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0   #000000,
                stop: 1   #ffffff
            );
        }}
        QSlider#tintSlider::sub-page:horizontal,
        QSlider#vibranceSlider::sub-page:horizontal {{
            border: 1px solid rgba(0,0,0,120);
            border-radius: 5px;
            background: rgba(255,255,255,40);
        }}


        QListWidget#bottomFilmstrip {{
            background-color: #18191b;
            border-top: 1px solid #303236;
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
        
        QWidget#imageToolBar {{
            background-color: #1e1f22;
            border: 1px solid #303236;
            border-bottom: none;
            padding: 2px 6px;
        }}
        QWidget#imageToolBar QPushButton {{
            min-width: 40px;
            padding: 2px 6px;
        }}
        QWidget#imageToolBar QLabel {{
            color: #e0e0e0;
        }}


        QScrollBar:horizontal, QScrollBar:vertical {{
            background-color: #18191b;
            border: none;
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
        """)
    # ---------- simple UI actions ----------

    def _on_toggle_filmstrip(self, checked: bool):
        self.thumbs.setVisible(checked)

    def _on_thumbnail_selection_changed(self):
        item = self.thumbs.currentItem()
        if item is not None:
            self._on_thumbnail_clicked(item)

    def _on_fs_double_clicked(self, index):
        path = self.fs_model.filePath(index)
        if not path:
            return

        if os.path.isdir(path):
            self.load_folder(path)
        else:
            folder = os.path.dirname(path)
            self.load_folder(folder)
            ext = os.path.splitext(path)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                item = self._item_for_path.get(path)
                if item:
                    self.thumbs.setCurrentItem(item)
                    self._on_thumbnail_clicked(item)

    # ---------- metadata helpers ----------
    def _read_exif_for_path(self, path: str):
        """
        Return a dict with a few EXIF fields we care about:
        iso, aperture, shutter, ev, focal.
        Values may be None if not present.
        """
        exif_info = {
            "iso": None,
            "aperture": None,
            "shutter": None,
            "ev": None,
            "focal": None,
        }

        try:
            with Image.open(path) as im:
                raw_exif = im._getexif() or {}
        except Exception:
            return exif_info

        # Convert tag IDs -> tag names
        data = {}
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            data[tag_name] = value

        # ISO
        if "ISOSpeedRatings" in data:
            exif_info["iso"] = str(data["ISOSpeedRatings"])

        # Aperture (FNumber is usually a rational)
        if "FNumber" in data:
            fnum = data["FNumber"]
            try:
                if isinstance(fnum, tuple) and len(fnum) == 2 and fnum[1] != 0:
                    f = fnum[0] / fnum[1]
                else:
                    f = float(fnum)
                exif_info["aperture"] = f"f/{f:.1f}"
            except Exception:
                exif_info["aperture"] = f"f/{fnum}"

        # Shutter speed: either ExposureTime or ShutterSpeedValue
        if "ExposureTime" in data:
            t = data["ExposureTime"]
            try:
                if isinstance(t, tuple) and len(t) == 2 and t[1] != 0:
                    num, den = t
                    if num >= den:
                        exif_info["shutter"] = f"{num/den:.3f}s"
                    else:
                        exif_info["shutter"] = f"1/{int(round(den/num))}s"
                else:
                    exif_info["shutter"] = f"{float(t):.4f}s"
            except Exception:
                exif_info["shutter"] = str(t)
        elif "ShutterSpeedValue" in data:
            # APEX value -> time
            try:
                val = data["ShutterSpeedValue"]
                if isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                    val = val[0] / val[1]
                t = 2 ** (-float(val))
                if t >= 1:
                    exif_info["shutter"] = f"{t:.3f}s"
                else:
                    exif_info["shutter"] = f"1/{int(round(1/t))}s"
            except Exception:
                pass

        # Exposure compensation (ExposureBiasValue) => EV
        if "ExposureBiasValue" in data:
            ev = data["ExposureBiasValue"]
            try:
                if isinstance(ev, tuple) and len(ev) == 2 and ev[1] != 0:
                    ev = ev[0] / ev[1]
                exif_info["ev"] = f"{ev:+.1f} EV"
            except Exception:
                exif_info["ev"] = str(ev)

        # Focal length
        if "FocalLength" in data:
            fl = data["FocalLength"]
            try:
                if isinstance(fl, tuple) and len(fl) == 2 and fl[1] != 0:
                    fl = fl[0] / fl[1]
                exif_info["focal"] = f"{fl:.0f} mm"
            except Exception:
                exif_info["focal"] = str(fl)

        return exif_info


    def _set_metadata_field(self, key: str, text: str):
        lbl = self._metadata_labels.get(key)
        if lbl is not None:
            lbl.setText(text)

    def _clear_metadata_fields(self):
        for lbl in self._metadata_labels.values():
            lbl.setText("—")

    def _update_metadata_for_current(self):
        path = self._current_path
        if not path or not os.path.isfile(path):
            self._clear_metadata_fields()
            return

        try:
            st = os.stat(path)
            size_mb = st.st_size / (1024 * 1024)
            modified_dt = datetime.fromtimestamp(st.st_mtime)
            modified_str = modified_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            size_mb = 0.0
            modified_str = "—"

        self._ensure_image_cached(path)
        cache = self._image_cache.get(path)
        if cache:
            img = cache["full"]
            w = img.width()
            h = img.height()
            depth = img.depth()
        else:
            w = h = depth = 0

        ext = os.path.splitext(path)[1].upper().lstrip(".") or "Unknown"

        self._set_metadata_field("file_name", os.path.basename(path))
        self._set_metadata_field("folder", os.path.dirname(path))
        self._set_metadata_field("format", ext)
        self._set_metadata_field("dimensions", f"{w} × {h}" if w and h else "—")
        self._set_metadata_field("depth", f"{depth} bit" if depth else "—")
        self._set_metadata_field("filesize", f"{size_mb:.1f} MB" if size_mb else "—")
        self._set_metadata_field("modified", modified_str)

                # --- EXIF fields (ISO, aperture, shutter, EV, focal length) ---
        exif = self._read_exif_for_path(path)


        if "iso" in self._metadata_labels:
            self._metadata_labels["iso"].setText(exif["iso"] or "—")
        if "aperture" in self._metadata_labels:
            self._metadata_labels["aperture"].setText(exif["aperture"] or "—")
        if "shutter" in self._metadata_labels:
            self._metadata_labels["shutter"].setText(exif["shutter"] or "—")
        if "ev" in self._metadata_labels:
            self._metadata_labels["ev"].setText(exif["ev"] or "—")
        if "focal" in self._metadata_labels:
            self._metadata_labels["focal"].setText(exif["focal"] or "—")


    # ---------- histogram update ----------

    def _update_histogram(self):
        if self._current_pixmap is None:
            self.hist_widget.clear_histogram()
        else:
            self.hist_widget.set_image(self._current_pixmap.toImage())

    # ---------- history tree context menu ----------

    def _on_history_context_menu(self, pos: QPoint):
        item = self.history_tree.itemAt(pos)
        if not item:
            return

        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path")
        params = data.get("params", {})

        if not path:
            return

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
            self._copy_active_edit(path, params)
        elif chosen is act_paste:
            self._paste_active_edit(path)
        elif act_remove is not None and chosen is act_remove:
            self._remove_active_edit_image(path)

    def _copy_active_edit(self, path: str, params: dict):
        self._copied_params = dict(params) if params else {}
        self.statusBar().showMessage(
            f"Copied values for {os.path.basename(path)}", 2000
        )

    def _paste_active_edit(self, path: str):
        if self._copied_params is None:
            self.statusBar().showMessage("Nothing to paste (no Active Edit copied yet).", 2000)
            return

        self._image_params[path] = dict(self._copied_params)
        self._edits[path] = dict(self._copied_params)

        self._show_image_version(path, self._copied_params, update_sliders=True)
        self._update_history_for_current_image()
        self.export_lrc(autosave=True)

        self.statusBar().showMessage(
            f"Pasted Active Edit values to {os.path.basename(path)}", 2000
        )

    def _remove_active_edit_image(self, path: str):
        self._image_params.pop(path, None)
        self._edits.pop(path, None)

        parent_item = self._image_parents.pop(path, None)
        if parent_item is not None:
            idx = self.history_tree.indexOfTopLevelItem(parent_item)
            if idx >= 0:
                self.history_tree.takeTopLevelItem(idx)

        if self._current_path == path:
            self._show_image_version(path, {}, update_sliders=True)

        self.export_lrc(autosave=True)
        self.statusBar().showMessage(
            f"Removed {os.path.basename(path)} from Active Edit.", 2000
        )

    # ---------- autosave ----------

    def _update_autosave_timer(self):
        if self._autosave_interval_min is None or self._autosave_interval_min <= 0:
            self._autosave_timer.stop()
            return
        interval_ms = int(self._autosave_interval_min * 60_000)
        self._autosave_timer.start(interval_ms)

    def _on_autosave_timer(self):
        if self._project_path is None:
            return
        self.export_lrc(autosave=True)

    def show_preferences_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Preferences - Lightroom Clone")
        layout = QFormLayout(dlg)

        spin = QSpinBox(dlg)
        spin.setRange(0, 120)
        spin.setSuffix(" min")
        spin.setValue(self._autosave_interval_min)
        layout.addRow("Autosave interval:", spin)

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

    # ---------- new project ----------

    def new_project(self):
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

        self.thumbs.clear()
        self.history_tree.clear()
        self.image_display.clear()
        self.image_display.setText("No Image Loaded")

        for slider, label, key in [
            (self.exposure_slider, self.exposure_value, "exposure"),
            (self.contrast_slider, self.contrast_value, "contrast"),
            (self.highlights_slider, self.highlights_value, "highlights"),
            (self.shadows_slider, self.shadows_value, "shadows"),
            (self.whites_slider, self.whites_value, "whites"),
            (self.blacks_slider, self.blacks_value, "blacks"),
            (self.saturation_slider, self.saturation_value, "saturation"),
            (self.temperature_slider, self.temperature_value, "temperature"),
            (self.tint_slider, self.tint_value, "tint"),
            (self.vibrance_slider, self.vibrance_value, "vibrance"),
        ]:
            slider.blockSignals(True)
            slider.setValue(128)
            slider.blockSignals(False)
            label.setText(f"{key.capitalize()}: 128")
            self._current_params[key] = 128

        self._clear_metadata_fields()
        self.hist_widget.clear_histogram()
        self.statusBar().showMessage("New blank project created.")

    # ---------- fullscreen ----------

    def toggle_fullscreen(self, checked: bool = False):
        if not self._is_fullscreen_mode:
            self._is_fullscreen_mode = True
            self.act_fullscreen.blockSignals(True)
            self.act_fullscreen.setChecked(True)
            self.act_fullscreen.blockSignals(False)

            self._fs_prev_geometry = self.saveGeometry()
            mb = self.menuBar()
            sb = self.statusBar()

            self._fs_prev_menubar_visible = mb.isVisible() if mb else True
            self._fs_prev_statusbar_visible = sb.isVisible() if sb else True
            self._fs_prev_left_visible = self.left_dock.isVisible()
            self._fs_prev_right_visible = self.right_dock.isVisible()
            self._fs_prev_filmstrip_visible = self.thumbs.isVisible()

            if mb:
                mb.hide()
            if sb:
                sb.hide()
            self.left_dock.hide()
            self.right_dock.hide()
            self.thumbs.hide()
            self.showFullScreen()
        else:
            self._is_fullscreen_mode = False
            self.act_fullscreen.blockSignals(True)
            self.act_fullscreen.setChecked(False)
            self.act_fullscreen.blockSignals(False)

            self.showNormal()
            if self._fs_prev_geometry is not None:
                self.restoreGeometry(self._fs_prev_geometry)

            mb = self.menuBar()
            sb = self.statusBar()
            if mb:
                mb.setVisible(self._fs_prev_menubar_visible)
            if sb:
                sb.setVisible(self._fs_prev_statusbar_visible)
                self.act_toggle_statusbar.blockSignals(True)
                self.act_toggle_statusbar.setChecked(self._fs_prev_statusbar_visible)
                self.act_toggle_statusbar.blockSignals(False)

            if self._fs_prev_left_visible:
                self.left_dock.show()
            if self._fs_prev_right_visible:
                self.right_dock.show()
            self.thumbs.setVisible(self._fs_prev_filmstrip_visible)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._is_fullscreen_mode:
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    # ---------- window layout helpers ----------

    def restore_default_layout(self):
        if hasattr(self, "_default_layout_state"):
            self.restoreState(self._default_layout_state)

    def _on_toggle_statusbar(self, checked: bool):
        sb = self.statusBar()
        if sb is not None:
            sb.setVisible(checked)

    # ---------- slider callbacks ----------



    def _on_saturation_changed(self, value: int):
        self.saturation_value.setText(f"Saturation: {value}")
        self._current_params["saturation"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_contrast_changed(self, value: int):
        self.contrast_value.setText(f"Contrast: {value}")
        self._current_params["contrast"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_exposure_changed(self, value: int):
        self.exposure_value.setText(f"Exposure: {value}")
        self._current_params["exposure"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_highlights_changed(self, value: int):
        self.highlights_value.setText(f"Highlights: {value}")
        self._current_params["highlights"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_shadows_changed(self, value: int):
        self.shadows_value.setText(f"Shadows: {value}")
        self._current_params["shadows"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_whites_changed(self, value: int):
        self.whites_value.setText(f"Whites: {value}")
        self._current_params["whites"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_blacks_changed(self, value: int):
        self.blacks_value.setText(f"Blacks: {value}")
        self._current_params["blacks"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_temperature_changed(self, value: int):
        self.temperature_value.setText(f"Temp: {value}")
        self._current_params["temperature"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_tint_changed(self, value: int):
        self.tint_value.setText(f"Tint: {value}")
        self._current_params["tint"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_vibrance_changed(self, value: int):
        self.vibrance_value.setText(f"Vibrance: {value}")
        self._current_params["vibrance"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_edit_committed(self):
        if not self._current_path or self._preview_base_image is None:
            return

        self._image_params[self._current_path] = dict(self._current_params)
        self._edits[self._current_path] = dict(self._current_params)

        self._update_history_for_current_image()
        self.export_lrc(autosave=True)
        self._update_histogram()

    # ---------- history tree logic ----------

    def _update_history_for_current_image(self):
        path = self._current_path
        if not path or self._current_pixmap is None:
            return

        if path not in self._image_parents:
            parent = QTreeWidgetItem(self.history_tree)
            parent.setText(0, os.path.basename(path))
            self._image_parents[path] = parent
        else:
            parent = self._image_parents[path]
            parent.takeChildren()

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

        parent.setIcon(0, edited_icon)
        parent.setData(0, Qt.UserRole, {"path": path, "params": dict(combined_params)})

        child_orig = QTreeWidgetItem(parent)
        child_orig.setText(0, "Original")
        child_orig.setIcon(0, orig_icon)
        child_orig.setData(0, Qt.UserRole, {"path": path, "params": {}})

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

    # ---------- Image pipeline ----------

    def _ensure_image_cached(self, path: str):
        if path in self._image_cache:
            return

        img = _load_qimage_any(path)
        if img is None:
            return

        max_w, max_h = 1200, 800
        preview = img.scaled(
            max_w, max_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ).convertToFormat(QImage.Format_RGBA8888)

        preview = img.convertToFormat(QImage.Format_RGBA8888)

        self._image_cache[path] = {"full": img, "preview": preview}



    def _show_image_version(self, path: str, params: dict, update_sliders: bool):
        self._ensure_image_cached(path)
        cache = self._image_cache.get(path)
        if not cache:
            return

        self._current_path = path
        self._preview_base_image = cache["preview"]
                # recompute base arrays for this image once
        self._prepare_base_arrays()


        if update_sliders:
            def get(name): return params.get(name, 128)

            self.saturation_slider.blockSignals(True)
            self.saturation_slider.setValue(get("saturation"))
            self.saturation_slider.blockSignals(False)
            self.saturation_value.setText(f"Saturation: {get('saturation')}")

            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(get("contrast"))
            self.contrast_slider.blockSignals(False)
            self.contrast_value.setText(f"Contrast: {get('contrast')}")

            self.exposure_slider.blockSignals(True)
            self.exposure_slider.setValue(get("exposure"))
            self.exposure_slider.blockSignals(False)
            self.exposure_value.setText(f"Exposure: {get('exposure')}")

            self.highlights_slider.blockSignals(True)
            self.highlights_slider.setValue(get("highlights"))
            self.highlights_slider.blockSignals(False)
            self.highlights_value.setText(f"Highlights: {get('highlights')}")

            self.shadows_slider.blockSignals(True)
            self.shadows_slider.setValue(get("shadows"))
            self.shadows_slider.blockSignals(False)
            self.shadows_value.setText(f"Shadows: {get('shadows')}")

            self.whites_slider.blockSignals(True)
            self.whites_slider.setValue(get("whites"))
            self.whites_slider.blockSignals(False)
            self.whites_value.setText(f"Whites: {get('whites')}")

            self.blacks_slider.blockSignals(True)
            self.blacks_slider.setValue(get("blacks"))
            self.blacks_slider.blockSignals(False)
            self.blacks_value.setText(f"Blacks: {get('blacks')}")

            self.temperature_slider.blockSignals(True)
            self.temperature_slider.setValue(get("temperature"))
            self.temperature_slider.blockSignals(False)
            self.temperature_value.setText(f"Temp: {get('temperature')}")

            self.tint_slider.blockSignals(True)
            self.tint_slider.setValue(get("tint"))
            self.tint_slider.blockSignals(False)
            self.tint_value.setText(f"Tint: {get('tint')}")

            self.vibrance_slider.blockSignals(True)
            self.vibrance_slider.setValue(get("vibrance"))
            self.vibrance_slider.blockSignals(False)
            self.vibrance_value.setText(f"Vibrance: {get('vibrance')}")

            self._current_params.update({
                "saturation": get("saturation"),
                "contrast": get("contrast"),
                "exposure": get("exposure"),
                "highlights": get("highlights"),
                "shadows": get("shadows"),
                "whites": get("whites"),
                "blacks": get("blacks"),
                "temperature": get("temperature"),
                "tint": get("tint"),
                "vibrance": get("vibrance"),
            })
                    # --- rendering caches / performance ---
            self._base_rgb = None          # float32 RGB of current base image
            self._base_lum = None          # float32 luminance
            self._L_norm = None            # L = lum / 255.0
            self._tone_masks = {}          # precomputed masks for hi/sh/wh/bl

            self._render_timer = QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.timeout.connect(self._render_current_params)

            self._update_metadata_for_current()

        if not params:
            self._current_pixmap = QPixmap.fromImage(self._preview_base_image)
            self._rescale_preview()
        else:
            self._apply_edit_params_to_current_image(params)

        self._update_histogram()

    def _prepare_base_arrays(self):
        """Compute base RGB, luminance and tone masks once per base image."""
        if self._preview_base_image is None:
            self._base_rgb = None
            self._base_lum = None
            self._L_norm = None
            self._tone_masks = {}
            return

        img = self._preview_base_image.convertToFormat(QImage.Format_RGBA8888)
        w = img.width()
        h = img.height()
        bpl = img.bytesPerLine()

        ptr = img.bits()
        # view of the QImage buffer
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl // 4, 4))
        rgb = arr[:, :w, :3].astype(np.float32).copy()  # copy so it's independent

        lum = (
            0.299 * rgb[..., 0] +
            0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2]
        )

        L = lum / 255.0
        highlights_mask = np.clip((L - 0.5) / 0.5, 0.0, 1.0)
        shadows_mask    = np.clip((0.5 - L) / 0.5, 0.0, 1.0)
        whites_mask     = np.clip((L - 0.8) / 0.2, 0.0, 1.0)
        blacks_mask     = np.clip((0.2 - L) / 0.2, 0.0, 1.0)

        self._base_rgb = rgb
        self._base_lum = lum
        self._L_norm = L
        self._tone_masks = {
            "highlights": highlights_mask,
            "shadows": shadows_mask,
            "whites": whites_mask,
            "blacks": blacks_mask,
        }

    def _apply_edit_params_to_current_image(self, params: dict):
        if self._preview_base_image is None:
            return

        # Make sure we have precomputed arrays
        if self._base_rgb is None or self._base_lum is None:
            self._prepare_base_arrays()
            if self._base_rgb is None:
                return

        # Start from base RGB every time
        rgb = self._base_rgb.copy()
        lum = self._base_lum
        L = self._L_norm
        lum3 = lum[..., None]

        img = self._preview_base_image.convertToFormat(QImage.Format_RGBA8888)
        w = img.width()
        h = img.height()
        bpl = img.bytesPerLine()

        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl // 4, 4))
        # we’ll write into arr[:, :w, :3] at the end

        # ===== adjustments (same logic as before, just using cached lum/masks) =====

        # temperature
        temp_val = params.get("temperature", 128)
        if temp_val != 128:
            t = (temp_val - 128.0) / 128.0 * 0.5
            rgb[..., 0] *= (1.0 + t)
            rgb[..., 2] *= (1.0 - t)

        # tint
        tint_val = params.get("tint", 128)
        if tint_val != 128:
            tt = (tint_val - 128.0) / 128.0 * 0.5
            rgb[..., 1] *= (1.0 - tt)
            rgb[..., 0] *= (1.0 + tt * 0.5)
            rgb[..., 2] *= (1.0 + tt * 0.5)

        # exposure
        exp_val = params.get("exposure", 128)
        if exp_val != 128:
            exp_stops = (exp_val - 128) / 128.0 * 2.0
            exp_factor = 2.0 ** exp_stops
            rgb *= exp_factor

        # contrast
        con_val = params.get("contrast", 128)
        if con_val != 128:
            con_factor = con_val / 128.0
            mid = 128.0
            rgb = (rgb - mid) * con_factor + mid

        # tone sliders – reuse precomputed masks
        def tone_strength(val, scale=0.5):
            return (val - 128.0) / 128.0 * scale

        hi_s = tone_strength(params.get("highlights", 128), 0.6)
        sh_s = tone_strength(params.get("shadows", 128), 0.6)
        wh_s = tone_strength(params.get("whites", 128), 0.8)
        bl_s = tone_strength(params.get("blacks", 128), 0.8)

        if any(abs(x) > 1e-3 for x in (hi_s, sh_s, wh_s, bl_s)):
            hm = self._tone_masks["highlights"]
            sm = self._tone_masks["shadows"]
            wm = self._tone_masks["whites"]
            bm = self._tone_masks["blacks"]

            rgb += hi_s * 255.0 * hm[..., None]
            rgb += sh_s * 255.0 * sm[..., None]
            rgb += wh_s * 255.0 * wm[..., None]
            rgb += bl_s * 255.0 * bm[..., None]

        # saturation
        sat_val = params.get("saturation", 128)
        if sat_val != 128:
            sat_factor = sat_val / 128.0
            rgb = lum3 + (rgb - lum3) * sat_factor

        # vibrance
        vib_val = params.get("vibrance", 128)
        if vib_val != 128:
            vib_amount = (vib_val - 128.0) / 128.0 * 0.75
            sat_dist = np.mean(np.abs(rgb - lum3), axis=-1) / 255.0
            weight = (1.0 - sat_dist)[..., None]
            rgb = lum3 + (rgb - lum3) * (1.0 + vib_amount * weight)

        # write back to QImage buffer
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        arr[:, :w, :3] = rgb_u8

        # keep alpha as-is
        self._current_pixmap = QPixmap.fromImage(img)
        self._rescale_preview()
        self._update_histogram()


    def _schedule_render(self):
        """Debounce rendering so rapid slider moves don't re-render every tick."""
        # 40–60 ms feels responsive but avoids spamming the CPU
        self._render_timer.start(50)

    def _render_current_params(self):
        if self._preview_base_image is not None:
            self._apply_edit_params_to_current_image(self._current_params)


    def _rescale_preview(self):
        if self._current_pixmap is None:
            return

        if self._zoom_mode == "fit":
            # Fit to scrollarea viewport
            vp_size = self.image_scroll.viewport().size()
            if vp_size.width() <= 0 or vp_size.height() <= 0:
                return
            scaled = self._current_pixmap.scaled(
                vp_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
            # Manual zoom factor
            w = int(self._current_pixmap.width() * self._zoom_factor)
            h = int(self._current_pixmap.height() * self._zoom_factor)
            w = max(1, w)
            h = max(1, h)
            scaled = self._current_pixmap.scaled(
                w, h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

        self.image_display.setPixmap(scaled)
        self.image_display.resize(scaled.size())
        self.image_display.setText("")
        self._update_zoom_label()


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

    def load_folder(self, folder_path: str):
        self.thumbs.clear()
        self._icon_cache.clear()
        self._loading.clear()
        self._item_for_path.clear()
        self._current_folder = folder_path

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
        if self._current_pixmap is not None and self._zoom_mode == "fit":
            self._rescale_preview()
        self._ensure_visible_thumbs()


        # ---------- event filter: drag to pan image ----------

    def eventFilter(self, obj, event):
        if obj is self.image_display:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if self.image_display.pixmap() and not self.image_display.pixmap().isNull():
                    self._dragging = True
                    self._drag_last_pos = event.pos()
                    self.image_display.setCursor(Qt.ClosedHandCursor)
                    return True

            elif event.type() == QEvent.MouseMove and self._dragging:
                delta = event.pos() - self._drag_last_pos
                self._drag_last_pos = event.pos()
                hbar = self.image_scroll.horizontalScrollBar()
                vbar = self.image_scroll.verticalScrollBar()
                hbar.setValue(hbar.value() - delta.x())
                vbar.setValue(vbar.value() - delta.y())
                return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._dragging = False
                self.image_display.setCursor(Qt.ArrowCursor)
                return True

        return super().eventFilter(obj, event)



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
        self.history_tree.clear()
        self._image_parents.clear()

        for path, params in self._image_params.items():
            if not os.path.isfile(path):
                continue
            self._ensure_image_cached(path)
            cache = self._image_cache.get(path)
            if not cache:
                continue

            self._current_path = path
            self._preview_base_image = cache["preview"]

            if params:
                self._apply_edit_params_to_current_image(params)
            else:
                self._current_pixmap = QPixmap.fromImage(self._preview_base_image)

            self._update_history_for_current_image()

    def _ensure_project_root(self):
        if self._project_root:
            return

        all_paths = list(self._image_params.keys()) or list(self._edits.keys())
        if all_paths:
            abs_paths = [os.path.abspath(p) for p in all_paths]
            try:
                self._project_root = os.path.commonpath(abs_paths)
            except Exception:
                self._project_root = os.path.dirname(abs_paths[0])
        else:
            folder = self._get_active_folder()
            if folder:
                self._project_root = os.path.abspath(folder)

    def _encode_project_path(self, abs_path: str) -> str:
        abs_path = os.path.abspath(abs_path)
        root = self._project_root
        if root:
            try:
                common = os.path.commonpath([abs_path, root])
                if common == root:
                    return _relpath_or_same(abs_path, root)
            except Exception:
                pass
        return abs_path

    def _decode_project_path(self, stored: str, root: str) -> str:
        if os.path.isabs(stored):
            return os.path.abspath(stored)
        if root:
            return _abspath_from_base(stored, root)
        return os.path.abspath(stored)

    def export_lrc(self, autosave: bool = False):
        source_edits = self._image_params if self._image_params else self._edits

        if not source_edits and not self._current_path:
            if autosave:
                return
            QMessageBox.warning(
                self,
                "Nothing to Save",
                "There are no images with edits in this project yet."
            )
            return

        self._ensure_project_root()
        default_folder = self._project_root or self._get_active_folder() or os.path.expanduser("~")

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
            "current_image": current_key,
            "edits": edits_rel,
        }

        path = None
        if self._project_path is None:
            if autosave:
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Project",
                os.path.join(default_folder, "project.lrc"),
                "Lightroom Clone Project (*.lrc)",
            )
            if not path:
                return
            if not path.lower().endswith(".lrc"):
                path += ".lrc"
            self._project_path = path
        else:
            path = self._project_path

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            msg = "Autosaved project to" if autosave else "Saved project to"
            self.statusBar().showMessage(f"{msg} {path}", 2000)
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
        self.import_project_from_path(path)

    def import_project_from_path(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to import project: {e}")
            return

        if data.get("schema") != "ECE 277 LightRoom Project":
            QMessageBox.critical(self, "Import Failed", "Invalid project file.")
            return

        project_root = data.get("project_root") or data.get("folder_path")
        if not project_root or not os.path.isdir(project_root):
            QMessageBox.critical(self, "Import Failed", "Project root folder does not exist.")
            return

        self._project_root = os.path.abspath(project_root)
        self.load_folder(self._project_root)

        self._image_params.clear()
        self._edits.clear()

        imported_edits = data.get("edits", {})
        for stored_path, params in imported_edits.items():
            abs_path = self._decode_project_path(stored_path, self._project_root)
            self._image_params[abs_path] = dict(params)
            self._edits[abs_path] = dict(params)

        self._rebuild_history_from_params()

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

        self._project_path = os.path.abspath(path)
        self.statusBar().showMessage(f"Imported project from {path}", 2000)


# ----------------- main ----------------- #

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if arg_path.lower().endswith(".lrc") and os.path.isfile(arg_path):
            try:
                window.import_project_from_path(arg_path)
            except Exception as e:
                print("Failed to auto-load project:", e)

    sys.exit(app.exec())
