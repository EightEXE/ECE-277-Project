import os
import sys
import json
from datetime import datetime

# Prefer software OpenGL for compatibility unless overridden.
os.environ.setdefault("QT_OPENGL", os.environ.get("LRC_QT_OPENGL", "software"))

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
    QMenu, QTabWidget, QTreeView, QFileSystemModel, QScrollArea, QPushButton, QFrame,
    QComboBox, QStackedWidget, QToolButton, QDoubleSpinBox, QCheckBox, QButtonGroup,
    QSplitter, QColorDialog
)
from PySide6.QtCore import (
    Qt, QSize, QRunnable, QThreadPool, Signal, QObject, QPoint, QTimer, QDir, QPointF, QRectF
)
from PySide6.QtGui import (
    QPixmap, QIcon, QImageReader, QFontMetrics, QImage, QPainter,
    QColor, QPen, QPolygon, QSurfaceFormat, QConicalGradient, QRadialGradient,
    QLinearGradient
)
from PIL import Image, ExifTags

_ENV_ENABLE = os.environ.get("LRC_ENABLE_OPENGL")
_ENV_DISABLE = os.environ.get("LRC_DISABLE_OPENGL")
if _ENV_ENABLE is not None:
    _USE_OPENGL_PREVIEW = _ENV_ENABLE.lower() in {"1", "true", "yes"}
elif _ENV_DISABLE is not None:
    _USE_OPENGL_PREVIEW = not (_ENV_DISABLE.lower() in {"1", "true", "yes"})
else:
    _USE_OPENGL_PREVIEW = True

_QT_OPENGL_MODE = os.environ.get("QT_OPENGL", "").lower()
fmt = QSurfaceFormat()
if _QT_OPENGL_MODE == "desktop":
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
else:
    fmt.setRenderableType(QSurfaceFormat.OpenGLES)
    fmt.setVersion(3, 0)
QSurfaceFormat.setDefaultFormat(fmt)

try:
    if _USE_OPENGL_PREVIEW:
        from lightroom_clone.gl_viewer import GLImageView  # type: ignore
    else:
        GLImageView = None
except Exception as e:
    print("[OpenGL] Preview disabled:", e)
    GLImageView = None

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

def _load_qimage_any(
    path: str,
    *,
    max_long_edge: int | None = None,
    raw_fast: bool = False,
) -> QImage | None:
    """
    Load a QImage from path.
    Uses rawpy for RAW files when available, otherwise QImageReader.
    Can optionally limit the output size for faster previews.
    Returns None on failure.
    """
    ext = os.path.splitext(path)[1].lower()

    def _maybe_scale(img: QImage) -> QImage:
        if max_long_edge is None:
            return img
        w = img.width()
        h = img.height()
        if w <= 0 or h <= 0:
            return img
        max_side = max(w, h)
        if max_side <= max_long_edge:
            return img
        return img.scaled(
            max_long_edge,
            max_long_edge,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    # RAW path: use rawpy if present
    if ext in RAW_EXTENSIONS and rawpy is not None:
        try:
            with rawpy.imread(path) as raw:
                post_kwargs = {
                    "output_bps": 8,
                    "no_auto_bright": True,
                    "gamma": (2.2, 4.5),
                    "use_camera_wb": True,
                }
                if raw_fast and max_long_edge is not None:
                    longest = max(raw.sizes.raw_height, raw.sizes.raw_width)
                    if longest > max_long_edge * 1.5:
                        post_kwargs["half_size"] = True
                rgb = raw.postprocess(**post_kwargs)
            h, w, _ = rgb.shape
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            img = img.copy()
            img = _maybe_scale(img)
            return img.convertToFormat(QImage.Format_RGBA8888)
        except Exception as e:
            print(f"[RAW] Failed to decode {path}: {e}")
            return None

    # Fallback: normal image via Qt
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    if max_long_edge is not None:
        size = reader.size()
        if size.isValid():
            max_side = max(size.width(), size.height())
            if max_side > max_long_edge:
                scale = max_long_edge / max_side
                scaled = QSize(
                    max(1, int(size.width() * scale)),
                    max(1, int(size.height() * scale)),
                )
                reader.setScaledSize(scaled)
    img = reader.read()
    if img.isNull():
        return None
    return img.convertToFormat(QImage.Format_RGBA8888)


def _rgb_to_hsv_np(rgb):
    rgb_norm = np.clip(rgb / 255.0, 0.0, 1.0)
    r = rgb_norm[..., 0]
    g = rgb_norm[..., 1]
    b = rgb_norm[..., 2]
    maxc = np.max(rgb_norm, axis=-1)
    minc = np.min(rgb_norm, axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nonzero = maxc > 0
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    mask = delta > 1e-6
    r_mask = (maxc == r) & mask
    g_mask = (maxc == g) & mask
    b_mask = (maxc == b) & mask

    h[r_mask] = ((g - b)[r_mask] / delta[r_mask]) % 6
    h[g_mask] = ((b - r)[g_mask] / delta[g_mask]) + 2
    h[b_mask] = ((r - g)[b_mask] / delta[b_mask]) + 4
    h = (h / 6.0) % 1.0

    return h, s, v


def _hsv_to_rgb_np(h, s, v):
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

    idx = i % 6
    mask = idx == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = idx == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = idx == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = idx == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = idx == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = idx == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255)



# ----------------- thumbnail worker ----------------- #

class _WorkerSignals(QObject):
    ready = Signal(str, QImage)


class _ThumbTask(QRunnable):
    """Background thumbnail generator for the bottom filmstrip."""

    def __init__(self, path: str, size: QSize):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = _WorkerSignals()

    def run(self):
        max_dim = max(self.size.width(), self.size.height()) * 2
        img = _load_qimage_any(self.path, max_long_edge=max_dim, raw_fast=True)
        if img is not None:
            max_side = max(self.size.width(), self.size.height()) * 2
            if img.width() > max_side or img.height() > max_side:
                img = img.scaled(
                    max_side,
                    max_side,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
        else:
            img = QImage()
        self.signals.ready.emit(self.path, img)



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


class SoftwareImageView(QWidget):
    """Fallback preview widget when OpenGL is unavailable."""

    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel("No Image Loaded", self)
        self._label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label, 1)

        self._pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._fit_mode = True

    def has_image(self) -> bool:
        return not self._pixmap.isNull()

    def clear_image(self):
        self._pixmap = QPixmap()
        self._label.setPixmap(QPixmap())
        self._label.setText("No Image Loaded")
        self._zoom_factor = 1.0
        self._fit_mode = True
        self.zoomChanged.emit(1.0)

    def set_image(self, image: QImage):
        if image is None or image.isNull():
            self.clear_image()
            return
        self._pixmap = QPixmap.fromImage(image)
        self._fit_mode = True
        self._apply_zoom()

    def update_cpu_pixmap(self, pixmap: QPixmap | None):
        if pixmap is None or pixmap.isNull():
            return
        self._pixmap = QPixmap(pixmap)
        self._apply_zoom()

    def set_params(self, params: dict):
        # Software view renders whatever pixmap is provided; edits happen elsewhere.
        return

    def set_manual_zoom(self, factor: float) -> float:
        if not self.has_image():
            return self._zoom_factor
        factor = max(0.05, min(12.0, factor))
        self._fit_mode = False
        self._zoom_factor = factor
        self._apply_zoom()
        self.zoomChanged.emit(self._zoom_factor)
        return self._zoom_factor

    def fit_to_window(self) -> float:
        if not self.has_image():
            return self._zoom_factor
        self._fit_mode = True
        self._zoom_factor = self._compute_fit_zoom()
        self._apply_zoom()
        self.zoomChanged.emit(self._zoom_factor)
        return self._zoom_factor

    def current_zoom(self) -> float:
        return self._zoom_factor

    def capture_image(self, max_side: int | None = None) -> QImage:
        if not self.has_image():
            return QImage()
        img = self._pixmap.toImage()
        if max_side and max(img.width(), img.height()) > max_side:
            img = img.scaled(
                max_side,
                max_side,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        return img

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode and self.has_image():
            self._zoom_factor = self._compute_fit_zoom()
            self._apply_zoom()
            self.zoomChanged.emit(self._zoom_factor)

    def _compute_fit_zoom(self) -> float:
        if not self.has_image() or self.width() <= 0 or self.height() <= 0:
            return self._zoom_factor
        pw = max(1, self._pixmap.width())
        ph = max(1, self._pixmap.height())
        return min(self.width() / pw, self.height() / ph)

    def _apply_zoom(self):
        if not self.has_image():
            self._label.setPixmap(QPixmap())
            self._label.setText("No Image Loaded")
            return

        target_w = max(1, int(self._pixmap.width() * self._zoom_factor))
        target_h = max(1, int(self._pixmap.height() * self._zoom_factor))
        scaled = self._pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.setText("")


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
            img = _load_qimage_any(path, max_long_edge=256, raw_fast=True)
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


class CurvesWidget(QWidget):
    pointsChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self._points = [(0.0, 0.0), (1.0, 1.0)]
        self._drag_index: int | None = None

    def get_points(self) -> list[tuple[float, float]]:
        return list(self._points)

    def set_points(self, pts: list[tuple[float, float]]):
        if len(pts) >= 2:
            self._points = sorted(pts, key=lambda p: p[0])
            self._points[0] = (0.0, self._points[0][1])
            self._points[-1] = (1.0, self._points[-1][1])
            self._emit_changed()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.fillRect(rect, QColor(30, 31, 35))
        painter.setPen(QPen(QColor(70, 72, 78)))
        for i in range(1, 4):
            x = rect.left() + rect.width() * i / 4
            y = rect.top() + rect.height() * i / 4
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        painter.setPen(QPen(QColor(120, 180, 255), 2))
        prev = None
        for x, y in self._points:
            px = rect.left() + x * rect.width()
            py = rect.bottom() - y * rect.height()
            if prev is not None:
                painter.drawLine(prev[0], prev[1], px, py)
            prev = (px, py)
        painter.setBrush(QColor(255, 200, 0))
        painter.setPen(QPen(Qt.black))
        for x, y in self._points:
            px = rect.left() + x * rect.width()
            py = rect.bottom() - y * rect.height()
            painter.drawEllipse(QPointF(px, py), 5, 5)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        idx = self._point_at(event.position())
        if idx is None:
            self._add_point(event.position())
        else:
            self._drag_index = idx
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mouseDoubleClickEvent(event)
        idx = self._point_at(event.position())
        if idx is not None and idx not in (0, len(self._points) - 1):
            self._points.pop(idx)
            self._emit_changed()
            self.update()
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_index is None:
            return super().mouseMoveEvent(event)
        self._move_point(self._drag_index, event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_index = None
        super().mouseReleaseEvent(event)

    def _point_at(self, pos: QPointF) -> int | None:
        rect = self.rect().adjusted(10, 10, -10, -10)
        for idx, (x, y) in enumerate(self._points):
            px = rect.left() + x * rect.width()
            py = rect.bottom() - y * rect.height()
            if QRectF(px - 8, py - 8, 16, 16).contains(pos):
                return idx
        return None

    def _add_point(self, pos: QPointF):
        rect = self.rect().adjusted(10, 10, -10, -10)
        x = (pos.x() - rect.left()) / rect.width()
        y = (rect.bottom() - pos.y()) / rect.height()
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        self._points.append((x, y))
        self._points.sort(key=lambda p: p[0])
        self._drag_index = self._points.index((x, y))
        self._emit_changed()
        self.update()


class ColorWheelWidget(QWidget):
    hueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self._hue = 0.0

    def set_hue(self, hue: float):
        self._hue = hue % 360
        self.hueChanged.emit(self._hue)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect().adjusted(6, 6, -6, -6)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)

        radius = min(rect.width(), rect.height()) / 2
        center = rect.center()

        gradient = QConicalGradient(center, 0)
        for angle in range(0, 360, 10):
            gradient.setColorAt(angle / 360.0, QColor.fromHsv(angle, 255, 255))
        painter.setBrush(gradient)
        painter.drawEllipse(rect)

        radial = QRadialGradient(center, radius)
        radial.setColorAt(0.0, QColor(255, 255, 255))
        radial.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(radial)
        painter.drawEllipse(rect)

        painter.setPen(QPen(Qt.white, 2))
        angle_rad = np.radians(self._hue)
        dot_x = center.x() + np.cos(angle_rad) * radius * 0.85
        dot_y = center.y() - np.sin(angle_rad) * radius * 0.85
        painter.drawEllipse(QPointF(dot_x, dot_y), 6, 6)

    def mousePressEvent(self, event):
        self._update_hue_from_pos(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._update_hue_from_pos(event.position())

    def _update_hue_from_pos(self, pos: QPointF):
        center = self.rect().center()
        dx = pos.x() - center.x()
        dy = center.y() - pos.y()
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        self.set_hue(angle)
    def _move_point(self, idx: int, pos: QPointF):
        rect = self.rect().adjusted(10, 10, -10, -10)
        x = (pos.x() - rect.left()) / rect.width()
        y = (rect.bottom() - pos.y()) / rect.height()
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        if idx == 0:
            x = 0.0
        elif idx == len(self._points) - 1:
            x = 1.0
        else:
            left = self._points[idx - 1][0] + 0.01
            right = self._points[idx + 1][0] - 0.01
            x = min(right, max(left, x))
        self._points[idx] = (x, min(1.0, max(0.0, y)))
        self._points = sorted(self._points, key=lambda p: p[0])
        self._drag_index = self._points.index((x, min(1.0, max(0.0, y))))
        self._emit_changed()


class GradientStopEditor(QWidget):
    stopsChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self._stops = [
            (0.0, QColor("#ff4b4b")),
            (0.5, QColor("#57d1ff")),
            (1.0, QColor("#4bff88")),
        ]
        self._drag_index: int | None = None

    def set_stops(self, stops: list[tuple[float, QColor]]):
        if not stops:
            return
        clean = []
        for pos, color in stops:
            if isinstance(color, str):
                color = QColor(color)
            clean.append((float(max(0.0, min(1.0, pos))), QColor(color)))
        self._stops = sorted(clean, key=lambda s: s[0])
        self.update()

    def stops(self) -> list[tuple[float, QColor]]:
        return [(pos, QColor(color)) for pos, color in self._stops]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track_rect = self.rect().adjusted(12, 12, -12, -26)
        gradient = QLinearGradient(track_rect.topLeft(), track_rect.topRight())
        for pos, color in self._stops:
            gradient.setColorAt(pos, color)
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#3d3e42")))
        painter.drawRect(track_rect)

        for idx, (pos, color) in enumerate(self._stops):
            x = track_rect.left() + pos * track_rect.width()
            points = [
                QPointF(x, track_rect.bottom() + 4),
                QPointF(x - 8, track_rect.bottom() + 22),
                QPointF(x + 8, track_rect.bottom() + 22),
            ]
            polygon = QPolygon()
            for pt in points:
                polygon.append(pt.toPoint())
            painter.setBrush(color)
            painter.setPen(Qt.black)
            painter.drawPolygon(polygon)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            idx = self._hit_test(event.position())
            if idx is None:
                self._add_stop(event.position())
            else:
                self._drag_index = idx
        elif event.button() == Qt.RightButton:
            idx = self._hit_test(event.position())
            if idx not in (None, 0, len(self._stops) - 1):
                self._stops.pop(idx)
                self.stopsChanged.emit(self.stops())
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_index is None:
            return super().mouseMoveEvent(event)
        self._move_stop(self._drag_index, event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_index = None
        super().mouseReleaseEvent(event)

    def _hit_test(self, pos: QPointF) -> int | None:
        rect = self.rect().adjusted(12, 12, -12, -26)
        for idx, (stop_pos, color) in enumerate(self._stops):
            x = rect.left() + stop_pos * rect.width()
            triangle = QRectF(x - 10, rect.bottom(), 20, 24)
            if triangle.contains(pos):
                return idx
        return None

    def _add_stop(self, pos: QPointF):
        rect = self.rect().adjusted(12, 12, -12, -26)
        t = float(max(0.0, min(1.0, (pos.x() - rect.left()) / rect.width())))
        color = QColor.fromHsvF(t, 1.0, 1.0)
        self._stops.append((t, color))
        self._stops.sort(key=lambda s: s[0])
        self.stopsChanged.emit(self.stops())
        self.update()

    def _move_stop(self, idx: int, pos: QPointF):
        rect = self.rect().adjusted(12, 12, -12, -26)
        t = float(max(0.0, min(1.0, (pos.x() - rect.left()) / rect.width())))
        if idx == 0:
            t = 0.0
        elif idx == len(self._stops) - 1:
            t = 1.0
        else:
            left = self._stops[idx - 1][0] + 0.01
            right = self._stops[idx + 1][0] - 0.01
            t = min(right, max(left, t))
        color = self._stops[idx][1]
        self._stops[idx] = (t, color)
        self._stops.sort(key=lambda s: s[0])
        self.stopsChanged.emit(self.stops())
        self.update()
class AdjustmentPanel(QWidget):
    GROUPS = [
        {
            "name": "Light",
            "icon": "☀",
            "tabs": [
                "Levels",
                "White Balance",
                "Brightness / Contrast",
                "Exposure",
                "Shadows / Highlights",
                "Vibrance",
                "Posterize",
            ],
        },
        {
            "name": "Color",
            "icon": "⚪⚪⚪",
            "tabs": [
                "HSL",
                "Recolor",
                "Black & White",
                "Selective Color",
                "Color Balance",
                "White Balance",
            ],
        },
        {
            "name": "Tone",
            "icon": "○",
            "tabs": [
                "Curves",
                "Channel Mixer",
                "Gradient Map",
                "Split Toning",
            ],
        },
        {
            "name": "Geometry",
            "icon": "▢",
            "tabs": [
                "Normals",
            ],
        },
        {
            "name": "FX",
            "icon": "fx",
            "tabs": [
                "Lens Filter",
            ],
        },
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._group_stack = QStackedWidget(self)
        self._group_sections: dict[tuple[str, str], CollapsibleSection] = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Group bar (horizontal icons)
        group_layout = QHBoxLayout()
        group_layout.setSpacing(6)
        group_layout.addStretch(1)
        self._group_button_group = QButtonGroup(self)
        self._group_button_group.buttonClicked.connect(self._on_group_button_clicked)

        for idx, group in enumerate(self.GROUPS):
            btn = QToolButton(self)
            btn.setText(group["icon"])
            btn.setToolTip(group["name"])
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setMinimumSize(36, 36)
            btn.setIconSize(QSize(24, 24))
            self._group_button_group.addButton(btn)
            self._group_button_group.setId(btn, idx)
            group_layout.addWidget(btn)
        group_layout.addStretch(1)
        main_layout.addLayout(group_layout)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(6)
        self.btn_add_preset = QPushButton("Add Preset")
        self.btn_merge = QPushButton("Merge")
        self.btn_delete = QPushButton("Delete")
        self.btn_reset = QPushButton("Reset")
        header.addWidget(self.btn_add_preset)
        header.addWidget(self.btn_merge)
        header.addWidget(self.btn_delete)
        header.addWidget(self.btn_reset)
        header.addStretch(1)
        main_layout.addLayout(header)

        # Group pages
        for group in self.GROUPS:
            group_widget = QWidget()
            vbox = QVBoxLayout(group_widget)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(6)
            for tab_name in group["tabs"]:
                section = CollapsibleSection(tab_name, self, start_collapsed=False)
                placeholder = self._create_placeholder(tab_name)
                section.content_layout().addWidget(placeholder)
                self._group_sections[(group["name"], tab_name)] = section
                vbox.addWidget(section)
            vbox.addStretch(1)
            self._group_stack.addWidget(group_widget)

        main_layout.addWidget(self._group_stack, 1)

        # Footer
        footer = QHBoxLayout()
        footer.setSpacing(6)
        footer.addWidget(QLabel("Opacity"))
        self.opacity_combo = QComboBox()
        for i in range(0, 101, 10):
            self.opacity_combo.addItem(f"{i}%", i)
        self.opacity_combo.setCurrentIndex(10)  # 100%
        footer.addWidget(self.opacity_combo)

        footer.addWidget(QLabel("Blend Mode"))
        self.blend_mode_combo = QComboBox()
        for mode in ["Normal", "Screen", "Multiply", "Overlay", "Soft Light", "Hard Light"]:
            self.blend_mode_combo.addItem(mode)
        footer.addStretch(1)
        self.gear_button = QToolButton()
        self.gear_button.setText("⚙")
        footer.addWidget(self.gear_button)
        main_layout.addLayout(footer)

        # Default selection
        first_btn = self._group_button_group.button(0)
        if first_btn:
            first_btn.setChecked(True)
            self._group_stack.setCurrentIndex(0)

    def _on_group_button_clicked(self, button):
        index = self._group_button_group.id(button)
        if index >= 0:
            self._group_stack.setCurrentIndex(index)

    def _create_placeholder(self, tab_name: str) -> QWidget:
        placeholder = QWidget()
        pv = QVBoxLayout(placeholder)
        pv.setContentsMargins(8, 8, 8, 8)
        desc = QLabel(f"{tab_name} controls coming soon.")
        desc.setWordWrap(True)
        pv.addWidget(desc)
        pv.addStretch(1)
        return placeholder

    def set_tab_content(self, group_name: str, tab_name: str, widget: QWidget):
        section = self._group_sections.get((group_name, tab_name))
        if section is None:
            return
        layout = section.content_layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        layout.addWidget(widget)
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

        self._image_cache: dict[str, dict] = {}   # path -> {"full": QImage|None, "preview": QImage, "mtime": float}
        self._image_params: dict[str, dict] = {}  # path -> params
        self._image_parents: dict[str, QTreeWidgetItem] = {}
        self._edits: dict[str, dict] = {}         # for .lrc export
        self._metadata_labels: dict[str, QLabel] = {}

        self._project_path: str | None = None
        self._project_root: str | None = None
        self._preview_long_edge = 2048

        self._fs_using_fullres = False
        self._fs_prev_preview_state: tuple | None = None

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
            "brightness": 128,
            "posterize_levels": 4,
            "hsl_hue": 0,
            "hsl_saturation": 0,
            "hsl_luminance": 0,
            "recolor_hue": 0,
            "recolor_saturation": 0,
            "recolor_lightness": 0,
            "bw_red": 0,
            "bw_yellow": 0,
            "bw_green": 0,
            "bw_cyan": 0,
            "bw_blue": 0,
            "bw_magenta": 0,
            "selective_cyan": 0,
            "selective_magenta": 0,
            "selective_yellow": 0,
            "selective_black": 0,
            "color_balance_cyan_red": 0,
            "color_balance_magenta_green": 0,
            "color_balance_yellow_blue": 0,
            "geometry_rotation": 0,
            "geometry_scale": 100,
            "fx_noise": 0,
            "fx_density": 50,
            "fx_color": "#ff8a3b",
        }
        self._gradient_map_stops: list[tuple[float, str]] = [
            (0.0, "#ff4b4b"),
            (0.5, "#57d1ff"),
            (1.0, "#4bff88"),
        ]

        # autosave
        self._autosave_interval_min = 5
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._update_autosave_timer()

        # preview / histogram timers
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_current_params)
                # --- zoom / pan state ---
        self._zoom_mode = "fit"   # "fit" or "manual"
        self._zoom_factor = 1.0   # 1.0 = 100%

        # ---------- Central image area ----------
        self._using_opengl = bool(GLImageView and GLImageView.is_supported())
        if self._using_opengl:
            try:
                self.image_view = GLImageView(self)
                print("[Preview] Using OpenGL preview")
            except Exception as e:
                print("[OpenGL] Failed to initialize preview, using software fallback:", e)
                self.image_view = SoftwareImageView(self)
                self._using_opengl = False
        else:
            self.image_view = SoftwareImageView(self)
            self._using_opengl = False
            print("[Preview] Using software preview (OpenGL disabled)")

        self.image_view.setObjectName("imageDisplay")

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
        self.vbox.addWidget(self.image_view, 1)

        # zoom button signals
        self.zoom_fit_btn.clicked.connect(self._on_zoom_fit)
        self.zoom_100_btn.clicked.connect(self._on_zoom_100)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)

        self.image_view.zoomChanged.connect(self._on_view_zoom_changed)

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

        self._init_icon_loader()

        self.adjust_tabs = QTabWidget()
        self.adjust_tabs.setObjectName("rightTabs")
        self.adjust_tabs.setTabPosition(QTabWidget.North)
        self.adjust_tabs.setIconSize(QSize(32, 32))
        self.adjust_tabs.setDocumentMode(True)

        self._build_color_tab()
        self._build_detail_tab()
        self._build_geometry_tab()
        self._build_fx_tab()
        self._build_metadata_tab()

        self.adjust_panel = AdjustmentPanel(self)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.hist_widget)
        splitter.addWidget(self.adjust_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.right_dock.setWidget(splitter)
        self._init_adjustment_panel_content()

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
        if not self.image_view.has_image():
            return

        if mode == "fit":
            self._zoom_mode = "fit"
            self._zoom_factor = self.image_view.fit_to_window()
        else:
            factor = max(0.1, min(8.0, factor))
            self._zoom_mode = "manual"
            self._zoom_factor = self.image_view.set_manual_zoom(factor)
        self._update_zoom_label()

    def _update_zoom_label(self):
        if not self.image_view.has_image():
            self.zoom_label.setText("—")
            return

        pct = int(round(self._zoom_factor * 100))
        self.zoom_label.setText(f"{pct}%")

    def _on_view_zoom_changed(self, factor: float):
        self._zoom_factor = factor
        self._update_zoom_label()

    def _on_zoom_in(self):
        if not self.image_view.has_image():
            return
        self._set_zoom(self._zoom_factor * 1.25, mode="manual")

    def _on_zoom_out(self):
        if not self.image_view.has_image():
            return
        self._set_zoom(self._zoom_factor / 1.25, mode="manual")

    def _on_zoom_fit(self):
        if not self.image_view.has_image():
            return
        self._set_zoom(self._zoom_factor, mode="fit")

    def _on_zoom_100(self):
        if not self.image_view.has_image():
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

    def _init_icon_loader(self):
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")

        def load_icon(name: str, fallback_role=None):
            path = os.path.join(icons_dir, name)
            if os.path.exists(path):
                return QIcon(path)
            if fallback_role is not None:
                return self.style().standardIcon(fallback_role)
            return QIcon()

        self._load_icon = load_icon

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

    def _create_labeled_slider(self, title: str, min_value: int, max_value: int, default: int, suffix: str = "", on_change=None):
        row = QHBoxLayout()
        row.setSpacing(6)
        label_title = QLabel(title)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_value, max_value)
        slider.setValue(default)
        value_label = QLabel(f"{default}{suffix}")

        def _update_text(val):
            value_label.setText(f"{val}{suffix}")
            if on_change:
                on_change(val)

        slider.valueChanged.connect(_update_text)
        slider.sliderReleased.connect(self._on_edit_committed)

        row.addWidget(label_title)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return row, slider

    def _init_adjustment_panel_content(self):
        self.adjust_panel.set_tab_content("Light", "Levels", self._create_levels_tab())
        self.adjust_panel.set_tab_content(
            "Light", "White Balance", self._create_light_white_balance_tab()
        )
        self.adjust_panel.set_tab_content(
            "Light", "Brightness / Contrast", self._create_light_brightness_tab()
        )
        self.adjust_panel.set_tab_content(
            "Light", "Exposure", self._create_light_exposure_tab()
        )
        self.adjust_panel.set_tab_content(
            "Light", "Shadows / Highlights", self._create_light_shadows_tab()
        )
        self.adjust_panel.set_tab_content(
            "Light", "Vibrance", self._create_light_vibrance_tab()
        )
        self.adjust_panel.set_tab_content("Light", "Posterize", self._create_light_posterize_tab())
        self.adjust_panel.set_tab_content("Color", "HSL", self._create_color_hsl_tab())
        self.adjust_panel.set_tab_content("Color", "Recolor", self._create_color_recolor_tab())
        self.adjust_panel.set_tab_content("Color", "Black & White", self._create_color_bw_tab())
        self.adjust_panel.set_tab_content("Color", "Selective Color", self._create_color_selective_tab())
        self.adjust_panel.set_tab_content("Color", "Color Balance", self._create_color_balance_tab())
        self.adjust_panel.set_tab_content("Tone", "Curves", self._create_tone_curves_tab())
        self.adjust_panel.set_tab_content("Tone", "Channel Mixer", self._create_tone_channel_mixer_tab())
        self.adjust_panel.set_tab_content("Tone", "Gradient Map", self._create_tone_gradient_map_tab())
        self.adjust_panel.set_tab_content("Tone", "Split Toning", self._create_tone_split_tone_tab())
        self.adjust_panel.set_tab_content("Geometry", "Normals", self._create_geometry_normals_tab())
        self.adjust_panel.set_tab_content("FX", "Lens Filter", self._create_fx_lens_filter_tab())

    def _create_levels_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        channel_row = QHBoxLayout()
        channel_row.setSpacing(8)
        lbl_model = QLabel("Color Model")
        self.levels_model_combo = QComboBox()
        self.levels_model_combo.addItems(["RGB", "Red", "Green", "Blue", "Alpha"])
        lbl_channel = QLabel("Channel")
        self.levels_channel_combo = QComboBox()
        self.levels_channel_combo.addItems(["Master", "Red", "Green", "Blue", "Alpha"])
        channel_row.addWidget(lbl_model)
        channel_row.addWidget(self.levels_model_combo)
        channel_row.addSpacing(12)
        channel_row.addWidget(lbl_channel)
        channel_row.addWidget(self.levels_channel_combo)
        channel_row.addStretch(1)
        layout.addLayout(channel_row)

        layout.addLayout(self._create_percentage_slider_row("Black Level", 0))
        layout.addLayout(self._create_percentage_slider_row("White Level", 100))
        layout.addLayout(self._create_gamma_slider_row("Gamma", 1.0))
        layout.addLayout(self._create_percentage_slider_row("Output Black Level", 0))
        layout.addLayout(self._create_percentage_slider_row("Output White Level", 100))

        layout.addStretch(1)
        return widget

    def _create_light_white_balance_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        temp_row = QHBoxLayout()
        temp_row.setSpacing(8)
        lbl_temp = QLabel("White Balance")
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 255)
        self.temperature_slider.setValue(self._current_params.get("temperature", 128))
        self.temperature_value = QLabel(f"Temp: {self.temperature_slider.value()}")
        self.temperature_slider.valueChanged.connect(self._on_temperature_changed)
        self.temperature_slider.sliderReleased.connect(self._on_edit_committed)
        temp_row.addWidget(lbl_temp)
        temp_row.addWidget(self.temperature_slider, 1)
        temp_row.addWidget(self.temperature_value)
        layout.addLayout(temp_row)

        tint_row = QHBoxLayout()
        tint_row.setSpacing(8)
        lbl_tint = QLabel("Tint")
        self.tint_slider = QSlider(Qt.Horizontal)
        self.tint_slider.setRange(0, 255)
        self.tint_slider.setValue(self._current_params.get("tint", 128))
        self.tint_value = QLabel(f"Tint: {self.tint_slider.value()}")
        self.tint_slider.valueChanged.connect(self._on_tint_changed)
        self.tint_slider.sliderReleased.connect(self._on_edit_committed)
        tint_row.addWidget(lbl_tint)
        tint_row.addWidget(self.tint_slider, 1)
        tint_row.addWidget(self.tint_value)
        layout.addLayout(tint_row)

        picker_btn = QPushButton("Picker")
        picker_btn.clicked.connect(self._on_white_balance_picker)
        picker_btn.setFixedWidth(120)
        layout.addWidget(picker_btn, alignment=Qt.AlignLeft)
        layout.addStretch(1)
        return widget

    def _create_light_brightness_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.brightness_slider, self.brightness_value, bright_row = self._make_slider_row("Brightness")
        self.brightness_slider.setValue(self._current_params.get("brightness", 128))
        self.brightness_value.setText(f"Brightness: {self.brightness_slider.value()}")
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.brightness_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(bright_row)

        self.contrast_slider, self.contrast_value, con_row = self._make_slider_row("Contrast")
        self.contrast_slider.setValue(self._current_params.get("contrast", 128))
        self.contrast_value.setText(f"Contrast: {self.contrast_slider.value()}")
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.contrast_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(con_row)

        self.brightness_linear_checkbox = QCheckBox("Linear")
        layout.addWidget(self.brightness_linear_checkbox)
        layout.addStretch(1)
        return widget

    def _create_light_exposure_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.exposure_slider, self.exposure_value, exp_row = self._make_slider_row("Exposure")
        self.exposure_slider.setValue(self._current_params.get("exposure", 128))
        self.exposure_value.setText(f"Exposure: {self.exposure_slider.value()}")
        self.exposure_slider.valueChanged.connect(self._on_exposure_changed)
        self.exposure_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(exp_row)
        layout.addStretch(1)
        return widget

    def _create_light_shadows_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.shadows_slider, self.shadows_value, sh_row = self._make_slider_row("Shadows")
        self.shadows_slider.setValue(self._current_params.get("shadows", 128))
        self.shadows_value.setText(f"Shadows: {self.shadows_slider.value()}")
        self.shadows_slider.valueChanged.connect(self._on_shadows_changed)
        self.shadows_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(sh_row)

        self.highlights_slider, self.highlights_value, hi_row = self._make_slider_row("Highlights")
        self.highlights_slider.setValue(self._current_params.get("highlights", 128))
        self.highlights_value.setText(f"Highlights: {self.highlights_slider.value()}")
        self.highlights_slider.valueChanged.connect(self._on_highlights_changed)
        self.highlights_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(hi_row)

        self.whites_slider, self.whites_value, wh_row = self._make_slider_row("Whites")
        self.whites_slider.setValue(self._current_params.get("whites", 128))
        self.whites_value.setText(f"Whites: {self.whites_slider.value()}")
        self.whites_slider.valueChanged.connect(self._on_whites_changed)
        self.whites_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(wh_row)

        self.blacks_slider, self.blacks_value, bl_row = self._make_slider_row("Blacks")
        self.blacks_slider.setValue(self._current_params.get("blacks", 128))
        self.blacks_value.setText(f"Blacks: {self.blacks_slider.value()}")
        self.blacks_slider.valueChanged.connect(self._on_blacks_changed)
        self.blacks_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(bl_row)

        layout.addStretch(1)
        return widget

    def _create_light_vibrance_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.vibrance_slider, self.vibrance_value, vib_row = self._make_slider_row("Vibrance")
        self.vibrance_slider.setValue(self._current_params.get("vibrance", 128))
        self.vibrance_value.setText(f"Vibrance: {self.vibrance_slider.value()}")
        self.vibrance_slider.valueChanged.connect(self._on_vibrance_changed)
        self.vibrance_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(vib_row)

        self.saturation_slider, self.saturation_value, sat_row = self._make_slider_row("Saturation")
        self.saturation_slider.setValue(self._current_params.get("saturation", 128))
        self.saturation_value.setText(f"Saturation: {self.saturation_slider.value()}")
        self.saturation_slider.valueChanged.connect(self._on_saturation_changed)
        self.saturation_slider.sliderReleased.connect(self._on_edit_committed)
        layout.addLayout(sat_row)

        layout.addStretch(1)
        return widget

    def _create_light_posterize_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel("Posterize Levels")
        self.posterize_slider = QSlider(Qt.Horizontal)
        self.posterize_slider.setRange(2, 32)
        self.posterize_slider.setValue(self._current_params.get("posterize_levels", 4))
        self.posterize_value = QLabel(f"Levels: {self.posterize_slider.value()}")
        self.posterize_slider.valueChanged.connect(self._on_posterize_levels_changed)
        self.posterize_slider.sliderReleased.connect(self._on_edit_committed)
        row.addWidget(lbl)
        row.addWidget(self.posterize_slider, 1)
        row.addWidget(self.posterize_value)
        layout.addLayout(row)
        layout.addStretch(1)
        return widget

    def _create_color_hsl_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.hsl_hsv_checkbox = QCheckBox("HSV Mode")
        layout.addWidget(self.hsl_hsv_checkbox)

        self.hsl_color_wheel = ColorWheelWidget(self)
        self.hsl_color_wheel.hueChanged.connect(self._on_color_wheel_hue_changed)
        self.hsl_color_wheel.set_hue(self._current_params.get("hsl_hue", 0))
        layout.addWidget(self.hsl_color_wheel)

        swatch_row = QHBoxLayout()
        swatch_colors = [
            ("R", "#ff4b4b"),
            ("O", "#ff8c3b"),
            ("Y", "#ffc93b"),
            ("G", "#58d16a"),
            ("C", "#42c5d9"),
            ("B", "#3a7bff"),
            ("M", "#d84bff"),
        ]
        for text, color in swatch_colors:
            btn = QToolButton()
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(f"background-color: {color}; border-radius: 14px;")
            btn.setToolTip(text)
            swatch_row.addWidget(btn)
        swatch_row.addStretch(1)
        picker_btn = QPushButton("Picker")
        picker_btn.setFixedWidth(80)
        swatch_row.addWidget(picker_btn)
        layout.addLayout(swatch_row)

        row, slider = self._create_labeled_slider(
            "Hue Shift", -180, 180, self._current_params.get("hsl_hue", 0), "°", self._on_hsl_hue_changed
        )
        self.hsl_hue_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Saturation Shift", -100, 100, self._current_params.get("hsl_saturation", 0), "%", self._on_hsl_saturation_changed
        )
        self.hsl_saturation_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Luminosity Shift", -100, 100, self._current_params.get("hsl_luminance", 0), "%", self._on_hsl_luminance_changed
        )
        self.hsl_luminance_slider = slider
        layout.addLayout(row)

        layout.addStretch(1)
        return widget

    def _create_color_recolor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        row, slider = self._create_labeled_slider(
            "Hue", -180, 180, self._current_params.get("recolor_hue", 0), "°", self._on_recolor_hue_changed
        )
        self.recolor_hue_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Saturation", -100, 100, self._current_params.get("recolor_saturation", 0), "%", self._on_recolor_saturation_changed
        )
        self.recolor_saturation_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Lightness", -100, 100, self._current_params.get("recolor_lightness", 0), "%", self._on_recolor_lightness_changed
        )
        self.recolor_lightness_slider = slider
        layout.addLayout(row)

        layout.addStretch(1)
        return widget

    def _create_color_bw_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        colors = [
            ("Red", "bw_red"),
            ("Yellow", "bw_yellow"),
            ("Green", "bw_green"),
            ("Cyan", "bw_cyan"),
            ("Blue", "bw_blue"),
            ("Magenta", "bw_magenta"),
        ]
        for label_text, key in colors:
            row, slider = self._create_labeled_slider(
                label_text, -100, 100, self._current_params.get(key, 0), "%", lambda val, k=key: self._on_bw_slider_changed(k, val)
            )
            layout.addLayout(row)

        picker_btn = QPushButton("Picker")
        picker_btn.setFixedWidth(90)
        layout.addWidget(picker_btn, alignment=Qt.AlignLeft)
        layout.addStretch(1)
        return widget

    def _create_color_selective_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Color"))
        self.selective_color_combo = QComboBox()
        self.selective_color_combo.addItems(
            ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas", "Whites", "Neutrals", "Blacks"]
        )
        top_row.addWidget(self.selective_color_combo, 1)
        self.selective_relative_checkbox = QCheckBox("Relative")
        top_row.addWidget(self.selective_relative_checkbox)
        layout.addLayout(top_row)

        row, slider = self._create_labeled_slider(
            "Cyan", -100, 100, self._current_params.get("selective_cyan", 0), "%", self._on_selective_cyan_changed
        )
        self.selective_c_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Magenta", -100, 100, self._current_params.get("selective_magenta", 0), "%", self._on_selective_magenta_changed
        )
        self.selective_m_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Yellow", -100, 100, self._current_params.get("selective_yellow", 0), "%", self._on_selective_yellow_changed
        )
        self.selective_y_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Black", -100, 100, self._current_params.get("selective_black", 0), "%", self._on_selective_black_changed
        )
        self.selective_k_slider = slider
        layout.addLayout(row)
        layout.addStretch(1)
        return widget

    def _create_color_balance_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Tonal Range"))
        self.color_balance_range = QComboBox()
        self.color_balance_range.addItems(["Shadows", "Midtones", "Highlights"])
        top_row.addWidget(self.color_balance_range, 1)
        layout.addLayout(top_row)

        row, slider = self._create_labeled_slider(
            "Cyan / Red", -100, 100, self._current_params.get("color_balance_cyan_red", 0), "%", self._on_color_balance_cyan_changed
        )
        self.cb_cyan_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Magenta / Green", -100, 100, self._current_params.get("color_balance_magenta_green", 0), "%", self._on_color_balance_magenta_changed
        )
        self.cb_magenta_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Yellow / Blue", -100, 100, self._current_params.get("color_balance_yellow_blue", 0), "%", self._on_color_balance_yellow_changed
        )
        self.cb_yellow_slider = slider
        layout.addLayout(row)

        self.color_balance_preserve = QCheckBox("Preserve Luminosity")
        layout.addWidget(self.color_balance_preserve)
        layout.addStretch(1)
        return widget

    def _create_tone_curves_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.curves_widget = CurvesWidget(self)
        layout.addWidget(self.curves_widget)

        dropdown_row = QHBoxLayout()
        dropdown_row.addWidget(QLabel("Color Model"))
        self.curve_model_combo = QComboBox()
        self.curve_model_combo.addItems(["RGB", "CMYK", "LAB", "Grey"])
        dropdown_row.addWidget(self.curve_model_combo)
        dropdown_row.addSpacing(12)
        dropdown_row.addWidget(QLabel("Channel"))
        self.curve_channel_combo = QComboBox()
        self.curve_channel_combo.addItems(["Master", "Red", "Green", "Blue", "Alpha"])
        dropdown_row.addWidget(self.curve_channel_combo)
        dropdown_row.addStretch(1)
        picker_btn = QPushButton("Picker")
        picker_btn.setFixedWidth(80)
        dropdown_row.addWidget(picker_btn)
        layout.addLayout(dropdown_row)

        numeric_row = QHBoxLayout()
        for lbl in ["X", "Y", "Min", "Max"]:
            numeric_row.addWidget(QLabel(f"{lbl}:"))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setValue(0.5 if lbl in ("X", "Y") else (0.0 if lbl == "Min" else 1.0))
            numeric_row.addWidget(spin)
        layout.addLayout(numeric_row)
        layout.addStretch(1)
        return widget

    def _create_tone_channel_mixer_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output Model"))
        self.channel_mixer_model = QComboBox()
        self.channel_mixer_model.addItems(["RGB"])
        output_row.addWidget(self.channel_mixer_model)
        output_row.addSpacing(12)
        output_row.addWidget(QLabel("Channel"))
        self.channel_mixer_channel = QComboBox()
        self.channel_mixer_channel.addItems(["Red", "Green", "Blue", "Alpha"])
        output_row.addWidget(self.channel_mixer_channel)
        layout.addLayout(output_row)

        for label_text in ["Red", "Green", "Blue", "Alpha", "Offset"]:
            row, slider = self._create_labeled_slider(
                label_text, -200, 200, 0, "%", lambda val, lbl=label_text: self._on_channel_mixer_changed(lbl.lower(), val)
            )
            layout.addLayout(row)
        layout.addStretch(1)
        return widget

    def _create_tone_gradient_map_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.gradient_editor = GradientStopEditor(self)
        self.gradient_editor.set_stops(
            [(pos, QColor(color)) for pos, color in self._gradient_map_stops]
        )
        self.gradient_editor.stopsChanged.connect(self._on_gradient_editor_changed)
        layout.addWidget(self.gradient_editor)

        control_row = QHBoxLayout()
        for text in ["Insert", "Copy", "Reverse", "Delete"]:
            btn = QPushButton(text)
            control_row.addWidget(btn)
        layout.addLayout(control_row)

        layout.addStretch(1)
        return widget

    def _create_tone_split_tone_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        for name in ["Highlights Hue", "Highlights Saturation", "Shadows Hue", "Shadows Saturation", "Balance"]:
            row, slider = self._create_labeled_slider(
                name,
                0 if "Saturation" in name or "Balance" in name else 0,
                100 if "Saturation" in name or "Balance" in name else 360,
                50 if "Balance" in name else 0,
                "%" if "Saturation" in name or "Balance" in name else "°",
            )
        layout.addLayout(row)

        layout.addStretch(1)
        return widget

    def _create_geometry_normals_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        row, slider = self._create_labeled_slider(
            "Rotation", -180, 180, 0, "°", self._on_geometry_rotation_changed
        )
        self.geometry_rotation_slider = slider
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Scale", 0, 200, 100, "%", self._on_geometry_scale_changed
        )
        self.geometry_scale_slider = slider
        layout.addLayout(row)

        flips = QHBoxLayout()
        self.flip_x_checkbox = QCheckBox("Flip X")
        self.flip_y_checkbox = QCheckBox("Flip Y")
        flips.addWidget(self.flip_x_checkbox)
        flips.addWidget(self.flip_y_checkbox)
        flips.addStretch(1)
        layout.addLayout(flips)
        layout.addStretch(1)
        return widget

    def _create_fx_lens_filter_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.fx_filter_button = QPushButton("Filter Color")
        self.fx_filter_button.setFixedWidth(160)
        self.fx_filter_button.clicked.connect(self._on_fx_color_clicked)
        self._update_fx_color_button()
        layout.addWidget(self.fx_filter_button, alignment=Qt.AlignLeft)

        row, slider = self._create_labeled_slider(
            "Noise", 0, 100, 0, "%", self._on_fx_noise_changed
        )
        layout.addLayout(row)

        row, slider = self._create_labeled_slider(
            "Optical Density", 0, 100, 50, "%", self._on_fx_density_changed
        )
        layout.addLayout(row)

        self.fx_preserve_lum_checkbox = QCheckBox("Preserve Luminosity")
        layout.addWidget(self.fx_preserve_lum_checkbox)
        layout.addStretch(1)
        return widget

    def _create_percentage_slider_row(self, title: str, default: int) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel(title)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(default)
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setSuffix("%")
        spin.setValue(default)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        return row

    def _create_gamma_slider_row(self, title: str, default: float) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel(title)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(10, 400)
        slider.setValue(int(default * 100))
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 4.0)
        spin.setSingleStep(0.05)
        spin.setValue(default)
        slider.valueChanged.connect(lambda val: spin.setValue(val / 100.0))
        spin.valueChanged.connect(lambda val: slider.setValue(int(val * 100)))
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        return row

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

        # Placeholder when histogram dock is closed
        hist_section = CollapsibleSection("Histogram", self, start_collapsed=False)
        hist_section.content_layout().addWidget(QLabel("Use the Histogram dock to view data."))
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
            self._push_params_to_view()
            self._schedule_render()



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
        accent = "#646b6d"

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

        img = self._ensure_full_image(path)
        if img is not None:
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

    def _update_histogram_from_view(self, force: bool = False):
        if not self.image_view.has_image():
            self.hist_widget.clear_histogram()
            if self._preview_base_image is not None:
                self._current_pixmap = QPixmap.fromImage(self._preview_base_image)
                self.image_view.update_cpu_pixmap(self._current_pixmap)
            else:
                self._current_pixmap = None
            return

        is_valid = getattr(self.image_view, "isValid", lambda: True)
        context_method = getattr(self.image_view, "context", None)
        if context_method is None:
            view_ready = True
        else:
            view_ready = is_valid() and context_method() is not None
        if not view_ready:
            if force:
                QTimer.singleShot(50, lambda f=True: self._update_histogram_from_view(f))
            return

        max_side = None if force else 512
        img = self.image_view.capture_image(max_side=max_side)
        if img.isNull():
            if force and self._preview_base_image is not None:
                self._current_pixmap = QPixmap.fromImage(self._preview_base_image)
                self.image_view.update_cpu_pixmap(self._current_pixmap)
                self.hist_widget.set_image(self._preview_base_image)
            else:
                QTimer.singleShot(50, lambda f=force: self._update_histogram_from_view(f))
            return

        self._current_pixmap = QPixmap.fromImage(img)
        self.image_view.update_cpu_pixmap(self._current_pixmap)
        self.hist_widget.set_image(img)

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
        self.image_view.clear_image()

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

    def _use_full_resolution_preview(self):
        if self._fs_using_fullres:
            return
        path = self._current_path
        if not path:
            return
        full_img = self._ensure_full_image(path)
        if full_img is None or full_img.isNull():
            return

        self._fs_prev_preview_state = (
            self._preview_base_image,
            self._base_rgb,
            self._base_lum,
            self._L_norm,
            self._tone_masks,
        )

        self._preview_base_image = full_img
        self._base_rgb = None
        self._base_lum = None
        self._L_norm = None
        self._tone_masks = {}
        self._prepare_base_arrays()
        self._apply_edit_params_to_current_image(self._current_params)
        self._fs_using_fullres = True
        self.statusBar().showMessage("Showing full-resolution preview", 2000)

    def _restore_preview_resolution(self):
        if not self._fs_using_fullres or not self._fs_prev_preview_state:
            return
        (
            prev_image,
            prev_rgb,
            prev_lum,
            prev_L,
            prev_masks,
        ) = self._fs_prev_preview_state
        self._preview_base_image = prev_image
        self._base_rgb = prev_rgb
        self._base_lum = prev_lum
        self._L_norm = prev_L
        self._tone_masks = prev_masks or {}
        if self._preview_base_image is not None:
            if self._base_rgb is None or self._base_lum is None:
                self._prepare_base_arrays()
            self._apply_edit_params_to_current_image(self._current_params)
        self._fs_prev_preview_state = None
        self._fs_using_fullres = False

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
            self._use_full_resolution_preview()
            self.showFullScreen()
        else:
            self._is_fullscreen_mode = False
            self.act_fullscreen.blockSignals(True)
            self.act_fullscreen.setChecked(False)
            self.act_fullscreen.blockSignals(False)

            self._restore_preview_resolution()
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
            self._push_params_to_view()
            self._schedule_render()

    def _on_contrast_changed(self, value: int):
        self.contrast_value.setText(f"Contrast: {value}")
        self._current_params["contrast"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_exposure_changed(self, value: int):
        self.exposure_value.setText(f"Exposure: {value}")
        self._current_params["exposure"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_highlights_changed(self, value: int):
        self.highlights_value.setText(f"Highlights: {value}")
        self._current_params["highlights"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_shadows_changed(self, value: int):
        self.shadows_value.setText(f"Shadows: {value}")
        self._current_params["shadows"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_whites_changed(self, value: int):
        self.whites_value.setText(f"Whites: {value}")
        self._current_params["whites"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_blacks_changed(self, value: int):
        self.blacks_value.setText(f"Blacks: {value}")
        self._current_params["blacks"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_temperature_changed(self, value: int):
        self.temperature_value.setText(f"Temp: {value}")
        self._current_params["temperature"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_tint_changed(self, value: int):
        self.tint_value.setText(f"Tint: {value}")
        self._current_params["tint"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_vibrance_changed(self, value: int):
        self.vibrance_value.setText(f"Vibrance: {value}")
        self._current_params["vibrance"] = value
        if self._preview_base_image is not None:
            self._push_params_to_view()
            self._schedule_render()

    def _on_brightness_changed(self, value: int):
        if hasattr(self, "brightness_value"):
            self.brightness_value.setText(f"Brightness: {value}")
        self._current_params["brightness"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_posterize_levels_changed(self, value: int):
        if hasattr(self, "posterize_value"):
            self.posterize_value.setText(f"Levels: {value}")
        self._current_params["posterize_levels"] = max(2, value)
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_white_balance_picker(self):
        QMessageBox.information(
            self,
            "White Balance Picker",
            "Picker functionality will be implemented soon.",
        )

    def _on_hsl_hue_changed(self, value: int):
        self._current_params["hsl_hue"] = value
        if hasattr(self, "hsl_color_wheel"):
            self.hsl_color_wheel.blockSignals(True)
            self.hsl_color_wheel.set_hue(value)
            self.hsl_color_wheel.blockSignals(False)
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_hsl_saturation_changed(self, value: int):
        self._current_params["hsl_saturation"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_hsl_luminance_changed(self, value: int):
        self._current_params["hsl_luminance"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_recolor_hue_changed(self, value: int):
        self._current_params["recolor_hue"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_recolor_saturation_changed(self, value: int):
        self._current_params["recolor_saturation"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_recolor_lightness_changed(self, value: int):
        self._current_params["recolor_lightness"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_bw_slider_changed(self, key: str, value: int):
        self._current_params[key] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_selective_cyan_changed(self, value: int):
        self._current_params["selective_cyan"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_selective_magenta_changed(self, value: int):
        self._current_params["selective_magenta"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_selective_yellow_changed(self, value: int):
        self._current_params["selective_yellow"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_selective_black_changed(self, value: int):
        self._current_params["selective_black"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_color_balance_cyan_changed(self, value: int):
        self._current_params["color_balance_cyan_red"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_color_balance_magenta_changed(self, value: int):
        self._current_params["color_balance_magenta_green"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_color_balance_yellow_changed(self, value: int):
        self._current_params["color_balance_yellow_blue"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_color_wheel_hue_changed(self, hue: float):
        if hasattr(self, "hsl_hue_slider"):
            self.hsl_hue_slider.blockSignals(True)
            self.hsl_hue_slider.setValue(int(round(hue)))
            self.hsl_hue_slider.blockSignals(False)
            self._on_hsl_hue_changed(int(round(hue)))

    def _on_geometry_rotation_changed(self, value: int):
        self._current_params["geometry_rotation"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_geometry_scale_changed(self, value: int):
        self._current_params["geometry_scale"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_fx_noise_changed(self, value: int):
        self._current_params["fx_noise"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_fx_density_changed(self, value: int):
        self._current_params["fx_density"] = value
        if self._preview_base_image is not None:
            self._schedule_render()

    def _update_fx_color_button(self):
        color = QColor(self._current_params.get("fx_color", "#ff8a3b"))
        self.fx_filter_button.setStyleSheet(
            f"background-color: {color.name()}; color: #000; border: 1px solid #555;"
        )

    def _on_fx_color_clicked(self):
        initial = QColor(self._current_params.get("fx_color", "#ff8a3b"))
        color = QColorDialog.getColor(initial, self, "Choose Filter Color")
        if color and color.isValid():
            self._current_params["fx_color"] = color.name()
            self._update_fx_color_button()
            if self._preview_base_image is not None:
                self._schedule_render()

    def _on_gradient_editor_changed(self, stops):
        self._gradient_map_stops = [(pos, color.name()) for pos, color in stops]
        if self._preview_base_image is not None:
            self._schedule_render()

    def _on_edit_committed(self):
        if not self._current_path or self._preview_base_image is None:
            return

        self._image_params[self._current_path] = dict(self._current_params)
        self._edits[self._current_path] = dict(self._current_params)

        self._update_history_for_current_image()
        self.export_lrc(autosave=True)
        self._update_histogram_from_view(force=True)

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
        try:
            stat_mtime = os.path.getmtime(path)
        except OSError:
            stat_mtime = None

        cache = self._image_cache.get(path)
        if cache and (stat_mtime is None or cache.get("mtime") == stat_mtime):
            return

        preview = _load_qimage_any(
            path,
            max_long_edge=self._preview_long_edge,
            raw_fast=True,
        )
        if preview is None:
            return

        full_img = cache.get("full") if cache else None
        self._image_cache[path] = {
            "full": full_img,
            "preview": preview,
            "mtime": stat_mtime,
        }

    def _ensure_full_image(self, path: str) -> QImage | None:
        self._ensure_image_cached(path)
        cache = self._image_cache.get(path)
        if not cache:
            return None
        img = cache.get("full")
        if img is None or img.isNull():
            img = _load_qimage_any(path)
            if img is not None:
                cache["full"] = img
        return cache.get("full")



    def _show_image_version(self, path: str, params: dict, update_sliders: bool):
        self._ensure_image_cached(path)
        cache = self._image_cache.get(path)
        if not cache:
            return

        self._current_path = path
        self._preview_base_image = cache["preview"]
        self.image_view.set_image(self._preview_base_image)
        self._zoom_mode = "fit"
        self._rescale_preview()
        self._current_pixmap = QPixmap.fromImage(self._preview_base_image)
        self.image_view.update_cpu_pixmap(self._current_pixmap)
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
            self._push_params_to_view()
                    # --- rendering caches / performance ---
            self._base_rgb = None          # float32 RGB of current base image
            self._base_lum = None          # float32 luminance
            self._L_norm = None            # L = lum / 255.0
            self._tone_masks = {}          # precomputed masks for hi/sh/wh/bl

            self._update_metadata_for_current()

        self._update_histogram_from_view(force=True)

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

        bright_val = params.get("brightness", 128)
        if bright_val != 128:
            offset = (bright_val - 128.0) / 128.0 * 64.0
            rgb += offset

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

        hsv_cache = {"h": None, "s": None, "v": None}

        def ensure_hsv():
            if hsv_cache["h"] is None:
                h, s, v = _rgb_to_hsv_np(rgb)
                hsv_cache["h"], hsv_cache["s"], hsv_cache["v"] = h, s, v
            return hsv_cache["h"], hsv_cache["s"], hsv_cache["v"]

        hue_shift = params.get("hsl_hue", 0) + params.get("recolor_hue", 0)
        sat_shift = params.get("hsl_saturation", 0) + params.get("recolor_saturation", 0)
        lum_shift = params.get("hsl_luminance", 0) + params.get("recolor_lightness", 0)
        if any(abs(x) > 1e-3 for x in (hue_shift, sat_shift, lum_shift)):
            h, s, v = ensure_hsv()
            if hue_shift:
                h = (h + hue_shift / 360.0) % 1.0
            if sat_shift:
                s = np.clip(s * (1.0 + sat_shift / 100.0), 0.0, 1.0)
            if lum_shift:
                v = np.clip(v + lum_shift / 100.0, 0.0, 1.0)
            rgb = _hsv_to_rgb_np(h, s, v)

        bw_keys = ["bw_red", "bw_yellow", "bw_green", "bw_cyan", "bw_blue", "bw_magenta"]
        if any(params.get(k, 0) != 0 for k in bw_keys):
            h, _, _ = ensure_hsv()
            hue_deg = (h * 360.0) % 360.0
            weight = np.zeros_like(hue_deg)
            color_ranges = [
                (0, "bw_red"),
                (60, "bw_yellow"),
                (120, "bw_green"),
                (180, "bw_cyan"),
                (240, "bw_blue"),
                (300, "bw_magenta"),
            ]
            for center, key in color_ranges:
                slider = params.get(key, 0) / 100.0
                if slider == 0:
                    continue
                diff = np.abs(((hue_deg - center + 180) % 360) - 180)
                influence = np.clip(1.0 - diff / 40.0, 0.0, 1.0)
                weight += slider * influence
            grey = (
                0.299 * rgb[..., 0] +
                0.587 * rgb[..., 1] +
                0.114 * rgb[..., 2]
            )
            grey = np.clip(grey * (1.0 + weight), 0.0, 255.0)
            rgb = np.stack([grey, grey, grey], axis=-1)

        gradient_stops = getattr(self, "_gradient_map_stops", None)
        if gradient_stops and len(gradient_stops) >= 2:
            sorted_stops = sorted(gradient_stops, key=lambda s: s[0])
            positions = np.array([max(0.0, min(1.0, pos)) for pos, _ in sorted_stops], dtype=np.float32)
            colors = np.array(
                [[QColor(color).red(), QColor(color).green(), QColor(color).blue()] for _, color in sorted_stops],
                dtype=np.float32,
            )
            L_norm = np.clip(self._L_norm, 0.0, 1.0).reshape(-1)
            r = np.interp(L_norm, positions, colors[:, 0])
            g = np.interp(L_norm, positions, colors[:, 1])
            b = np.interp(L_norm, positions, colors[:, 2])
            mapped = np.stack([r, g, b], axis=-1).reshape(rgb.shape)
            rgb = mapped

        posterize_levels = params.get("posterize_levels", 0)
        if posterize_levels and posterize_levels > 1:
            bins = max(2, int(posterize_levels))
            step = 255.0 / (bins - 1)
            rgb = np.round(rgb / step) * step

        # write back to QImage buffer
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        arr[:, :w, :3] = rgb_u8

        # keep alpha as-is
        self._current_pixmap = QPixmap.fromImage(img)
        self.image_view.update_cpu_pixmap(self._current_pixmap)


    def _schedule_render(self):
        """Debounce rendering so rapid slider moves don't re-render every tick."""
        if not self.image_view.has_image():
            return
        # 40–60 ms feels responsive but avoids spamming grabFramebuffer
        self._render_timer.start(60)

    def _render_current_params(self):
        if self._preview_base_image is None:
            return
        self._apply_edit_params_to_current_image(self._current_params)
        self._update_histogram_from_view()

    def _push_params_to_view(self):
        if self._preview_base_image is None:
            return
        self.image_view.set_params(self._current_params)


    def _rescale_preview(self):
        if not self.image_view.has_image():
            return

        if self._zoom_mode == "fit":
            self._zoom_factor = self.image_view.fit_to_window()
        else:
            self._zoom_factor = self.image_view.set_manual_zoom(self._zoom_factor)
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

    def _on_thumb_ready(self, path: str, image: QImage):
        if image.isNull():
            icon = QIcon()
        else:
            pm = QPixmap.fromImage(image)
            pm = pm.scaled(
                self.thumbs.iconSize(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            icon = QIcon(pm)

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
        self.image_view.clear_image()

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
        if self.image_view.has_image() and self._zoom_mode == "fit":
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
                self.image_view.update_cpu_pixmap(self._current_pixmap)

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

