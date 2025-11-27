import os
from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QRunnable, Qt, Signal, QObject, QSize
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)
from PIL import Image

from .constants import IMAGE_EXTENSIONS
from .utils import load_qimage_any

# Optional CUDA for thumbnail scaling
try:
    from numba import cuda  # type: ignore
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False
    cuda = None

_USE_CUDA_THUMBS = _CUDA_AVAILABLE and os.environ.get("GS_USE_CUDA_THUMBS", "0") == "1"


def _mipped_resize_pil(pil: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Downsample progressively to avoid expensive single huge resize."""
    w, h = pil.size
    target_long = max(target_w, target_h)
    while min(w, h) > target_long * 2:
        w = max(target_w, w // 2)
        h = max(target_h, h // 2)
        pil = pil.resize((w, h), Image.BILINEAR)
    return pil.resize((target_w, target_h), Image.LANCZOS)


def set_cuda_thumbs_enabled(enabled: bool) -> bool:
    """Allow runtime toggling of CUDA thumbnail scaling."""
    global _USE_CUDA_THUMBS
    _USE_CUDA_THUMBS = bool(enabled) and _CUDA_AVAILABLE
    return _USE_CUDA_THUMBS


def get_cuda_thumbs_enabled() -> bool:
    return bool(_USE_CUDA_THUMBS and _CUDA_AVAILABLE)


if _CUDA_AVAILABLE:
    @cuda.jit(fastmath=True)
    def _thumb_resize_nearest(src, dst):
        y, x = cuda.grid(2)
        if y >= dst.shape[0] or x >= dst.shape[1]:
            return
        src_h = src.shape[0]
        src_w = src.shape[1]
        scale_y = src_h / dst.shape[0]
        scale_x = src_w / dst.shape[1]
        sy = int(y * scale_y)
        sx = int(x * scale_x)
        if sy >= src_h:
            sy = src_h - 1
        if sx >= src_w:
            sx = src_w - 1
        dst[y, x, 0] = src[sy, sx, 0]
        dst[y, x, 1] = src[sy, sx, 1]
        dst[y, x, 2] = src[sy, sx, 2]


class CollapsibleSection(QWidget):
    """Simple DxO-style collapsible panel."""

    def __init__(self, title: str, parent: Optional[QWidget] = None, start_collapsed: bool = False):
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


class HistogramWidget(QWidget):
    """RGB overlaid histogram (0–255) drawn as filled polygons."""

    def __init__(self, parent: Optional[QWidget] = None):
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

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        # Neutral panel background for histogram frame
        painter.fillRect(self.rect(), QColor(36, 36, 36))

        r = self.rect().adjusted(6, 6, -6, -6)
        painter.setPen(QPen(QColor(58, 58, 58)))

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


class ThumbnailFileSystemModel(QFileSystemModel):
    """QFileSystemModel that shows thumbnails for supported image files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumb_cache: dict[str, QIcon] = {}
        self._max_thumbs_per_folder = 500

    def _make_thumb_icon(self, path: str) -> Optional[QIcon]:
        try:
            img = load_qimage_any(path)
            if img is None:
                return None
            img = img.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return QIcon(QPixmap.fromImage(img))
        except Exception:
            return None

    def data(self, index, role=Qt.DecorationRole):  # noqa: D401,N802
        if role != Qt.DecorationRole:
            return super().data(index, role)

        if not index.isValid() or index.column() != 0:
            return super().data(index, role)

        parent_index = index.parent()
        if parent_index.isValid():
            try:
                if self.rowCount(parent_index) > self._max_thumbs_per_folder:
                    return super().data(index, role)
            except Exception:
                return super().data(index, role)

        path = self.filePath(index)
        if not path or self.isDir(index):
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
        icon = QIcon()
        target_w = max(1, self.size.width())
        target_h = max(1, self.size.height())

        # Skip CUDA for very small thumbs to avoid GPU under-utilization warnings
        use_cuda = _USE_CUDA_THUMBS and (target_w * target_h >= 4096)

        if use_cuda:
            try:
                pil = Image.open(self.path).convert("RGB")
                pil = _mipped_resize_pil(pil, target_w, target_h * 2)  # slight overshoot for crop-then-CUDA
                arr = np.asarray(pil, dtype=np.uint8)
                d_in = cuda.to_device(arr)
                d_out = cuda.device_array((target_h, target_w, 3), dtype=np.uint8)
                threads = (16, 16)
                blocks = (
                    (target_w + threads[0] - 1) // threads[0],
                    (target_h + threads[1] - 1) // threads[1],
                )
                _thumb_resize_nearest[blocks, threads](d_in, d_out)
                out = d_out.copy_to_host()
                img = QImage(out.data, target_w, target_h, 3 * target_w, QImage.Format_RGB888).copy()
                icon = QIcon(QPixmap.fromImage(img))
                self.signals.ready.emit(self.path, icon)
                return
            except Exception as exc:  # pragma: no cover - safety net
                # Corrupt or unsupported images should silently fall back to CPU path
                use_cuda = False

        img = load_qimage_any(self.path)
        if img is not None:
            # build a mip-like downscale before final quality resize
            try:
                pil = Image.fromqimage(img)
                pil = _mipped_resize_pil(pil, target_w, target_h)
                img = pil.toqimage()
            except Exception:
                img = img.scaled(self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon = QIcon(QPixmap.fromImage(img))

        self.signals.ready.emit(self.path, icon)


__all__ = [
    "CollapsibleSection",
    "HistogramWidget",
    "ThumbnailFileSystemModel",
    "_ThumbTask",
    "_WorkerSignals",
    "set_cuda_thumbs_enabled",
    "get_cuda_thumbs_enabled",
]
