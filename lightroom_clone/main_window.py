from __future__ import annotations
import base64
import copy
import os
import sys
import json
import cv2
import time
import tempfile
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from io import BytesIO
from typing import Callable, Iterable, NamedTuple

import numpy as np

if __package__ is None:
    from pathlib import Path

    package_path = Path(__file__).resolve().parent
    parent = str(package_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    __package__ = "lightroom_clone"

if __package__ is None:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QStatusBar,
    QListView,
    QSizePolicy,
    QLayout,
    QAbstractItemView,
    QStyle,
    QDockWidget,
    QSlider,
    QHBoxLayout,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QDialog,
    QFormLayout,
    QSpinBox,
    QDialogButtonBox,
    QMenu,
    QTabWidget,
    QTreeView,
    QFileSystemModel,
    QScrollArea,
    QPushButton,
    QFrame,
    QComboBox,
    QToolButton,
    QButtonGroup,
    QStackedWidget,
    QCheckBox,
    QDoubleSpinBox,
    QLineEdit,
    QTabBar,
    QColorDialog,
    QMenuBar,
)
from PySide6.QtCore import Qt, QSize, QPoint, QRect, QTimer, QDir, QEvent, QThreadPool, QRunnable, QObject, Signal
from PySide6.QtGui import QPixmap, QIcon, QFontMetrics, QColor, QImage, QPainter
from PIL import Image, ExifTags, ImageFilter

from .adjustments_panel import AdjustmentsPanel
from .groups import LightGroupWidget, ColorGroupWidget, ToneGroupWidget, FXGroupWidget
from .constants import (
    IMAGE_EXTENSIONS,
    LRC_VERSION,
    PROJECT_EXTENSION,
    LEGACY_PROJECT_EXTENSION,
)
from .color_tools import (
    apply_black_white,
    apply_color_balance,
    apply_color_white_balance,
    apply_hsl,
    apply_recolor,
    apply_selective_color,
    default_color_params,
    merge_color_params,
)
from .utils import (
    abspath_from_base as _abspath_from_base,
    load_image_full_linear as _load_image_full_linear,
    load_image_preview_linear as _load_image_preview_linear,
    load_linear_image as _load_linear_image,
    load_qimage_any as _load_qimage_any,
    qimage_from_linear,
    relpath_or_same as _relpath_or_same,
    to_linear,
    to_srgb,
)
from .widgets import (
    HistogramWidget,
    ThumbnailFileSystemModel,
    _ThumbTask,
    get_cuda_thumbs_enabled,
    set_cuda_thumbs_enabled,
)

# Optional acceleration backends
try:  # numba CPU JIT
    import numba  # type: ignore
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    njit = None
    prange = None

try:  # numba CUDA (optional)
    if _HAVE_NUMBA:
        from numba import cuda  # type: ignore
        _CUDA_AVAILABLE = cuda.is_available()
    else:
        _CUDA_AVAILABLE = False
        cuda = None
except Exception:
    _CUDA_AVAILABLE = False
    cuda = None

def _resource_path(*parts: str) -> str:
    """Resolve asset paths inside development sources or a PyInstaller bundle."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is None:
        bundle_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.normpath(os.path.join(bundle_root, *parts))


class OutputTransformMode(Enum):
    STANDARD = "standard"
    FILMIC = "filmic"


@dataclass
class EditParams:
    exposure: float = 0.0          # in stops (0 = neutral)
    contrast: float = 1.0          # multiplier (1.0 = neutral)
    highlights: float = 0.0        # -1..1
    shadows: float = 0.0           # -1..1
    whites: float = 0.0            # -1..1
    blacks: float = 0.0            # -1..1
    temperature: float = 0.0       # -1..1
    tint: float = 0.0              # -1..1
    vibrance: float = 0.0          # -1..1
    saturation: float = 1.0        # multiplier (1.0 = neutral)
    levels_black: float = 0.0
    levels_white: float = 1.0
    levels_gamma: float = 1.0
    levels_out_black: float = 0.0
    levels_out_white: float = 1.0
    levels_color_model: str = "RGB"
    levels_channel: str = "Master"
    color: dict | None = None
    curve_param_highlights: float = 0.0
    curve_param_lights: float = 0.0
    curve_param_darks: float = 0.0
    curve_param_shadows: float = 0.0
    curve_points_rgb: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_r: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_g: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_points_b: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    chmix_red_r: float = 1.0
    chmix_red_g: float = 0.0
    chmix_red_b: float = 0.0
    chmix_green_r: float = 0.0
    chmix_green_g: float = 1.0
    chmix_green_b: float = 0.0
    chmix_blue_r: float = 0.0
    chmix_blue_g: float = 0.0
    chmix_blue_b: float = 1.0
    grad_stops: list[dict] = field(default_factory=lambda: [
        {"pos": 0.0, "color": (0.0, 0.0, 0.0), "opacity": 1.0},
        {"pos": 1.0, "color": (1.0, 1.0, 1.0), "opacity": 1.0},
    ])
    grad_blend_mode: str = "Normal"
    grad_opacity: float = 0.0
    split_shadow_hue: float = 0.0
    split_shadow_sat: float = 0.0
    split_mid_hue: float = 0.0
    split_mid_sat: float = 0.0
    split_high_hue: float = 0.0
    split_high_sat: float = 0.0
    split_balance: float = 0.0
    normal_map: np.ndarray | None = None
    normal_light_x: float = 0.0
    normal_light_y: float = 0.0
    normal_light_elev: float = 0.5
    normal_intensity: float = 1.0
    normal_specular: float = 0.0
    normal_diffuse: float = 1.0
    view_exposure_bias: float = 0.0
    output_transform: OutputTransformMode = OutputTransformMode.STANDARD
    # geometry placeholder; populated by _dict_to_geometry_params
    geometry: "GeometryParams" | None = None


@dataclass
class GeometryParams:
    # Transform
    rotate_deg: float = 0.0
    scale_uniform: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    anchor: str = "center"

    # Perspective
    vertical_persp: float = 0.0
    horizontal_persp: float = 0.0
    rotate_x_deg: float = 0.0
    rotate_y_deg: float = 0.0
    upright_mode: str = "off"
    guided_lines: list = field(default_factory=list)

    # Lens
    lens_profile_enabled: bool = False
    lens_profile_amount: float = 1.0
    lens_profile_name: str = ""
    distortion: float = 0.0
    vignette_amount: float = 0.0
    vignette_midpoint: float = 50.0
    defish_amount: float = 0.0

    # Crop & Guides
    crop_enabled: bool = False
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_w: float = 1.0
    crop_h: float = 1.0
    crop_aspect_mode: str = "original"
    guides_mode: str = "none"
    show_safe_areas: bool = False

    # Warp
    keystone_top: float = 0.0
    keystone_bottom: float = 0.0
    volume_deform: float = 0.0
    mesh_enabled: bool = False
    mesh_rows: int = 4
    mesh_cols: int = 4


class TileKey(NamedTuple):
    level: int
    tx: int
    ty: int


# --- Acceleration toggles ---
_USE_NUMBA_BASIC = _HAVE_NUMBA and os.environ.get("GS_USE_NUMBA", "1") == "1"
# Default to CUDA when available; users can opt-out with GS_USE_CUDA=0
_USE_CUDA_BASIC = _CUDA_AVAILABLE and os.environ.get("GS_USE_CUDA", "1") == "1"
_USE_NUMBA_SHARPEN = _HAVE_NUMBA and os.environ.get("GS_USE_NUMBA_SHARPEN", "1") == "1"
_USE_CUDA_SHARPEN = _CUDA_AVAILABLE and os.environ.get("GS_USE_CUDA_SHARPEN", "1") == "1"


def get_cuda_basic_enabled() -> bool:
    return _USE_CUDA_BASIC


def set_cuda_basic_enabled(enabled: bool):
    global _USE_CUDA_BASIC
    _USE_CUDA_BASIC = bool(enabled and _CUDA_AVAILABLE)


def get_cuda_sharpen_enabled() -> bool:
    return _USE_CUDA_SHARPEN


def set_cuda_sharpen_enabled(enabled: bool):
    global _USE_CUDA_SHARPEN
    _USE_CUDA_SHARPEN = bool(enabled and _CUDA_AVAILABLE)


def _f32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def _basic_numba_kernel(src: np.ndarray, temp: float, tint: float, exposure: float,
                            contrast: float, hi: float, sh: float, wh: float, bl: float,
                            saturation: float) -> np.ndarray:
        h, w, _ = src.shape
        out = np.empty_like(src)
        exp_mul = 2.0 ** exposure
        for y in prange(h):
            for x in range(w):
                r = src[y, x, 0]
                g = src[y, x, 1]
                b = src[y, x, 2]

                if temp != 0.0:
                    r *= (1.0 + temp)
                    b *= (1.0 - temp)
                if tint != 0.0:
                    g *= (1.0 - tint)
                    r *= (1.0 + tint * 0.5)
                    b *= (1.0 + tint * 0.5)

                if exposure != 0.0:
                    r *= exp_mul
                    g *= exp_mul
                    b *= exp_mul

                if contrast != 1.0:
                    mid = 0.5
                    r = (r - mid) * contrast + mid
                    g = (g - mid) * contrast + mid
                    b = (b - mid) * contrast + mid

                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                if lum < 0.0:
                    lum = 0.0
                if lum > 1.0:
                    lum = 1.0

                if hi != 0.0 or sh != 0.0 or wh != 0.0 or bl != 0.0:
                    highlights = (lum - 0.5) / 0.5
                    if highlights < 0.0:
                        highlights = 0.0
                    if highlights > 1.0:
                        highlights = 1.0

                    shadows = (0.5 - lum) / 0.5
                    if shadows < 0.0:
                        shadows = 0.0
                    if shadows > 1.0:
                        shadows = 1.0

                    whites = (lum - 0.8) / 0.2
                    if whites < 0.0:
                        whites = 0.0
                    if whites > 1.0:
                        whites = 1.0

                    blacks = (0.2 - lum) / 0.2
                    if blacks < 0.0:
                        blacks = 0.0
                    if blacks > 1.0:
                        blacks = 1.0

                    r += hi * highlights + sh * shadows + wh * whites + bl * blacks
                    g += hi * highlights + sh * shadows + wh * whites + bl * blacks
                    b += hi * highlights + sh * shadows + wh * whites + bl * blacks

                if r < 0.0:
                    r = 0.0
                if g < 0.0:
                    g = 0.0
                if b < 0.0:
                    b = 0.0
                if r > 1.0:
                    r = 1.0
                if g > 1.0:
                    g = 1.0
                if b > 1.0:
                    b = 1.0

                lum2 = 0.2126 * r + 0.7152 * g + 0.0722 * b
                if lum2 < 0.0:
                    lum2 = 0.0
                if lum2 > 1.0:
                    lum2 = 1.0
                if saturation != 1.0:
                    r = lum2 + (r - lum2) * saturation
                    g = lum2 + (g - lum2) * saturation
                    b = lum2 + (b - lum2) * saturation

                out[y, x, 0] = r
                out[y, x, 1] = g
                out[y, x, 2] = b
        return out


if _CUDA_AVAILABLE:
    # CUDA kernel for the same basic operations (WB + tone + saturation)
    @cuda.jit(fastmath=True)
    def _basic_cuda_kernel(src, dst, temp, tint, exposure, contrast, hi, sh, wh, bl, saturation):
        y, x = cuda.grid(2)
        if y >= src.shape[0] or x >= src.shape[1]:
            return
        r = src[y, x, 0]
        g = src[y, x, 1]
        b = src[y, x, 2]

        if temp != 0.0:
            r *= (1.0 + temp)
            b *= (1.0 - temp)
        if tint != 0.0:
            g *= (1.0 - tint)
            r *= (1.0 + tint * 0.5)
            b *= (1.0 + tint * 0.5)

        if exposure != 0.0:
            mul = 2.0 ** exposure
            r *= mul
            g *= mul
            b *= mul

        if contrast != 1.0:
            mid = 0.5
            r = (r - mid) * contrast + mid
            g = (g - mid) * contrast + mid
            b = (b - mid) * contrast + mid

        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum < 0.0:
            lum = 0.0
        if lum > 1.0:
            lum = 1.0

        if hi != 0.0 or sh != 0.0 or wh != 0.0 or bl != 0.0:
            highlights = (lum - 0.5) / 0.5
            if highlights < 0.0:
                highlights = 0.0
            if highlights > 1.0:
                highlights = 1.0

            shadows = (0.5 - lum) / 0.5
            if shadows < 0.0:
                shadows = 0.0
            if shadows > 1.0:
                shadows = 1.0

            whites = (lum - 0.8) / 0.2
            if whites < 0.0:
                whites = 0.0
            if whites > 1.0:
                whites = 1.0

            blacks = (0.2 - lum) / 0.2
            if blacks < 0.0:
                blacks = 0.0
            if blacks > 1.0:
                blacks = 1.0

            r += hi * highlights + sh * shadows + wh * whites + bl * blacks
            g += hi * highlights + sh * shadows + wh * whites + bl * blacks
            b += hi * highlights + sh * shadows + wh * whites + bl * blacks

        if r < 0.0:
            r = 0.0
        if g < 0.0:
            g = 0.0
        if b < 0.0:
            b = 0.0
        if r > 1.0:
            r = 1.0
        if g > 1.0:
            g = 1.0
        if b > 1.0:
            b = 1.0

        lum2 = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum2 < 0.0:
            lum2 = 0.0
        if lum2 > 1.0:
            lum2 = 1.0

        if saturation != 1.0:
            r = lum2 + (r - lum2) * saturation
            g = lum2 + (g - lum2) * saturation
            b = lum2 + (b - lum2) * saturation

        dst[y, x, 0] = r
        dst[y, x, 1] = g
        dst[y, x, 2] = b


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def _sharpen_numba_kernel(src: np.ndarray, amount: float) -> np.ndarray:
        h, w, _ = src.shape
        dst = np.empty_like(src)
        for y in prange(h):
            y0 = max(0, y - 1)
            y1 = min(h - 1, y + 1)
            for x in range(w):
                x0 = max(0, x - 1)
                x1 = min(w - 1, x + 1)
                r = 0.0
                g = 0.0
                b = 0.0
                count = 0
                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        r += src[yy, xx, 0]
                        g += src[yy, xx, 1]
                        b += src[yy, xx, 2]
                        count += 1
                blur_r = r / count
                blur_g = g / count
                blur_b = b / count
                src_r = src[y, x, 0]
                src_g = src[y, x, 1]
                src_b = src[y, x, 2]
                dst_r = src_r + amount * (src_r - blur_r)
                dst_g = src_g + amount * (src_g - blur_g)
                dst_b = src_b + amount * (src_b - blur_b)
                if dst_r < 0.0:
                    dst_r = 0.0
                if dst_g < 0.0:
                    dst_g = 0.0
                if dst_b < 0.0:
                    dst_b = 0.0
                if dst_r > 1.0:
                    dst_r = 1.0
                if dst_g > 1.0:
                    dst_g = 1.0
                if dst_b > 1.0:
                    dst_b = 1.0
                dst[y, x, 0] = dst_r
                dst[y, x, 1] = dst_g
                dst[y, x, 2] = dst_b
        return dst


if _CUDA_AVAILABLE:
    @cuda.jit(fastmath=True)
    def _sharpen_cuda_kernel(src, dst, amount: float):
        y, x = cuda.grid(2)
        if y >= src.shape[0] or x >= src.shape[1]:
            return
        h = src.shape[0]
        w = src.shape[1]
        r = 0.0
        g = 0.0
        b = 0.0
        count = 0.0
        for yy in range(max(0, y - 1), min(h - 1, y + 1) + 1):
            for xx in range(max(0, x - 1), min(w - 1, x + 1) + 1):
                r += src[yy, xx, 0]
                g += src[yy, xx, 1]
                b += src[yy, xx, 2]
                count += 1.0
        blur_r = r / count
        blur_g = g / count
        blur_b = b / count

        src_r = src[y, x, 0]
        src_g = src[y, x, 1]
        src_b = src[y, x, 2]
        dst_r = src_r + amount * (src_r - blur_r)
        dst_g = src_g + amount * (src_g - blur_g)
        dst_b = src_b + amount * (src_b - blur_b)

        if dst_r < 0.0:
            dst_r = 0.0
        if dst_g < 0.0:
            dst_g = 0.0
        if dst_b < 0.0:
            dst_b = 0.0
        if dst_r > 1.0:
            dst_r = 1.0
        if dst_g > 1.0:
            dst_g = 1.0
        if dst_b > 1.0:
            dst_b = 1.0

        dst[y, x, 0] = dst_r
        dst[y, x, 1] = dst_g
        dst[y, x, 2] = dst_b

_NORMAL_MAP_CACHE: dict[str, np.ndarray] = {}


def _normalize_curve_points(points: list[tuple[float, float]] | list | None) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for pt in points or []:
        try:
            x, y = pt
        except Exception:
            continue
        x = float(max(0.0, min(1.0, x)))
        y = float(max(0.0, min(1.0, y)))
        pts.append((x, y))
    if not pts:
        pts = [(0.0, 0.0), (1.0, 1.0)]
    pts = sorted(pts, key=lambda p: p[0])
    if pts[0][0] > 0.0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < 1.0:
        pts.append((1.0, pts[-1][1]))
    dedup: list[tuple[float, float]] = []
    last_x = None
    for x, y in pts:
        if last_x is not None and abs(x - last_x) < 1e-4:
            dedup[-1] = (x, y)
        else:
            dedup.append((x, y))
        last_x = x
    return dedup


def _normalize_grad_stops(stops: list[dict] | None) -> list[dict]:
    if not stops:
        stops = [
            {"pos": 0.0, "color": (0.0, 0.0, 0.0), "opacity": 1.0},
            {"pos": 1.0, "color": (1.0, 1.0, 1.0), "opacity": 1.0},
        ]
    norm: list[dict] = []
    for st in stops:
        pos = float(st.get("pos", 0.0))
        col = st.get("color", (0.0, 0.0, 0.0))
        opacity = float(st.get("opacity", 1.0))
        try:
            r, g, b = col
        except Exception:
            r = g = b = 0.0
        norm.append({
            "pos": max(0.0, min(1.0, pos)),
            "color": (float(r), float(g), float(b)),
            "opacity": max(0.0, min(1.0, opacity)),
        })
    norm = sorted(norm, key=lambda s: s["pos"])
    if norm[0]["pos"] > 0.0:
        norm.insert(0, {"pos": 0.0, "color": norm[0]["color"], "opacity": norm[0]["opacity"]})
    if norm[-1]["pos"] < 1.0:
        norm.append({"pos": 1.0, "color": norm[-1]["color"], "opacity": norm[-1]["opacity"]})
    return norm


def _decode_normal_map_entry(entry) -> np.ndarray | None:
    if entry is None:
        return None
    try:
        import hashlib
        cache_key = None
        if isinstance(entry, dict):
            if "data" in entry:
                data = entry.get("data", "")
                cache_key = f"data:{hashlib.sha1(data.encode('utf-8')).hexdigest()}"
                if cache_key in _NORMAL_MAP_CACHE:
                    return _NORMAL_MAP_CACHE[cache_key]
                raw = base64.b64decode(data)
                img = Image.open(BytesIO(raw)).convert("RGB")
                arr = np.asarray(img, dtype=np.float32) / 255.0
            elif "path" in entry:
                cache_key = f"path:{entry.get('path')}"
                if cache_key in _NORMAL_MAP_CACHE:
                    return _NORMAL_MAP_CACHE[cache_key]
                arr = _load_linear_image(entry.get("path"))  # type: ignore
            else:
                arr = None
        elif isinstance(entry, list):
            arr = np.asarray(entry, dtype=np.float32)
        else:
            arr = None

        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr, arr, np.ones_like(arr)], axis=-1)
        arr = np.clip(arr, 0.0, 1.0)
        if cache_key:
            _NORMAL_MAP_CACHE[cache_key] = arr
        return arr
    except Exception:
        return None


def _coerce_output_transform(val) -> OutputTransformMode:
    if isinstance(val, OutputTransformMode):
        return val
    if isinstance(val, str):
        lower = val.lower()
        if lower == OutputTransformMode.FILMIC.value:
            return OutputTransformMode.FILMIC
    return OutputTransformMode.STANDARD


def _dict_to_edit_params(params: dict) -> EditParams:
    """Normalize UI-friendly dict parameters into an EditParams dataclass."""
    params = params or {}
    get = params.get

    def center(val: float, scale: float = 1.0) -> float:
        return (float(val) - 128.0) / 128.0 * scale

    def _contrast_scale(raw_val: float) -> float:
        """Map 0-255 UI value to a gentle 0.5-1.5 contrast multiplier."""
        return 1.0 + ((raw_val - 128.0) / 128.0) * 0.5

    return EditParams(
        exposure=center(get("exposure", 128), 2.0),
        contrast=_contrast_scale(float(get("contrast", 128))),
        highlights=center(get("highlights", 128), 0.6),
        shadows=center(get("shadows", 128), 0.6),
        whites=center(get("whites", 128), 0.8),
        blacks=center(get("blacks", 128), 0.8),
        temperature=center(get("temperature", 128), 0.5),
        tint=center(get("tint", 128), 0.5),
        vibrance=center(get("vibrance", 128), 0.75),
        saturation=float(get("saturation", 128)) / 128.0,
        levels_black=float(get("levels_black", 0)) / 100.0,
        levels_white=float(get("levels_white", 100)) / 100.0,
        levels_gamma=float(get("levels_gamma", 1.0)),
        levels_out_black=float(get("levels_out_black", 0)) / 100.0,
        levels_out_white=float(get("levels_out_white", 100)) / 100.0,
        levels_color_model=get("levels_color_model", "RGB"),
        levels_channel=get("levels_channel", "Master"),
        color=merge_color_params(get("color"), params),
        curve_param_highlights=float(get("curve_param_highlights", 0)) / 100.0,
        curve_param_lights=float(get("curve_param_lights", 0)) / 100.0,
        curve_param_darks=float(get("curve_param_darks", 0)) / 100.0,
        curve_param_shadows=float(get("curve_param_shadows", 0)) / 100.0,
        curve_points_rgb=_normalize_curve_points(get("curve_points_rgb", [])),
        curve_points_r=_normalize_curve_points(get("curve_points_r", [])),
        curve_points_g=_normalize_curve_points(get("curve_points_g", [])),
        curve_points_b=_normalize_curve_points(get("curve_points_b", [])),
        chmix_red_r=float(get("chmix_red_r", 100)) / 100.0,
        chmix_red_g=float(get("chmix_red_g", 0)) / 100.0,
        chmix_red_b=float(get("chmix_red_b", 0)) / 100.0,
        chmix_green_r=float(get("chmix_green_r", 0)) / 100.0,
        chmix_green_g=float(get("chmix_green_g", 100)) / 100.0,
        chmix_green_b=float(get("chmix_green_b", 0)) / 100.0,
        chmix_blue_r=float(get("chmix_blue_r", 0)) / 100.0,
        chmix_blue_g=float(get("chmix_blue_g", 0)) / 100.0,
        chmix_blue_b=float(get("chmix_blue_b", 100)) / 100.0,
        grad_stops=_normalize_grad_stops(get("grad_stops")),
        grad_blend_mode=get("grad_blend_mode", "Normal"),
        grad_opacity=float(get("grad_opacity", 0)) / 100.0,
        split_shadow_hue=float(get("split_shadow_hue", 0)),
        split_shadow_sat=float(get("split_shadow_sat", 0)) / 100.0,
        split_mid_hue=float(get("split_mid_hue", 0)),
        split_mid_sat=float(get("split_mid_sat", 0)) / 100.0,
        split_high_hue=float(get("split_high_hue", 0)),
        split_high_sat=float(get("split_high_sat", 0)) / 100.0,
        split_balance=float(get("split_balance", 0)) / 100.0,
        normal_map=_decode_normal_map_entry(get("normal_map")),
        normal_light_x=float(get("normal_light_x", 0)) / 100.0,
        normal_light_y=float(get("normal_light_y", 0)) / 100.0,
        normal_light_elev=float(get("normal_light_elev", 50)) / 100.0,
        normal_intensity=float(get("normal_intensity", 100)) / 100.0,
        normal_specular=float(get("normal_specular", 0)) / 100.0,
        normal_diffuse=float(get("normal_diffuse", 100)) / 100.0,
        view_exposure_bias=float(get("view_exposure_bias", 0.0)),
        output_transform=_coerce_output_transform(get("output_transform", OutputTransformMode.STANDARD.value)),
        geometry=_dict_to_geometry_params(get("geometry")),
    )


def _dict_to_geometry_params(params: dict | None) -> GeometryParams:
    p = params or {}
    amt = p.get("lens_profile_amount", p.get("profile_amount", 100))
    if amt is None:
        amt = 0
    amt = float(amt)
    if amt > 1.5:  # assume percent
        amt /= 100.0
    return GeometryParams(
        rotate_deg=float(p.get("rotate_deg", 0.0)),
        scale_uniform=float(p.get("scale_uniform", 1.0)),
        scale_x=float(p.get("scale_x", 1.0)),
        scale_y=float(p.get("scale_y", 1.0)),
        offset_x=float(p.get("offset_x", 0.0)),
        offset_y=float(p.get("offset_y", 0.0)),
        flip_horizontal=bool(p.get("flip_horizontal", False)),
        flip_vertical=bool(p.get("flip_vertical", False)),
        anchor=p.get("anchor", "center"),
        vertical_persp=float(p.get("vertical_persp", 0.0)),
        horizontal_persp=float(p.get("horizontal_persp", 0.0)),
        rotate_x_deg=float(p.get("rotate_x_deg", 0.0)),
        rotate_y_deg=float(p.get("rotate_y_deg", 0.0)),
        upright_mode=p.get("upright_mode", "off"),
        guided_lines=list(p.get("guided_lines", [])),
        lens_profile_enabled=bool(p.get("lens_profile_enabled", p.get("use_lens_profile", False))),
        lens_profile_amount=max(0.0, min(1.0, amt)),
        lens_profile_name=p.get("lens_profile_name", ""),
        distortion=float(p.get("distortion", 0.0)),
        vignette_amount=float(p.get("vignette_amount", 0.0)),
        vignette_midpoint=float(p.get("vignette_midpoint", 50.0)),
        defish_amount=float(p.get("defish_amount", 0.0)),
        crop_enabled=bool(p.get("crop_enabled", False)),
        crop_x=float(p.get("crop_x", 0.0)),
        crop_y=float(p.get("crop_y", 0.0)),
        crop_w=float(p.get("crop_w", 1.0)),
        crop_h=float(p.get("crop_h", 1.0)),
        crop_aspect_mode=p.get("crop_aspect_mode", "original"),
        guides_mode=p.get("guides_mode", "none"),
        show_safe_areas=bool(p.get("show_safe_areas", False)),
        keystone_top=float(p.get("keystone_top", 0.0)),
        keystone_bottom=float(p.get("keystone_bottom", 0.0)),
        volume_deform=float(p.get("volume_deform", 0.0)),
        mesh_enabled=bool(p.get("mesh_enabled", False)),
        mesh_rows=int(p.get("mesh_rows", 4)),
        mesh_cols=int(p.get("mesh_cols", 4)),
    )


def apply_white_balance(image: np.ndarray, params: EditParams, quality: str) -> np.ndarray:
    rgb = image
    if abs(params.temperature) > 1e-4:
        t = params.temperature
        rgb[..., 0] *= (1.0 + t)
        rgb[..., 2] *= (1.0 - t)

    if abs(params.tint) > 1e-4:
        tt = params.tint
        rgb[..., 1] *= (1.0 - tt)
        rgb[..., 0] *= (1.0 + tt * 0.5)
        rgb[..., 2] *= (1.0 + tt * 0.5)
    return rgb


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply_parametric_curve(rgb: np.ndarray, params: EditParams) -> np.ndarray:
    if all(abs(v) < 1e-4 for v in [
        params.curve_param_highlights,
        params.curve_param_lights,
        params.curve_param_darks,
        params.curve_param_shadows,
    ]):
        return rgb

    lum = np.clip(
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2],
        0.0,
        1.0,
    )

    w_shadow = 1.0 - _smoothstep(0.18, 0.42, lum)
    w_dark = _smoothstep(0.08, 0.35, lum) * (1.0 - _smoothstep(0.55, 0.75, lum))
    w_light = _smoothstep(0.45, 0.72, lum) * (1.0 - _smoothstep(0.82, 0.95, lum))
    w_high = _smoothstep(0.60, 0.95, lum)

    delta = (
        params.curve_param_shadows * w_shadow +
        params.curve_param_darks * w_dark +
        params.curve_param_lights * w_light +
        params.curve_param_highlights * w_high
    ) * 0.35

    lum_new = np.clip(lum + delta, 0.0, 1.0)
    scale = np.where(lum > 1e-4, lum_new / np.maximum(lum, 1e-4), 1.0)
    return np.clip(rgb * scale[..., None], 0.0, 1.0)


def _build_curve_lut(points: list[tuple[float, float]], size: int = 1024) -> np.ndarray:
    pts = _normalize_curve_points(points)
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

    t_vals = np.linspace(0.0, 1.0, size, dtype=np.float32)
    lut = np.zeros(size, dtype=np.float32)
    for idx, tx in enumerate(t_vals):
        seg = int(np.clip(np.searchsorted(xs, tx) - 1, 0, len(xs) - 2))
        h = xs[seg + 1] - xs[seg]
        if h <= 0:
            lut[idx] = ys[seg]
            continue
        t = (tx - xs[seg]) / h
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        lut[idx] = (
            h00 * ys[seg]
            + h10 * h * m[seg]
            + h01 * ys[seg + 1]
            + h11 * h * m[seg + 1]
        )
    return np.clip(lut, 0.0, 1.0)


def _is_linear_curve(points: list[tuple[float, float]]) -> bool:
    pts = _normalize_curve_points(points)
    return len(pts) <= 2 and abs(pts[0][1]) < 1e-4 and abs(pts[-1][1] - 1.0) < 1e-4


def _apply_point_curves(rgb: np.ndarray, params: EditParams) -> np.ndarray:
    out = np.clip(rgb, 0.0, 1.0)
    lut_size = 1024
    if not _is_linear_curve(params.curve_points_rgb):
        lut = _build_curve_lut(params.curve_points_rgb, lut_size)
        idx = np.clip((out * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
        out = lut[idx]

    for key, channel_idx in [
        ("curve_points_r", 0),
        ("curve_points_g", 1),
        ("curve_points_b", 2),
    ]:
        pts = getattr(params, key, None)
        if pts is None or _is_linear_curve(pts):
            continue
        lut = _build_curve_lut(pts, lut_size)
        channel = out[..., channel_idx]
        idx = np.clip((channel * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
        out[..., channel_idx] = lut[idx]
    return np.clip(out, 0.0, 1.0)


def apply_tone_curve(image: np.ndarray, params: EditParams, quality: str) -> np.ndarray:
    rgb = image
    in_black = params.levels_black
    in_white = params.levels_white
    out_black = params.levels_out_black
    out_white = params.levels_out_white
    gamma = params.levels_gamma

    color_model = params.levels_color_model
    channel_name = params.levels_channel

    if channel_name == "Master" and color_model in ("Red", "Green", "Blue"):
        channel_name = color_model

    if channel_name == "Red":
        chan_idx = [0]
    elif channel_name == "Green":
        chan_idx = [1]
    elif channel_name == "Blue":
        chan_idx = [2]
    else:
        chan_idx = [0, 1, 2]

    if not (
        abs(in_black) < 1e-3 and
        abs(in_white - 1.0) < 1e-3 and
        abs(gamma - 1.0) < 1e-3 and
        abs(out_black) < 1e-3 and
        abs(out_white - 1.0) < 1e-3
    ):
        rgb_norm = np.clip(rgb, 0.0, 1.0)
        in_span = max(1e-6, in_white - in_black)
        out_span = max(1e-6, out_white - out_black)

        sub = rgb_norm[..., chan_idx]
        sub = (sub - in_black) / in_span
        sub = np.clip(sub, 0.0, 1.0)

        if abs(gamma - 1.0) > 1e-3:
            sub = np.power(sub, 1.0 / gamma)

        sub = out_black + sub * out_span
        rgb_norm[..., chan_idx] = sub
        rgb = rgb_norm

    rgb = _apply_parametric_curve(rgb, params)
    rgb = _apply_point_curves(rgb, params)
    return rgb


def apply_basic_adjustments(image: np.ndarray, params: EditParams, quality: str, allow_cuda: bool = True) -> np.ndarray:
    accel = _apply_basic_adjustments_accel(image, params, allow_cuda=allow_cuda)
    if accel is not None:
        return accel

    rgb = image
    lum = (
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2]
    )
    lum3 = lum[..., None]

    if abs(params.exposure) > 1e-5:
        rgb *= 2.0 ** params.exposure

    if abs(params.contrast - 1.0) > 1e-5:
        mid = 0.5
        rgb = (rgb - mid) * params.contrast + mid

    if any(abs(x) > 1e-3 for x in (params.highlights, params.shadows, params.whites, params.blacks)):
        L = np.clip(lum, 0.0, 1.0)
        highlights_mask = np.clip((L - 0.5) / 0.5, 0.0, 1.0)
        shadows_mask = np.clip((0.5 - L) / 0.5, 0.0, 1.0)
        whites_mask = np.clip((L - 0.8) / 0.2, 0.0, 1.0)
        blacks_mask = np.clip((0.2 - L) / 0.2, 0.0, 1.0)

        rgb += params.highlights * highlights_mask[..., None]
        rgb += params.shadows * shadows_mask[..., None]
        rgb += params.whites * whites_mask[..., None]
        rgb += params.blacks * blacks_mask[..., None]

    rgb = np.clip(rgb, 0.0, 1.0)
    if abs(params.saturation - 1.0) > 1e-3:
        rgb = lum3 + (rgb - lum3) * params.saturation
    return np.clip(rgb, 0.0, 1.0)


def apply_color_adjustments(image: np.ndarray, params: EditParams, quality: str) -> np.ndarray:
    rgb = image
    lum = (
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2]
    )
    lum3 = lum[..., None]

    if abs(params.vibrance) > 1e-4:
        sat_dist = np.mean(np.abs(rgb - lum3), axis=-1)
        weight = (1.0 - np.clip(sat_dist, 0.0, 1.0))[..., None]
        rgb = lum3 + (rgb - lum3) * (1.0 + params.vibrance * weight)

    rgb = np.clip(rgb, 0.0, 1.0)

    color_params = params.color or {}
    rgb = apply_color_white_balance(rgb, color_params.get("white_balance", {}))
    rgb = apply_hsl(rgb, color_params.get("hsl", {}))
    rgb = apply_selective_color(rgb, color_params.get("selective_color", {}))
    rgb = apply_color_balance(rgb, color_params.get("color_balance", {}))
    rgb = apply_recolor(rgb, color_params.get("recolor", {}))
    rgb = apply_black_white(rgb, color_params.get("black_white", {}))
    return np.clip(rgb, 0.0, 1.0)


def apply_channel_mixer(image: np.ndarray, params: EditParams, quality: str) -> np.ndarray:
    mat = np.array([
        [params.chmix_red_r, params.chmix_red_g, params.chmix_red_b],
        [params.chmix_green_r, params.chmix_green_g, params.chmix_green_b],
        [params.chmix_blue_r, params.chmix_blue_g, params.chmix_blue_b],
    ], dtype=np.float32)
    mixed = np.einsum("...c,cd->...d", image, mat, dtype=np.float32)
    return np.clip(mixed, 0.0, 1.0)


def _blend_layers(base: np.ndarray, blend: np.ndarray, mode: str) -> np.ndarray:
    mode_l = (mode or "normal").lower()
    if mode_l == "multiply":
        return base * blend
    if mode_l == "screen":
        return 1.0 - (1.0 - base) * (1.0 - blend)
    if mode_l == "overlay":
        return np.where(base < 0.5, 2.0 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend))
    if mode_l == "soft light":
        return (1.0 - 2.0 * blend) * base * base + 2.0 * blend * base
    return blend


def apply_gradient_map(image: np.ndarray, params: EditParams, quality: str) -> np.ndarray:
    stops = params.grad_stops or []
    if params.grad_opacity <= 1e-4:
        return image
    if len(stops) < 2:
        return image

    positions = np.array([s.get("pos", 0.0) for s in stops], dtype=np.float32)
    colors = np.array([s.get("color", (0.0, 0.0, 0.0)) for s in stops], dtype=np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    opacities = np.array([s.get("opacity", 1.0) for s in stops], dtype=np.float32)

    lut_size = 1024
    t_vals = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    lut_r = np.interp(t_vals, positions, colors[:, 0])
    lut_g = np.interp(t_vals, positions, colors[:, 1])
    lut_b = np.interp(t_vals, positions, colors[:, 2])
    lut_a = np.interp(t_vals, positions, opacities)

    lum = np.clip(
        0.2126 * image[..., 0] +
        0.7152 * image[..., 1] +
        0.0722 * image[..., 2],
        0.0,
        1.0,
    )
    idx = np.clip((lum * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
    grad_rgb = np.stack([lut_r[idx], lut_g[idx], lut_b[idx]], axis=-1)
    alpha = np.clip(params.grad_opacity, 0.0, 1.0) * lut_a[idx][..., None]
    blended = _blend_layers(image, grad_rgb, params.grad_blend_mode)
    rgb = image * (1.0 - alpha) + blended * alpha
    return np.clip(rgb, 0.0, 1.0)


def apply_split_toning(image: np.ndarray, params: EditParams, quality: str) -> np.ndarray:
    rgb = np.clip(image, 0.0, 1.0)
    lum = np.clip(
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2],
        0.0,
        1.0,
    )

    def _hue_sat_to_rgb(h_deg: float, sat: float) -> np.ndarray:
        h = (h_deg % 360.0) / 360.0
        k = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float32)
        col = np.clip(np.abs(((h + k) % 1.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0)
        return col * sat + (1.0 - sat)

    bal = params.split_balance
    shadow_cut = 0.5 - bal * 0.2
    highlight_cut = 0.5 + bal * 0.2
    shadow_w = np.clip((shadow_cut - lum) / max(shadow_cut, 1e-6), 0.0, 1.0)
    highlight_w = np.clip((lum - highlight_cut) / max(1.0 - highlight_cut, 1e-6), 0.0, 1.0)
    mid_w = np.clip(1.0 - shadow_w - highlight_w, 0.0, 1.0)
    weight_sum = shadow_w + mid_w + highlight_w
    weight_sum = np.where(weight_sum < 1e-6, 1.0, weight_sum)
    shadow_w /= weight_sum
    mid_w /= weight_sum
    highlight_w /= weight_sum

    shadow_col = _hue_sat_to_rgb(params.split_shadow_hue, 1.0)
    mid_col = _hue_sat_to_rgb(params.split_mid_hue, 1.0)
    high_col = _hue_sat_to_rgb(params.split_high_hue, 1.0)

    tint = (
        shadow_w[..., None] * params.split_shadow_sat * shadow_col +
        mid_w[..., None] * params.split_mid_sat * mid_col +
        highlight_w[..., None] * params.split_high_sat * high_col
    )
    alpha = np.clip(
        shadow_w * params.split_shadow_sat +
        mid_w * params.split_mid_sat +
        highlight_w * params.split_high_sat,
        0.0,
        1.5,
    )[..., None]

    return np.clip(rgb * (1.0 - alpha) + tint, 0.0, 1.0)


def apply_normal_relighting(image: np.ndarray, params: EditParams, quality: str) -> np.ndarray:
    normals = params.normal_map
    if normals is None:
        return image

    arr = np.asarray(normals, dtype=np.float32)
    if arr.max() > 1.001 or arr.min() < -0.001:
        arr = np.clip(arr, 0.0, 255.0) / 255.0
    if arr.shape[:2] != image.shape[:2]:
        normal_img = Image.fromarray(np.clip((arr * 255.0).round(), 0, 255).astype(np.uint8))
        normal_img = normal_img.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
        arr = np.asarray(normal_img, dtype=np.float32) / 255.0

    normals_vec = arr * 2.0 - 1.0
    norm_len = np.linalg.norm(normals_vec, axis=-1, keepdims=True)
    norm_len = np.where(norm_len < 1e-6, 1.0, norm_len)
    normals_vec = normals_vec / norm_len

    lx = params.normal_light_x
    ly = params.normal_light_y
    lz = max(0.05, params.normal_light_elev)
    light_dir = np.array([lx, ly, lz], dtype=np.float32)
    light_dir = light_dir / max(1e-6, np.linalg.norm(light_dir))
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec = light_dir + view
    half_vec = half_vec / max(1e-6, np.linalg.norm(half_vec))

    diffuse = np.clip(np.sum(normals_vec * light_dir, axis=-1), 0.0, 1.0)
    spec = np.clip(np.sum(normals_vec * half_vec, axis=-1), 0.0, 1.0) ** 16
    shading = 1.0 + params.normal_intensity * (
        params.normal_diffuse * diffuse + params.normal_specular * spec
    )
    return np.clip(image * shading[..., None], 0.0, 1.0)


def apply_noise_reduction(image: np.ndarray, params: EditParams) -> np.ndarray:
    # Placeholder for a heavier NR stage; keep CPU cheap for now.
    return image


def apply_sharpening(image: np.ndarray, params: EditParams, allow_cuda: bool = True) -> np.ndarray:
    accel = _apply_sharpening_accel(image, allow_cuda=allow_cuda)
    if accel is not None:
        return accel

    srgb = to_srgb(np.clip(image, 0.0, 1.0))
    u8 = (srgb * 255.0).round().astype(np.uint8)
    pil_img = Image.fromarray(u8, mode="RGB")
    sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=2))
    sharpened_arr = np.asarray(sharpened, dtype=np.float32) / 255.0
    return to_linear(sharpened_arr)


@njit(cache=True, fastmath=True)
def _njit_exposure_contrast(rgb: np.ndarray, exposure: float, contrast: float) -> np.ndarray:
    h, w, c = rgb.shape
    out = np.empty_like(rgb)
    exp_mul = 2.0 ** exposure
    for y in range(h):
        for x in range(w):
            for k in range(c):
                val = rgb[y, x, k] * exp_mul
                val = (val - 0.5) * contrast + 0.5
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
                out[y, x, k] = val
    return out


@njit(cache=True, fastmath=True)
def _njit_white_balance(rgb: np.ndarray, r_gain: float, g_gain: float, b_gain: float) -> np.ndarray:
    h, w, _ = rgb.shape
    out = np.empty_like(rgb)
    for y in range(h):
        for x in range(w):
            r = rgb[y, x, 0] * r_gain
            g = rgb[y, x, 1] * g_gain
            b = rgb[y, x, 2] * b_gain
            if r < 0.0:
                r = 0.0
            if g < 0.0:
                g = 0.0
            if b < 0.0:
                b = 0.0
            if r > 1.0:
                r = 1.0
            if g > 1.0:
                g = 1.0
            if b > 1.0:
                b = 1.0
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b
    return out


@njit(cache=True, fastmath=True)
def _njit_lut_curve(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    h, w, c = rgb.shape
    out = np.empty_like(rgb)
    for y in range(h):
        for x in range(w):
            for k in range(c):
                val = rgb[y, x, k]
                idx = int(val * 255.0)
                if idx < 0:
                    idx = 0
                elif idx > 255:
                    idx = 255
                out[y, x, k] = lut[idx]
    return out


@cuda.jit
def _cuda_box_blur(src, dst, ksize):
    y, x = cuda.grid(2)
    if y >= src.shape[0] or x >= src.shape[1]:
        return
    half = ksize // 2
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    count = 0
    for dy in range(-half, half + 1):
        yy = y + dy
        if yy < 0 or yy >= src.shape[0]:
            continue
        for dx in range(-half, half + 1):
            xx = x + dx
            if xx < 0 or xx >= src.shape[1]:
                continue
            acc0 += src[yy, xx, 0]
            acc1 += src[yy, xx, 1]
            acc2 += src[yy, xx, 2]
            count += 1
    if count > 0:
        dst[y, x, 0] = acc0 / count
        dst[y, x, 1] = acc1 / count
        dst[y, x, 2] = acc2 / count


def _apply_affine_pillow(img: np.ndarray, rotate_deg: float, scale_x: float, scale_y: float, offset_x: float, offset_y: float) -> np.ndarray:
    """Lightweight affine using Pillow; expects linear float 0..1 array."""
    if img is None:
        return img
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    srgb = to_srgb(np.clip(img, 0.0, 1.0))
    u8 = (srgb * 255.0).round().astype(np.uint8)
    pil_img = Image.fromarray(u8, mode="RGB")

    # Scale
    if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
        new_w = max(1, int(round(w * scale_x)))
        new_h = max(1, int(round(h * scale_y)))
        pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        ox = int(round((w - new_w) / 2))
        oy = int(round((h - new_h) / 2))
        canvas.paste(pil_img, (ox, oy))
        pil_img = canvas

    # Offset (normalized -1..1 to pixels)
    dx = int(round(offset_x * w * 0.5))
    dy = int(round(offset_y * h * 0.5))
    if dx or dy:
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(pil_img, (dx, dy))
        pil_img = canvas

    # Rotation around center
    if abs(rotate_deg) > 1e-3:
        pil_img = pil_img.rotate(rotate_deg, resample=Image.BICUBIC, expand=False, center=(w / 2, h / 2), fillcolor=(0, 0, 0))

    out = np.asarray(pil_img, dtype=np.float32) / 255.0
    return to_linear(out)


def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = (
        Ia * wa[..., None]
        + Ib * wb[..., None]
        + Ic * wc[..., None]
        + Id * wd[..., None]
    )
    return out


def apply_lens_profile(image: np.ndarray, geom: GeometryParams) -> np.ndarray:
    if image is None:
        return image
    amount = max(0.0, min(1.0, float(getattr(geom, "lens_profile_amount", 0.0) or 0.0)))
    if not getattr(geom, "lens_profile_enabled", False) or amount <= 1e-4:
        return image

    img = np.ascontiguousarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)

    # Radial distortion correction (k1)
    k1 = -0.15 * amount
    scale = 1.0 + k1 * (r ** 2)
    x_corr = xx * scale
    y_corr = yy * scale

    # Map back to pixel coords
    x_pix = (x_corr + 1.0) * 0.5 * (w - 1)
    y_pix = (y_corr + 1.0) * 0.5 * (h - 1)
    x_pix = np.clip(x_pix, 0, w - 1)
    y_pix = np.clip(y_pix, 0, h - 1)

    corrected = _bilinear_sample(img, x_pix, y_pix)

    # Vignette correction
    r_norm = np.clip(r / np.sqrt(2.0), 0.0, 1.0)
    vig_strength = 0.4 * amount
    gain = 1.0 + vig_strength * (1.0 - (r_norm ** 2))
    corrected = np.clip(corrected * gain[..., None], 0.0, 1.0)

    # Blend with original
    out = img * (1.0 - amount) + corrected * amount
    return np.clip(out, 0.0, 1.0)


def _apply_crop(img: np.ndarray, geom: GeometryParams) -> np.ndarray:
    if img is None or not geom.crop_enabled:
        return img
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img
    x0 = int(round(geom.crop_x * w))
    y0 = int(round(geom.crop_y * h))
    cw = int(round(geom.crop_w * w))
    ch = int(round(geom.crop_h * h))
    x1 = max(0, min(w, x0 + cw))
    y1 = max(0, min(h, y0 + ch))
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1, ...]


def apply_geometry(image: np.ndarray, params: EditParams, quality: str, allow_cuda: bool = True) -> np.ndarray:
    """Apply geometry before tone/color; lightweight approximation to keep pipeline stable."""
    geom = params.geometry or GeometryParams()
    # If geometry is effectively neutral, skip to avoid tile artifacts.
    if _geometry_is_identity(geom):
        return image

    img = image

    # Lens profile first
    img = apply_lens_profile(img, geom)

    # Flips
    if geom.flip_horizontal and img is not None:
        img = np.ascontiguousarray(img[:, ::-1, :])
    if geom.flip_vertical and img is not None:
        img = np.ascontiguousarray(img[::-1, :, :])

    # Affine (rotate/scale/offset)
    if img is not None:
        img = _apply_affine_pillow(
            img,
            geom.rotate_deg,
            geom.scale_x if abs(geom.scale_uniform - 1.0) < 1e-3 else geom.scale_uniform,
            geom.scale_y if abs(geom.scale_uniform - 1.0) < 1e-3 else geom.scale_uniform,
            geom.offset_x,
            geom.offset_y,
        )

    # TODO: Perspective / Upright / Keystone / Lens / Warp (placeholder no-ops for now)

    # Avoid cropping for now to keep tile sizes stable
    return img


def _geometry_is_identity(geom) -> bool:
    if geom is None:
        return True
    if isinstance(geom, dict):
        geom = _dict_to_geometry_params(geom)
    return (
        abs(geom.rotate_deg) < 1e-3
        and abs(geom.scale_uniform - 1.0) < 1e-3
        and abs(geom.scale_x - 1.0) < 1e-3
        and abs(geom.scale_y - 1.0) < 1e-3
        and abs(geom.offset_x) < 1e-3
        and abs(geom.offset_y) < 1e-3
        and not geom.flip_horizontal
        and not geom.flip_vertical
        and abs(geom.vertical_persp) < 1e-3
        and abs(geom.horizontal_persp) < 1e-3
        and abs(geom.rotate_x_deg) < 1e-3
        and abs(geom.rotate_y_deg) < 1e-3
        and abs(geom.distortion) < 1e-3
        and abs(geom.vignette_amount) < 1e-3
        and abs(geom.defish_amount) < 1e-3
        and abs(geom.keystone_top) < 1e-3
        and abs(geom.keystone_bottom) < 1e-3
        and abs(geom.volume_deform) < 1e-3
        and not geom.mesh_enabled
        and not geom.crop_enabled
    )


def _apply_geometry_opencv(img: np.ndarray, geom: GeometryParams) -> np.ndarray:
    """Fast affine using OpenCV to avoid per-pixel loops."""
    if img is None:
        return img
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    center = (w * 0.5, h * 0.5)
    scale = geom.scale_uniform if abs(geom.scale_uniform - 1.0) > 1e-3 else 1.0
    M = cv2.getRotationMatrix2D(center, geom.rotate_deg, scale)
    M[0, 2] += geom.offset_x * w * 0.5
    M[1, 2] += geom.offset_y * h * 0.5
    flags = cv2.INTER_LINEAR
    img = cv2.warpAffine(img, M, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)
    if geom.flip_horizontal:
        img = cv2.flip(img, 1)
    if geom.flip_vertical:
        img = cv2.flip(img, 0)
    return img


def apply_filmic_tonemap(img_lin: np.ndarray) -> np.ndarray:
    """Placeholder kept for backward compatibility; replaced by Hable curve below."""
    return _hable_tonemap(np.maximum(img_lin, 0.0))


def _hable_tonemap(x: np.ndarray) -> np.ndarray:
    """
    Uncharted 2 / Hable filmic curve, applied in linear space.
    x: linear RGB, float32, can be >1.0
    returns: tonemapped RGB (still linear), not yet sRGB encoded.
    """
    A = 0.15
    B = 0.50
    C = 0.10
    D = 0.20
    E = 0.02
    F = 0.30
    W = 11.2  # white point

    x2 = x * x  # noqa: F841 - kept for clarity with supplied reference
    numerator = x * (A * x + C * B) + D * E
    denominator = x * (A * x + B) + D * F
    y = numerator / (denominator + 1e-8) - E / F

    xw = W
    numerator_w = xw * (A * xw + C * B) + D * E
    denominator_w = xw * (A * xw + B) + D * F
    y_w = numerator_w / (denominator_w + 1e-8) - E / F
    white_scale = 1.0 / max(y_w, 1e-6)

    y *= white_scale
    return y


def apply_filmic_tonemap(img_lin: np.ndarray, exposure_bias: float = 0.0) -> np.ndarray:
    """
    Full filmic output transform in linear space.

    img_lin: float32 linear sRGB, [0, +∞)
    exposure_bias: additional EV offset (0 = neutral, +1 = +1 stop, etc.)
    """
    if exposure_bias != 0.0:
        img_lin = img_lin * (2.0 ** exposure_bias)

    filmic = _hable_tonemap(np.maximum(img_lin, 0.0))
    filmic = np.clip(filmic, 0.0, 1.0)
    return filmic


def apply_output_transform(img_lin: np.ndarray, mode: OutputTransformMode, view_exposure_bias: float = 0.0) -> np.ndarray:
    if mode == OutputTransformMode.FILMIC:
        return apply_filmic_tonemap(img_lin, exposure_bias=view_exposure_bias)
    return np.clip(img_lin, 0.0, 1.0)


def apply_pipeline_cpu(image: np.ndarray, params: dict | EditParams, quality: str = "final") -> np.ndarray:
    if image is None:
        return None
    edit_params = params if isinstance(params, EditParams) else _dict_to_edit_params(params or {})
    geom = edit_params.geometry or GeometryParams()
    rgb = np.asarray(image, dtype=np.float32).copy()
    rgb = np.clip(rgb, 0.0, 1.0)

    # White balance
    rgb = _njit_white_balance(rgb, 1.0 + edit_params.temperature, 1.0, 1.0 + edit_params.tint)

    # Exposure first (pure linear gain), then contrast
    rgb = _njit_exposure_contrast(rgb, edit_params.exposure, 1.0)
    rgb = _njit_exposure_contrast(rgb, 0.0, edit_params.contrast)

    # Tone ops (placeholder LUT)
    lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    rgb = _njit_lut_curve(rgb, lut)

    # Levels / curves / tone mapping
    rgb = apply_tone_curve(rgb, edit_params, quality)

    # Color pipeline
    rgb = apply_channel_mixer(rgb, edit_params, quality)
    rgb = apply_gradient_map(rgb, edit_params, quality)
    rgb = apply_split_toning(rgb, edit_params, quality)
    rgb = apply_color_adjustments(rgb, edit_params, quality)
    rgb = apply_normal_relighting(rgb, edit_params, quality)

    # Optics / geometry
    rgb = apply_lens_profile(rgb, geom)
    rgb = _apply_geometry_opencv(rgb, geom)

    # Detail
    if quality == "final":
        rgb = apply_noise_reduction(rgb, edit_params)
        rgb = apply_sharpening(rgb, edit_params, allow_cuda=False)

    # Output transform (last nonlinear step before sRGB encoding)
    rgb = apply_output_transform(rgb, edit_params.output_transform, edit_params.view_exposure_bias)
    return np.clip(rgb, 0.0, 1.0)


def apply_pipeline_cuda(image: np.ndarray, params: dict | EditParams, quality: str = "final") -> np.ndarray:
    """GPU-heavy path: upload once, run heavy blur, then finish on CPU."""
    if image is None:
        return None
    edit_params = params if isinstance(params, EditParams) else _dict_to_edit_params(params or {})
    rgb = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    d_in = cuda.to_device(rgb)
    d_out = cuda.device_array_like(d_in)
    threads = (16, 16)
    blocks = ((rgb.shape[1] + threads[0] - 1) // threads[0],
              (rgb.shape[0] + threads[1] - 1) // threads[1])
    _cuda_box_blur[blocks, threads](d_in, d_out, 9)
    rgb_blurred = d_out.copy_to_host()
    # Finish with CPU pipeline for color/tone
    return apply_pipeline_cpu(rgb_blurred, edit_params, quality)


def apply_pipeline(image: np.ndarray, params: dict | EditParams, quality: str = "final") -> np.ndarray:
    use_cuda = False
    if isinstance(params, EditParams):
        use_cuda = getattr(params, "use_cuda", False)
    elif isinstance(params, dict):
        use_cuda = params.get("use_cuda", False)
    if use_cuda and cuda.is_available():
        return apply_pipeline_cuda(image, params, quality)
    return apply_pipeline_cpu(image, params, quality)


def _apply_basic_adjustments_accel(image: np.ndarray, params: EditParams, allow_cuda: bool = True) -> np.ndarray | None:
    """Choose the fastest available backend (CUDA > numba CPU) for the heavy basic stage."""
    img = _f32(image)
    if allow_cuda and _USE_CUDA_BASIC:
        try:
            h, w, _ = img.shape
            # Pad tiny tiles so the CUDA grid has enough work to occupy SMs.
            min_dim = 256
            pad_h = max(0, min_dim - h)
            pad_w = max(0, min_dim - w)
            if pad_h > 0 or pad_w > 0:
                img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            else:
                img_padded = img

            ph, pw, _ = img_padded.shape
            threads = (16, 16)
            blocks = ((pw + threads[0] - 1) // threads[0], (ph + threads[1] - 1) // threads[1])
            # For tiny previews the grid can under-utilize the GPU; step down block size aggressively to spawn more blocks.
            if blocks[0] * blocks[1] < 256:
                threads = (8, 8)
                blocks = ((pw + threads[0] - 1) // threads[0], (ph + threads[1] - 1) // threads[1])
            d_in = cuda.to_device(img_padded)
            d_out = cuda.device_array_like(img_padded)
            _basic_cuda_kernel[blocks, threads](
                d_in, d_out,
                params.temperature, params.tint,
                params.exposure, params.contrast,
                params.highlights, params.shadows, params.whites, params.blacks,
                params.saturation,
            )
            out = d_out.copy_to_host()
            if pad_h > 0 or pad_w > 0:
                out = out[:h, :w, :]
            return out
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print("[Accel] CUDA basic stage failed, falling back to CPU:", exc)

    if _USE_NUMBA_BASIC:
        try:
            return _basic_numba_kernel(
                img,
                params.temperature, params.tint,
                params.exposure, params.contrast,
                params.highlights, params.shadows, params.whites, params.blacks,
                params.saturation,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print("[Accel] Numba basic stage failed, falling back to NumPy:", exc)

    return None


def _apply_sharpening_accel(image: np.ndarray, allow_cuda: bool = True) -> np.ndarray | None:
    """Sharpen via CUDA or numba if enabled; otherwise fall back to PIL."""
    img = _f32(np.clip(image, 0.0, 1.0))
    amount = 0.8  # approximates 80% unsharp mask strength

    if allow_cuda and _USE_CUDA_SHARPEN:
        try:
            h, w, _ = img.shape
            min_dim = 256
            pad_h = max(0, min_dim - h)
            pad_w = max(0, min_dim - w)
            if pad_h > 0 or pad_w > 0:
                img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            else:
                img_padded = img
            ph, pw, _ = img_padded.shape
            threads = (16, 16)
            blocks = ((pw + threads[0] - 1) // threads[0], (ph + threads[1] - 1) // threads[1])
            d_in = cuda.to_device(img_padded)
            d_out = cuda.device_array_like(img_padded)
            _sharpen_cuda_kernel[blocks, threads](d_in, d_out, amount)
            out = d_out.copy_to_host()
            if pad_h > 0 or pad_w > 0:
                out = out[:h, :w, :]
            return out
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print("[Accel] CUDA sharpening failed, falling back:", exc)

    if _USE_NUMBA_SHARPEN:
        try:
            return _sharpen_numba_kernel(img, amount)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print("[Accel] Numba sharpening failed, falling back:", exc)

    return None


class _FullRenderSignals(QObject):
    finished = Signal(np.ndarray, dict, int)


class _FullRenderWorker(QRunnable):
    """Background full-resolution renderer."""

    def __init__(self, image: np.ndarray, params: dict, token: int):
        super().__init__()
        self._image = image
        self._params = params
        self._token = token
        self.signals = _FullRenderSignals()

    def run(self):
        try:
            result = apply_pipeline(self._image, self._params, quality="final")
            if (
                result is None
                or not np.isfinite(result).all()
                or result.max() < 1e-5
            ):
                result = apply_pipeline_cpu(self._image, self._params, quality="final")
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print("[FullRenderWorker] render error:", exc)
            return
        try:
            self.signals.finished.emit(result, self._params, self._token)
        except RuntimeError:
            # Likely app shutting down; safely ignore to avoid noisy traceback.
            return


class _TileRenderSignals(QObject):
    finished = Signal(object, np.ndarray, int)


def _extract_tile(level_image: np.ndarray, tile_key: TileKey, tile_size: int) -> np.ndarray:
    y0 = tile_key.ty * tile_size
    x0 = tile_key.tx * tile_size
    return level_image[y0:y0 + tile_size, x0:x0 + tile_size].copy()


class TileRenderTask(QRunnable):
    """Background tile renderer that applies the full-quality pipeline on a mip tile."""

    def __init__(
        self,
        tile_key: TileKey,
        base_level: np.ndarray,
        params: EditParams,
        version: int,
        tile_size: int,
        quality: str = "final",
    ):
        super().__init__()
        self.tile_key = tile_key
        self.base_level = base_level
        self.params = params
        self.version = version
        self.tile_size = tile_size
        self.quality = quality
        self.signals = _TileRenderSignals()

    def run(self):
        try:
            tile = _extract_tile(self.base_level, self.tile_key, self.tile_size)
            result = apply_pipeline(tile, self.params, quality=self.quality)
            if (
                result is None
                or not np.isfinite(result).all()
                or result.max() < 1e-5
            ):
                result = apply_pipeline_cpu(tile, self.params, quality=self.quality)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print("[TileRenderTask] render error:", exc)
            return
        try:
            self.signals.finished.emit(self.tile_key, result, self.version)
        except RuntimeError:
            return


class TitleBar(QWidget):
    """Custom top bar with logo, title, menubar, and window buttons."""

    def __init__(self, parent=None, icon_pix: QPixmap | None = None, menu_bar: QMenuBar | None = None):
        super().__init__(parent)
        self._mouse_pos = None
        self._menu_bar = menu_bar

        self.setObjectName("TitleBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(10)

        self.icon_label = QLabel(self)
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setScaledContents(True)
        if icon_pix and not icon_pix.isNull():
            self.icon_label.setPixmap(icon_pix.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.title_label = QLabel("Gradience Studio", self)
        self.title_label.setObjectName("TitleBarTitle")
        self.title_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)

        if self._menu_bar is not None:
            self._menu_bar.setParent(self)
            self._menu_bar.setNativeMenuBar(False)
            layout.addWidget(self._menu_bar, 1)

        layout.addStretch(1)

        self.min_button = QPushButton("–", self)
        self.max_button = QPushButton("□", self)
        self.close_button = QPushButton("✕", self)

        for btn in (self.min_button, self.max_button, self.close_button):
            btn.setObjectName("TitleBarButton")
            btn.setFixedSize(30, 22)

        layout.addWidget(self.min_button)
        layout.addWidget(self.max_button)
        layout.addWidget(self.close_button)

        self.min_button.clicked.connect(self._on_minimize)
        self.max_button.clicked.connect(self._on_maximize_restore)
        self.close_button.clicked.connect(self._on_close)

    # Drag/move support
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._mouse_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._mouse_pos is not None and event.buttons() & Qt.LeftButton:
            delta = event.globalPosition().toPoint() - self._mouse_pos
            window = self.window()
            window.move(window.pos() + delta)
            self._mouse_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._mouse_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._on_maximize_restore()
        super().mouseDoubleClickEvent(event)

    def _on_minimize(self):
        self.window().showMinimized()

    def _on_maximize_restore(self):
        w = self.window()
        if w.isMaximized():
            w.showNormal()
        else:
            w.showMaximized()

    def _on_close(self):
        self.window().close()


class DebugOverlay(QWidget):
    """Lightweight overlay that graphs frame times and render stats."""

    def __init__(self, parent=None, history: int = 240):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self._frame_ms_history: deque[float] = deque(maxlen=history)
        self._fps_history: deque[float] = deque(maxlen=history)
        self._last_fps: float = 0.0
        self._last_ms: float = 0.0
        self._meta: dict = {}
        self.setFixedSize(260, 150)

    def update_metrics(self, fps: float, frame_ms: float, meta: dict | None = None):
        self._last_fps = fps
        self._last_ms = frame_ms
        self._frame_ms_history.append(frame_ms if frame_ms >= 0 else 0.0)
        self._fps_history.append(max(0.0, fps))
        self._meta = meta or {}
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        painter.fillRect(rect, QColor(12, 12, 12, 210))

        margin = 10
        text_color = QColor(220, 230, 240)
        accent = QColor(80, 170, 255)
        warn = QColor(255, 140, 80)

        fps_avg = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0.0
        frame_p95 = 0.0
        if self._frame_ms_history:
            frame_p95 = float(np.percentile(list(self._frame_ms_history), 95))

        lines = [
            f"FPS: {self._last_fps:5.1f} (avg {fps_avg:4.1f})",
            f"Frame: {self._last_ms:5.1f} ms (p95 {frame_p95:4.1f} ms)",
        ]
        if self._meta:
            tiles = f"{self._meta.get('tiles_inflight', 0)}/{self._meta.get('tile_cache', 0)}"
            zoom_desc = self._meta.get("zoom_desc", "")
            level = self._meta.get("preview_level", 0)
            threads = self._meta.get("render_threads", 0)
            lines.append(f"Tiles inflight/cache: {tiles} | threads: {threads}")
            lines.append(f"Preview L{level} @ {zoom_desc}")

        painter.setPen(text_color)
        y = margin + painter.fontMetrics().ascent()
        for line in lines:
            painter.drawText(margin, y, line)
            y += painter.fontMetrics().height() + 2

        graph_top = y + 6
        graph_rect = QRect(
            margin,
            graph_top,
            max(10, rect.width() - 2 * margin),
            max(12, rect.height() - graph_top - margin),
        )

        # Graph background
        painter.fillRect(graph_rect, QColor(30, 30, 30, 190))
        painter.setPen(QColor(70, 70, 70))
        painter.drawRect(graph_rect)

        if not self._frame_ms_history:
            return

        max_ms = max(16.0, max(self._frame_ms_history))
        graph_bottom = graph_rect.bottom()
        graph_height = graph_rect.height()
        graph_left = graph_rect.left()
        graph_right = graph_rect.right()

        for ref_ms, color in [(16.6, accent), (33.3, warn)]:
            y_ref = graph_bottom - int(min(1.0, ref_ms / max_ms) * graph_height)
            painter.setPen(QColor(color.red(), color.green(), color.blue(), 120))
            painter.drawLine(graph_left, y_ref, graph_right, y_ref)
            painter.drawText(graph_left + 2, y_ref - 2, f"{ref_ms:.0f} ms")

        painter.setPen(accent)
        values = list(self._frame_ms_history)
        n = len(values)
        prev_x = prev_y = None
        for idx, ms in enumerate(values):
            if n == 1:
                x = graph_left
            else:
                x = graph_left + int(idx * (graph_rect.width()) / max(1, n - 1))
            y_val = graph_bottom - int(min(1.0, ms / max_ms) * graph_height)
            y_val = max(graph_rect.top(), min(graph_rect.bottom(), y_val))
            if prev_x is not None:
                painter.drawLine(prev_x, prev_y, x, y_val)
            prev_x, prev_y = x, y_val


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---------- Window basics ----------
        self.setWindowTitle("Gradience Studio")
        self.setWindowIcon(QIcon("icon.ico"))
        self.resize(1200, 800)
        self.setMinimumSize(600, 400)
        self._adjust_popups: dict[str, QDialog] = {}

        central = QWidget(self)
        self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central)
        self.vbox.setContentsMargins(8, 8, 8, 8)
        self.vbox.setSpacing(8)

        icon_pix = QPixmap("icon.ico")
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)
        self.menu_bar.setObjectName("MainMenuBar")

        self.header_filename_label = QLabel("No file loaded")
        self.header_filename_label.setObjectName("headerFilename")
        self.header_filename_label.setAlignment(Qt.AlignCenter)

        # Body layout (existing UI)
        self.body_container = QWidget()
        self.body_layout = QVBoxLayout(self.body_container)
        self.body_layout.setContentsMargins(8, 8, 8, 8)
        self.body_layout.setSpacing(8)


        # ---------- State ----------
        self._current_folder: str | None = None
        self._current_path: str | None = None
        self._current_pixmap: QPixmap | None = None
        self._preview_base_linear: np.ndarray | None = None
        self._base_image: np.ndarray | None = None
        self._mipmaps: list[np.ndarray] = []
        self._mip_scales: list[float] = []
        self._render_version = 0
        self._full_render_token = 0
        self._shutting_down = False
        self._force_smallest_preview = False
        self._drag_preview_mode = False
        self._drag_render_timer_ms = 20
        self._tile_size = 256
        self._thumb_icon_px = 80
        self._tile_cache: dict[TileKey, np.ndarray] = {}
        self._tile_tasks_inflight: set[TileKey] = set()
        self._display_qimage: QImage | None = None
        self._preview_qimage: QImage | None = None
        self._display_scale: float = 1.0
        self._preview_level = 0

        self._image_cache: dict[str, dict] = {}   # path -> {"linear_full": np.ndarray, "preview_source": np.ndarray, "preview_linear": np.ndarray}
        self._image_params: dict[str, dict] = {}  # path -> params
        self._image_parents: dict[str, QTreeWidgetItem] = {}
        self._edits: dict[str, dict] = {}         # for project export
        self._metadata_labels: dict[str, QLabel] = {}

        self._project_path: str | None = None
        self._project_root: str | None = None

        # performance caps
        self._max_thumb_inflight = 32

        # Debug / perf overlay
        self._debug_overlay_enabled = False
        self._debug_overlay: DebugOverlay | None = None
        self._last_frame_present_time: float | None = None
        self._last_paint_time: float | None = None
        self._debug_start_time: float | None = None
        self._perf_log: list[tuple[float, float, float]] = []  # (t, fps, ms)
        self._perf_log_max = 12000
        self._debug_log_path: str | None = None

        # fullscreen state defaults
        self._is_fullscreen_mode = False
        self._fs_prev_geometry = None
        self._fs_prev_menubar_visible = True
        self._fs_prev_statusbar_visible = True
        self._fs_prev_left_visible = True
        self._fs_prev_right_visible = True
        self._fs_prev_filmstrip_visible = True

        # preview scale used by thumbnail/render pipeline; set early to avoid attribute errors
        self._preview_scale = 1.0
        # zoom/pan defaults (set early so render paths have them)
        self._zoom_mode = "fit"
        self._zoom_factor = 1.0
        self._dragging = False
        self._drag_last_pos = QPoint()

        # current parameters (Light + Color tabs)
        self._color_defaults = default_color_params()
        self._geometry_defaults = copy.deepcopy(_dict_to_geometry_params(None).__dict__)
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
            # Levels defaults
            "levels_black": 0,
            "levels_white": 100,
            "levels_gamma": 1.0,
            "levels_out_black": 0,
            "levels_out_white": 100,
            "levels_color_model": "RGB",
            "levels_channel": "Master",
            "levels_linear": False,
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
            "normal_map": None,
            "normal_light_x": 0,
            "normal_light_y": 0,
            "normal_light_elev": 50,
            "normal_intensity": 100,
            "normal_specular": 0,
            "normal_diffuse": 100,
            "view_exposure_bias": 0.0,
            "output_transform": OutputTransformMode.STANDARD.value,
            "geometry": copy.deepcopy(self._geometry_defaults),
        }
        for _color in ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]:
            self._current_params.update({
                f"hsl_hue_{_color}": 0,
                f"hsl_sat_{_color}": 0,
                f"hsl_lum_{_color}": 0,
            })
        self._current_params["color"] = copy.deepcopy(self._color_defaults)
        self._default_params_template = copy.deepcopy(self._current_params)
        self._active_edit: dict = {"effects": []}
        self._active_edits_by_path: dict[str, dict] = {}

        # autosave
        self._autosave_interval_min = 5
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._update_autosave_timer()

        # Debounced render timer (must exist before panels emit renders)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_current_params)

        # Idle timer to upgrade preview mip after user stops dragging sliders
        self._idle_mip_timer = QTimer(self)
        self._idle_mip_timer.setSingleShot(True)
        self._idle_mip_timer.timeout.connect(self._on_idle_preview_upgrade)

        # High-quality render debounce
        self._full_render_timer = QTimer(self)
        self._full_render_timer.setSingleShot(True)
        self._full_render_timer.timeout.connect(self._start_full_render_worker)

        # Progressive tile render debounce
        self._tile_render_timer = QTimer(self)
        self._tile_render_timer.setSingleShot(True)
        self._tile_render_timer.timeout.connect(self._start_background_tiles)

        # Debug heartbeat so overlay keeps updating even when no new frames are painted
        self._debug_timer = QTimer(self)
        self._debug_timer.setInterval(16)  # ~60 Hz virtual frame updates while idle
        self._debug_timer.timeout.connect(self._on_debug_heartbeat)

        # Render caches (reset per image)
        self._reset_render_cache()

        # ---------- Central image area ----------
                # ---------- Central image area + zoom toolbar ----------
        self.image_display = QLabel("No folder open. File → Import Folder")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(200, 200)
        self.image_display.setObjectName("imageDisplay")

        # Scroll area to allow panning
        self.image_scroll = QScrollArea()
        self.image_scroll.setObjectName("imageScroll")
        self.image_scroll.setFrameShape(QFrame.NoFrame)
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setWidget(self.image_display)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.horizontalScrollBar().valueChanged.connect(lambda _: self._schedule_background_tiles(0))
        self.image_scroll.verticalScrollBar().valueChanged.connect(lambda _: self._schedule_background_tiles(0))

        # Debug overlay lives on top of the viewport so it scrolls with the image area
        self._debug_overlay = DebugOverlay(self.image_scroll.viewport())
        self._debug_overlay.hide()

        # Preview/zoom toolbar
        self.image_toolbar = QWidget()
        self.image_toolbar.setObjectName("imageToolBar")
        tb = QHBoxLayout(self.image_toolbar)
        tb.setContentsMargins(8, 4, 8, 4)
        tb.setSpacing(8)

        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_100_btn = QPushButton("100%")
        self.zoom_out_btn = QPushButton("−")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)

        # NEW: preview resolution combo
        self.preview_res_combo = QComboBox()
        self.preview_res_combo.setMinimumWidth(130)
        self.preview_res_combo.addItem("Preview: Full", 1.0)
        self.preview_res_combo.addItem("Preview: 3/4", 0.75)
        self.preview_res_combo.addItem("Preview: 1/2", 0.5)
        self.preview_res_combo.addItem("Preview: 1/4", 0.25)
        self.preview_res_combo.addItem("Preview: 1/8", 0.125)

        tb.addWidget(self.preview_res_combo)
        tb.addSpacing(6)
        tb.addWidget(self.zoom_fit_btn)
        tb.addWidget(self.zoom_100_btn)
        tb.addWidget(self.zoom_out_btn)
        tb.addWidget(self.zoom_label)
        tb.addWidget(self.zoom_in_btn)
        tb.addStretch(1)

        filename_row = QHBoxLayout()
        filename_row.setContentsMargins(12, 6, 12, 6)
        filename_row.addWidget(self.header_filename_label, 0, Qt.AlignCenter)

        self.body_layout.addWidget(self.image_toolbar, 0)
        self.body_layout.addWidget(self.image_scroll, 1)
        self.body_layout.addLayout(filename_row, 0)

        # zoom button signals
        self.zoom_fit_btn.clicked.connect(self._on_zoom_fit)
        self.zoom_100_btn.clicked.connect(self._on_zoom_100)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)

        # NEW: preview resolution change
        self.preview_res_combo.currentIndexChanged.connect(self._on_preview_res_changed)

        # drag-to-pan on the image
        self.image_display.installEventFilter(self)


        # ---------- Bottom filmstrip ----------
        self.thumbs = QListWidget()
        self.thumbs.setObjectName("bottomFilmstrip")
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(self._thumb_icon_px, self._thumb_icon_px))
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

        self.body_layout.addWidget(self.thumbs, 0)

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
        # Separate pool for full-resolution renders so thumbnails stay responsive
        self._render_pool = QThreadPool(self)
        self._render_pool.setMaxThreadCount(max(2, min(6, os.cpu_count() or 4)))
        self._icon_cache: dict[str, QIcon] = {}
        self._loading: set[str] = set()
        self._item_for_path: dict[str, QListWidgetItem] = {}

        # placeholder icon
        self._placeholder_icon = QIcon(QPixmap(icon_w, icon_h))

        # ---------- Menus ----------
        self._build_menus()

        # ---------- Right dock: adjustments & metadata ----------
        self.hist_widget = HistogramWidget(self)
        self.hist_widget.setObjectName("histogramPanel")

        self.right_dock = QDockWidget("Adjustments", self)
        self.right_dock.setObjectName("rightSidebar")
        self.right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.right_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.right_dock.setMinimumWidth(320)
        self.hist_dock = QDockWidget("Histogram", self)
        self.hist_dock.setObjectName("histSidebar")
        self.hist_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.hist_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.hist_dock.setMinimumWidth(240)
        self.hist_dock.setMinimumHeight(140)

    # NEW unified adjustments panel
        self.adjust_panel = AdjustmentsPanel(self)
        # Compact typography and padding within the right dock to reduce overflow risk.
        self.adjust_panel.setStyleSheet(
            "#adjustmentsRoot { font-size: 12px; }"
            "#adjustmentsRoot QLabel { min-width: 0px; }"
        )
        self.right_dock.setWidget(self.adjust_panel)
        self.hist_dock.setWidget(self.hist_widget)


        # ---------- Left dock: active edit + folders ----------
        self.left_dock = QDockWidget("Library", self)
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

        # Use the native dock title bar (blank) to retain movability
        self.left_dock.setWindowTitle("")
        self.left_dock.setWidget(self.left_tabs)
        self.left_dock.setMinimumWidth(260)

        # history context menu
        self.history_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_tree.customContextMenuRequested.connect(
            self._on_history_context_menu
        )

        # ---------- Add docks ----------
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.hist_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        try:
            # Stack histogram above adjustments on the right
            self.splitDockWidget(self.hist_dock, self.right_dock, Qt.Vertical)
        except Exception:
            pass
        try:
            # Nudge docks to balanced widths and vertical split sizes
            self.resizeDocks([self.left_dock, self.hist_dock, self.right_dock], [260, 300, 320], Qt.Horizontal)
            self.resizeDocks([self.hist_dock, self.right_dock], [180, 520], Qt.Vertical)
        except Exception:
            pass

        # ---------- Window menu: layout & visibility ----------
        self._build_window_menu()
        # Apply theme once UI is built
        self._apply_styles()

        # Final assembly: body container only (native title bar)
        self.vbox.addWidget(self.body_container, 1)

    def show_histogram_dock(self):
        """Toggle the histogram dock visibility without altering layout."""
        if not hasattr(self, "hist_dock") or self.hist_dock is None:
            return

        if self.hist_dock.isVisible():
            self.hist_dock.hide()
            return

        self.hist_dock.show()

        # fullscreen state
        self._is_fullscreen_mode = False
        self._fs_prev_geometry = None
        self._fs_prev_menubar_visible = True
        self._fs_prev_statusbar_visible = True
        self._fs_prev_left_visible = True
        self._fs_prev_right_visible = True
        self._fs_prev_filmstrip_visible = True
        
        # --- zoom / pan state ---
        self._zoom_mode = "fit"   # "fit" or "manual"
        self._zoom_factor = 1.0   # 1.0 = 100%
        self._dragging = False
        self._drag_last_pos = QPoint()

        # ---------- Apply theme ----------
        self._apply_styles()

        # Initialize sliders to neutral positions
        try:
            self.adjust_panel.light_group.reset_ui_to_defaults()
        except Exception:
            pass
        try:
            self.adjust_panel.color_group.reset_ui_to_defaults()
        except Exception:
            pass
        try:
            self.adjust_panel.tone_group.reset_ui_to_defaults()
        except Exception:
            pass

        # Save default layout for restore
        self._default_layout_state = self.saveState()

        self._copied_params: dict | None = None

    # -------- menu builders -------- #

    def _reset_render_cache(self):
        """Clear cached base arrays/masks so they rebuild for the next image."""
        self._base_linear = None
        self._base_lum = None
        self._L_norm = None
        self._tone_masks = {}
        self._preview_base_linear = None
        self._base_image = None
        self._mipmaps = []
        self._mip_scales = []
        self._tile_cache = {}
        self._tile_tasks_inflight = set()
        self._display_qimage = None
        self._preview_qimage = None
        self._render_version += 1
        self._full_render_token += 1
        self._drag_preview_mode = False

    def _update_debug_overlay_geometry(self):
        if not self._debug_overlay:
            return
        parent = self.image_scroll.viewport() if hasattr(self, "image_scroll") else None
        if parent is None:
            return
        margin = 12
        w = self._debug_overlay.width()
        h = self._debug_overlay.height()
        x = max(margin, parent.width() - w - margin)
        y = margin
        self._debug_overlay.setGeometry(x, y, w, h)
        self._debug_overlay.raise_()

    def _debug_overlay_zoom_text(self) -> str:
        zoom_mode = getattr(self, "_zoom_mode", "fit")
        if zoom_mode == "fit":
            pct = int(round(getattr(self, "_display_scale", 1.0) * 100))
            return f"{pct}% (fit)"
        return f"{getattr(self, '_zoom_factor', 1.0):.2f}x"

    def _set_debug_overlay_enabled(self, enabled: bool):
        self._debug_overlay_enabled = bool(enabled)
        if not self._debug_overlay:
            return
        if self._debug_overlay_enabled:
            self._last_frame_present_time = time.perf_counter()
            self._last_paint_time = self._last_frame_present_time
            self._debug_start_time = self._last_frame_present_time
            self._debug_overlay._frame_ms_history.clear()
            self._debug_overlay._fps_history.clear()
            self._perf_log.clear()
            self._debug_log_path = None
            self._update_debug_overlay_geometry()
            self._debug_overlay.show()
            self._debug_overlay.raise_()
            self._debug_timer.start()
        else:
            self._debug_timer.stop()
            self._debug_overlay.hide()

    def _debug_overlay_meta(self) -> dict:
        pool = getattr(self, "_render_pool", None)
        return {
            "tiles_inflight": len(getattr(self, "_tile_tasks_inflight", [])),
            "tile_cache": len(getattr(self, "_tile_cache", {})),
            "render_threads": pool.activeThreadCount() if pool else 0,
            "preview_level": getattr(self, "_preview_level", 0),
            "zoom_desc": self._debug_overlay_zoom_text(),
        }

    def _record_perf_sample(self, fps: float, frame_ms: float):
        if self._debug_start_time is None:
            self._debug_start_time = time.perf_counter()
        t = time.perf_counter() - self._debug_start_time
        self._perf_log.append((t, fps, frame_ms))
        if len(self._perf_log) > self._perf_log_max:
            # Simple decimation to keep memory bounded
            self._perf_log = self._perf_log[len(self._perf_log) - self._perf_log_max :]

    def _save_perf_graph(self):
        if not self._perf_log:
            return
        try:
            w = 1200
            h = 320
            margin = 30
            img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            painter = QPainter(img)
            painter.fillRect(img.rect(), QColor(12, 12, 12, 240))

            accent = QColor(80, 170, 255)
            warn = QColor(255, 140, 80)
            grid = QColor(70, 70, 70, 140)
            text = QColor(230, 230, 230)

            # Axes
            painter.setPen(grid)
            painter.drawRect(margin, margin, w - 2 * margin, h - 2 * margin)

            data = list(self._perf_log)
            times = [s[0] for s in data]
            frame_ms_vals = [s[2] for s in data]
            if not times or not frame_ms_vals:
                painter.end()
                return

            t_min, t_max = min(times), max(times)
            t_span = max(0.001, t_max - t_min)
            max_ms = max(16.0, max(frame_ms_vals), 33.3)

            plot_w = w - 2 * margin
            plot_h = h - 2 * margin
            x0 = margin
            y0 = margin

            # Reference lines
            for ref_ms, color in [(16.6, accent), (33.3, warn)]:
                y = y0 + int((1.0 - min(ref_ms / max_ms, 1.0)) * plot_h)
                painter.setPen(QColor(color.red(), color.green(), color.blue(), 120))
                painter.drawLine(x0, y, x0 + plot_w, y)
                painter.setPen(text)
                painter.drawText(x0 + 4, y - 2, f"{ref_ms:.1f} ms")

            # Downsample to fit width
            stride = max(1, int(len(data) / plot_w))
            painter.setPen(accent)
            prev_pt = None
            for idx in range(0, len(data), stride):
                t, fps, ms = data[idx]
                x = x0 + int(((t - t_min) / t_span) * plot_w)
                y = y0 + int((1.0 - min(ms / max_ms, 1.0)) * plot_h)
                y = max(y0, min(y0 + plot_h, y))
                if prev_pt is not None:
                    painter.drawLine(prev_pt[0], prev_pt[1], x, y)
                prev_pt = (x, y)

            # Labels
            painter.setPen(text)
            painter.drawText(
                margin,
                h - 8,
                f"FPS graph ({len(self._perf_log)} samples, duration {t_span:.1f}s)",
            )

            painter.end()

            tmp_dir = tempfile.gettempdir()
            fname = f"gradience_fps_log_{int(time.time())}.png"
            out_path = os.path.join(tmp_dir, fname)
            img.save(out_path, "PNG")
            self._debug_log_path = out_path
        except Exception:
            self._debug_log_path = None

    def _update_debug_overlay_metrics(self, fps: float, frame_ms: float):
        self._record_perf_sample(fps, frame_ms)
        if not (self._debug_overlay_enabled and self._debug_overlay and self._debug_overlay.isVisible()):
            return
        self._debug_overlay.update_metrics(fps, frame_ms, self._debug_overlay_meta())

    def _mark_frame_presented(self):
        now = time.perf_counter()
        frame_ms = 0.0
        fps = 0.0
        if self._last_paint_time is not None:
            dt = now - self._last_paint_time
            if dt > 0:
                frame_ms = dt * 1000.0
                fps = 1.0 / dt
        self._last_paint_time = now
        self._last_frame_present_time = now
        self._update_debug_overlay_metrics(fps, frame_ms)

    def _on_debug_heartbeat(self):
        if not (self._debug_overlay_enabled and self._debug_overlay and self._debug_overlay.isVisible()):
            return
        interval_ms = float(self._debug_timer.interval())
        if interval_ms <= 0:
            return
        self._last_frame_present_time = time.perf_counter()
        frame_ms = interval_ms
        fps = 1000.0 / interval_ms
        self._update_debug_overlay_metrics(fps, frame_ms)

    def _update_header_filename(self, path: str | None):
        """Update header filename label without altering core logic."""
        if not hasattr(self, "header_filename_label"):
            return
        text = "No file loaded"
        if path:
            try:
                text = os.path.basename(path)
            except Exception:
                text = path
        self.header_filename_label.setText(text)

        # ---------- zoom helpers ----------

    def _upsert_effect(
        self,
        effect_key: str,
        effect_type: str,
        params: dict,
        mask: str = "Global",
        label: str | None = None,
        path: str | None = None,
    ):
        """
        Insert/update an effect node in the active edit stack and trigger render.

        effect_key is a unique string per logical effect, e.g.
        'levels:RGB:Master' or 'contrast:global'.
        """
        target_path = path or self._current_path
        if not target_path:
            return

        effects_doc = self._active_edits_by_path.setdefault(target_path, {"effects": []})
        self._active_edit = effects_doc

        stack = effects_doc.setdefault("effects", [])

        for eff in stack:
            if eff.get("key") == effect_key and eff.get("mask", "Global") == mask:
                eff.setdefault("params", {}).update(params)
                eff["type"] = effect_type
                eff["enabled"] = True
                if label:
                    eff["label"] = label
                break
        else:
            stack.append({
                "key": effect_key,
                "type": effect_type,
                "mask": mask,
                "label": label or effect_key,
                "params": dict(params),
                "enabled": True,
            })

        self._active_edits_by_path[target_path] = effects_doc

        # whatever you already use to trigger a preview re-render
        if hasattr(self, "_schedule_render"):
            self._schedule_render()
        else:
            self._apply_edit_params_to_current_image(self._current_params)

        if target_path == self._current_path and self._current_pixmap is not None:
            self._update_history_for_current_image()

    def _ensure_active_edit_tree(self, path: str, params: dict | None = None) -> dict:
        """
        Guarantee that we have an active edit document for a path.
        If none exists yet, start with an empty effect list (only user changes add effects).
        """
        if not path:
            return {"effects": []}

        if path not in self._active_edits_by_path:
            self._active_edits_by_path[path] = {"effects": []}

        return self._active_edits_by_path[path]

    def _format_effect_label(self, base_label: str, params: dict) -> str:
        """Render a compact label with the primary numeric value, if any."""
        if not params:
            return base_label
        numeric_values = [
            v for v in params.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if len(numeric_values) == 1:
            return f"{base_label} ({numeric_values[0]:g})"
        return base_label

    def _format_effect_tooltip(self, params: dict) -> str:
        """Readable parameter summary for tree tooltips."""
        if not params:
            return ""
        parts = []
        for k, v in params.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.3g}")
            else:
                parts.append(f"{k}: {v}")
        return ", ".join(parts)

    def _remove_effect(self, effect_key: str, mask: str = "Global", path: str | None = None):
        """Remove an effect entry when parameters return to defaults."""
        target_path = path or self._current_path
        if not target_path:
            return

        edit_doc = self._active_edits_by_path.get(target_path)
        if not edit_doc:
            return

        stack = edit_doc.get("effects", [])
        new_stack = [e for e in stack if not (e.get("key") == effect_key and e.get("mask", "Global") == mask)]

        if len(new_stack) != len(stack):
            edit_doc["effects"] = new_stack
            self._active_edits_by_path[target_path] = edit_doc
            if target_path == self._current_path and self._current_pixmap is not None:
                self._update_history_for_current_image()

    def _remove_active_effect(self, path: str, effect_key: str, mask: str = "Global", params: dict | None = None):
        """
        Remove an effect from Active Edit and reset corresponding params to defaults,
        then refresh history/render for the current image.
        """
        if not path:
            return

        edit_doc = self._active_edits_by_path.get(path)
        if edit_doc:
            stack = edit_doc.get("effects", [])
            edit_doc["effects"] = [e for e in stack if not (e.get("key") == effect_key and e.get("mask", "Global") == mask)]
            self._active_edits_by_path[path] = edit_doc

        # Reset parameters tied to this effect
        def reset_param(param_key: str, default_val):
            self._current_params[param_key] = default_val
            if path in self._image_params:
                self._image_params[path][param_key] = default_val

        if effect_key.startswith("levels:"):
            lvl_defaults = {
                "levels_black": 0,
                "levels_white": 100,
                "levels_gamma": 1.0,
                "levels_out_black": 0,
                "levels_out_white": 100,
                "levels_linear": False,
                "levels_color_model": "RGB",
                "levels_channel": "Master",
            }
            if params:
                cm = params.get("levels_color_model", "RGB")
                ch = params.get("levels_channel", "Master")
                lvl_defaults["levels_color_model"] = cm
                lvl_defaults["levels_channel"] = ch
            for k, v in lvl_defaults.items():
                reset_param(k, v)
        else:
            param_key = effect_key.split(":", 1)[0]
            default_val = self._default_params_template.get(param_key, 128)
            reset_param(param_key, default_val)
            if param_key == "contrast":
                reset_param("bc_linear", self._default_params_template.get("bc_linear", False))

        if path == self._current_path:
            # Reapply current parameters to reflect the removal
            self._apply_edit_params_to_current_image(self._current_params)
            self._update_history_for_current_image()


    def _set_zoom(self, factor: float, mode: str = "manual"):
        if self._current_pixmap is None:
            return

        factor = max(0.1, min(8.0, factor))  # clamp 10%–800%
        self._zoom_factor = factor
        self._zoom_mode = mode
        self._update_zoom_label()
        self._rescale_preview()
        self._schedule_background_tiles()

    def _update_zoom_label(self):
        if self._current_pixmap is None:
            self.zoom_label.setText("—")
            return

        if self._base_image is not None:
            ratio = self._display_scale
        else:
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

    def _on_preview_res_changed(self, index: int):
        """
        Called when the user changes the Preview Resolution combo.
        Rebuilds the preview for the current image and re-applies edits.
        """
        ui_state = None
        color_state = None
        if hasattr(self, "adjust_panel"):
            try:
                if hasattr(self.adjust_panel, "light_group"):
                    ui_state = self.adjust_panel.light_group.capture_ui_state()
            except Exception:
                ui_state = None
            try:
                if hasattr(self.adjust_panel, "color_group"):
                    color_state = self.adjust_panel.color_group.capture_ui_state()
            except Exception:
                color_state = None

        data = self.preview_res_combo.currentData()
        if data is None:
            return

        self._preview_scale = float(data)
        self._render_version += 1
        self._tile_cache.clear()
        self._tile_tasks_inflight.clear()

        if not self._current_path or self._base_image is None:
            return

        # Rebuild preview for this image using the new scale
        cache = self._image_cache.get(self._current_path)
        if not cache:
            return

        # Make sure we have the linear image cached
        if "linear_full" not in cache:
            self._ensure_image_cached(self._current_path)
            cache = self._image_cache.get(self._current_path)
            if not cache or "linear_full" not in cache:
                return

        # *** THIS is the important line ***
        self._rebuild_preview_for_path(self._current_path)
        self._build_mipmaps(self._base_image)
        preview_linear = cache.get("preview_linear")
        if preview_linear is None:
            preview_linear = self._mipmaps[0] if self._mipmaps else None
        self._preview_base_linear = preview_linear
        self._render_preview()
        self._schedule_full_render()

        # Restore slider positions to whatever the user set
        if hasattr(self, "adjust_panel"):
            if ui_state and hasattr(self.adjust_panel, "light_group"):
                try:
                    self.adjust_panel.light_group.restore_ui_state(ui_state)
                except Exception:
                    pass
            if color_state and hasattr(self.adjust_panel, "color_group"):
                try:
                    self.adjust_panel.color_group.restore_ui_state(color_state)
                except Exception:
                    pass



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

    # ---------- pop-out adjustment windows ----------

    def _open_adjust_window(self, kind: str, title: str, section: str | None = None):
        """Open a floating window that mirrors the right-dock panels."""
        existing = self._adjust_popups.get(kind)
        if existing and existing.isVisible():
            if section and hasattr(existing, "_panel"):
                panel = existing._panel
                if hasattr(panel, "show_only_section"):
                    panel.show_only_section(section)
                elif hasattr(panel, "expand_section"):
                    panel.expand_section(section)
            existing.raise_()
            existing.activateWindow()
            return

        widget_cls = {
            "levels": LightGroupWidget,
            "color": ColorGroupWidget,
            "detail": ToneGroupWidget,
            "fx": FXGroupWidget,
        }.get(kind)
        if widget_cls is None:
            return

        dlg = QDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        panel = widget_cls(self, dlg)
        dlg._panel = panel  # type: ignore[attr-defined]
        layout.addWidget(panel)
        if section and hasattr(panel, "show_only_section"):
            panel.show_only_section(section)
        elif section and hasattr(panel, "expand_section"):
            panel.expand_section(section)
        self._adjust_popups[kind] = dlg
        dlg.finished.connect(lambda _: self._adjust_popups.pop(kind, None))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()


    def _build_menus(self):
        # Gradience Studio (app) menu
        app_menu = self.menuBar().addMenu("&Gradience Studio")

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

        # Quick access adjustment menus
        adjust_menus = [
            ("&Levels", "levels", "Levels Panel", [
                "Levels", "White Balance", "Brightness / Contrast", "Exposure", "Shadows / Highlights", "Vibrance", "Posterize",
            ]),
            ("C&olor", "color", "Color Panel", [
                "HSL", "Recolor", "Black & White", "Selective Color", "Color Balance", "White Balance",
            ]),
            ("&Detail", "detail", "Detail Panel", [
                "Parametric Curve", "Point Curve Editor", "Channel Mixer", "Gradient Map", "Split Toning", "Normals",
            ]),
            ("F&X", "fx", "FX Panel", ["Lens Filter"]),
        ]
        for label, key, title, sections in adjust_menus:
            menu = self.menuBar().addMenu(label)
            if sections:
                for sect in sections:
                    act = menu.addAction(sect)
                    act.triggered.connect(lambda checked=False, k=key, t=title, s=sect: self._open_adjust_window(k, t, s))
                menu.addSeparator()
            panel_action = menu.addAction(f"Open {title}")
            panel_action.triggered.connect(lambda checked=False, k=key, t=title: self._open_adjust_window(k, t))

        

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
        # Edit menu (placeholder to keep menus visible and aligned)
        edit_menu = self.menuBar().addMenu("&Edit")

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
        # Affinity-style neutral dark palette
        ui_window_bg = "#1E1E1E"
        ui_panel_bg = "#2A2A2A"
        ui_deep_bg = "#252525"
        ui_toolbar_bg = "#2F2F2F"
        ui_canvas_bg = "#252525"
        ui_tray_bg = "#252525"
        ui_control_bg = "#2D2D2D"
        ui_border = "#3A3A3A"
        ui_divider = "#3C3C3C"
        ui_outline = "#3F3F3F"

        text_primary = "#E5E5E5"
        text_secondary = "#B8B8B8"
        text_disabled = "#666666"

        hover_bg = "#3C3C3C"
        pressed_bg = "#454545"
        focus_ring = "#4A4A4A"

        slider_handle = "#D0D0D0"
        slider_filled = "#7A7A7A"
        slider_empty = "#3C3C3C"
        # Brand gradient for slider fill only
        gs_blue = "#1E6DFF"
        gs_indigo = "#5B3CFF"
        gs_purple = "#7C2CFF"
        gs_magenta = "#C12AFF"
        gs_orange = "#FF7A2F"
        grad = f"stop:0 {gs_blue}, stop:0.25 {gs_indigo}, stop:0.5 {gs_purple}, stop:0.75 {gs_magenta}, stop:1 {gs_orange}"
        grad_start = gs_blue
        grad_end = gs_orange
        # embed a tiny checkmark so we never depend on external icon files
        check_icon_css = f"""
            border: 1px solid {ui_border};
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {gs_blue},
                stop:0.25 {gs_indigo},
                stop:0.5 {gs_purple},
                stop:0.75 {gs_magenta},
                stop:1 {gs_orange});
        """

        self.setStyleSheet(f"""
        QMainWindow {{
            background-color: {ui_window_bg};
            color: {text_primary};
        }}
        QWidget#TitleBar {{
            background-color: #202020;
            border-bottom: 1px solid #2A2A2A;
            min-height: 0px;
        }}
        * {{
            font-family: "Segoe UI", "Inter", sans-serif;
            font-size: 12px;
            color: {text_primary};
        }}
        *:disabled {{
            color: {text_disabled};
            font-weight: 400;
        }}
        QLabel {{
            background: transparent;
            font-weight: 400;
            font-size: 12px;
        }}

        /* Menu */
        QMenuBar#MainMenuBar {{
            background-color: transparent;
            color: {text_primary};
            padding: 0 8px;
            height: 32px;
        }}
        QMenuBar#MainMenuBar::item {{
            padding: 6px 12px;
            margin: 0 6px;
            background: transparent;
        }}
        QMenuBar#MainMenuBar::item:selected {{
            background-color: #2A2A2A;
            border-radius: 4px;
            color: {text_primary};
        }}
        QLabel#TitleBarTitle {{
            font-size: 14px;
            font-weight: 600;
            color: {text_primary};
            padding-left: 0px;
        }}
        QMenuBar#MainMenuBar {{
            background-color: transparent;
            color: {text_primary};
        }}
        QMenuBar#MainMenuBar::item {{
            padding: 4px 10px;
            margin: 0 2px;
            background: transparent;
        }}
        QMenuBar#MainMenuBar::item:selected {{
            background-color: #2A2A2A;
            border-radius: 4px;
            color: {text_primary};
        }}
        QMenu {{
            background-color: {ui_panel_bg};
            border: 1px solid {ui_border};
            padding: 6px 0;
        }}
        QMenu::item {{ padding: 6px 18px; color: {text_primary}; }}
        QMenu::item:selected {{ background-color: {ui_control_bg}; }}

        /* Header */
        QWidget#headerBar {{
            background-color: {ui_panel_bg};
            border: 1px solid {ui_border};
            border-radius: 6px;
        }}
        QLabel#headerTitle {{
            font-size: 20px;
            font-weight: 700;
        }}
        QLabel#headerFilename {{
            color: {text_secondary};
            font-weight: 400;
            font-size: 12px;
        }}
        QToolButton#headerBtn {{
            background-color: {ui_control_bg};
            border: 1px solid {ui_border};
            border-radius: 5px;
            padding: 6px 10px;
        }}
        QToolButton#headerBtn:hover {{
            border-color: {focus_ring};
            background-color: {ui_control_bg};
        }}
        QToolButton#headerBtn:pressed {{ background-color: #252525; }}
        QToolButton#headerBtn:disabled {{ color: {text_disabled}; }}

        /* Image toolbar */
        QWidget#imageToolBar {{
            background-color: {ui_panel_bg};
            border: 1px solid {ui_border};
            border-radius: 6px;
        }}
        QWidget#imageToolBar QPushButton {{
            background-color: {ui_control_bg};
            border: 1px solid {ui_outline};
            border-radius: 4px;
            padding: 4px 10px;
            font-weight: 500;
            font-size: 13px;
            color: #E4E4E4;
        }}
        QWidget#imageToolBar QPushButton:hover {{ background-color: {hover_bg}; }}
        QWidget#imageToolBar QPushButton:pressed {{ background-color: {pressed_bg}; }}
        QWidget#imageToolBar QLabel {{ color: {text_primary}; }}

        QComboBox {{
            background-color: {ui_control_bg};
            border: 1px solid {ui_divider};
            padding: 4px 8px;
            border-radius: 4px;
            color: {text_primary};
        }}
        QComboBox::drop-down {{ border: none; width: 18px; }}
        QComboBox QAbstractItemView {{
            background-color: {ui_control_bg};
            selection-background-color: {ui_panel_bg};
            selection-color: {text_primary};
            border: 1px solid {ui_border};
        }}

        /* Filmstrip */
        QListWidget#bottomFilmstrip {{
            background-color: {ui_tray_bg};
            border-top: 1px solid {ui_border};
        }}
        QListWidget#bottomFilmstrip::item {{
            border: 1px solid transparent;
            padding: 6px 4px 2px 4px;
            margin: 2px;
            color: {text_secondary};
        }}
        QListWidget#bottomFilmstrip::item:selected {{
            border: 2px solid {ui_border};
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {grad});
            color: {text_primary};
        }}
        QListWidget#bottomFilmstrip::item:hover:!selected {{
            border: 1px solid {ui_border};
            background-color: {hover_bg};
        }}

        /* Docks */
        QDockWidget {{
            background-color: {ui_panel_bg};
            border: 1px solid {ui_border};
        }}
        QDockWidget::title {{
            padding: 6px 10px;
            background-color: {ui_panel_bg};
            color: {text_primary};
        }}
        QDockWidget#leftSidebar {{ border-right: 1px solid {ui_border}; }}
        QDockWidget#rightSidebar {{ border-left: 1px solid {ui_border}; }}
        QDockWidget#rightSidebar QWidget, QDockWidget#histSidebar QWidget {{
            background-color: {ui_panel_bg};
        }}
        QDockWidget#leftSidebar QWidget {{
            background-color: {ui_tray_bg};
        }}
        QWidget#adjustGradientStrip {{
            height: 4px;
            border-radius: 2px;
            margin: 4px 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {grad});
        }}

        /* Left tabs */
        QTabWidget#leftTabs::pane {{ border: none; }}
        QTabWidget#leftTabs QTabBar::tab {{
            padding: 6px 10px;
            color: {text_secondary};
            background: transparent;
            border: none;
            margin-right: 4px;
        }}
        QTabWidget#leftTabs QTabBar::tab:selected {{
            color: {text_primary};
            border-bottom: 2px solid transparent;
            border-image: linear-gradient(90deg, {grad}) 1;
        }}
        /* Left tab content + trees */
        QWidget#glassPanelLeft {{
            background-color: {ui_tray_bg};
            border: 1px solid {ui_border};
            border-radius: 6px;
        }}
        QTreeWidget#editHistoryTree, QTreeView#folderTree {{
            background-color: {ui_tray_bg};
            border: 1px solid {ui_border};
            padding: 4px 2px;
        }}
        QTreeWidget#editHistoryTree::item, QTreeView#folderTree::item {{
            padding: 2px 4px;
            margin: 1px 0;
        }}
        QTreeWidget#editHistoryTree::item:selected, QTreeView#folderTree::item:selected {{
            background-color: #181818;
            border: 1px solid {ui_border};
            border-radius: 2px;
        }}
        QTreeWidget#editHistoryTree::item:hover:!selected, QTreeView#folderTree::item:hover:!selected {{
            background-color: #181818;
        }}

        /* Histogram panel */
        QWidget#histSidebar {{
            background-color: {ui_panel_bg};
            border: 1px solid {ui_border};
        }}
        QWidget#histSidebar QWidget {{
            background-color: {ui_panel_bg};
        }}
        HistogramWidget, QWidget#histogramPanel, HistogramWidget#histogramPanel {{
            background-color: #000000;
            border: 1px solid {ui_border};
            border-radius: 4px;
            min-height: 120px;
        }}

        /* Adjustments area */
        QWidget#adjustmentsRoot,
        QWidget#adjustStack,
        QWidget#adjustDetailContainer,
        QWidget#adjustScrollContainer {{
            background-color: {ui_panel_bg};
        }}
        QScrollArea#adjustScroll {{
            background: {ui_panel_bg};
            border: none;
        }}
        QScrollArea#adjustScroll QWidget {{
            background: {ui_panel_bg};
        }}
        QPushButton#sectionHeaderButton {{
            background-color: {ui_panel_bg};
            border: 1px solid {ui_border};
            border-radius: 2px;
            padding: 6px 8px;
            font-weight: 600;
            font-size: 12px;
        }}
        QPushButton#sectionHeaderButton:checked {{
            background-color: {ui_control_bg};
            border-color: transparent;
            border-image: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {grad_start}, stop:1 {grad_end}) 1;
        }}
        QWidget#sectionContent {{
            background-color: {ui_panel_bg};
            border: 1px solid {ui_border};
            border-top: none;
        }}

        /* Adjustments tabs */
        QWidget#adjustIconBar {{
            background-color: {ui_panel_bg};
            border-bottom: 1px solid {ui_border};
        }}
        QWidget#adjustIconBar QToolButton {{
            border: none;
            padding: 6px 10px;
            color: {text_secondary};
            font-weight: 500;
            font-size: 13px;
        }}
        QWidget#adjustIconBar QToolButton:checked {{
            color: {text_primary};
            border-bottom: 2px solid transparent;
            border-image: linear-gradient(90deg, {grad}) 1;
        }}

        /* Sliders */
        QSlider::groove:horizontal {{
            border: 1px solid {ui_border};
            height: 10px;
            margin: 6px 10px;
            border-radius: 5px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:1 #000000);
        }}
        QSlider::sub-page:horizontal {{
            border: none;
            border-radius: 5px;
            background: transparent;
        }}
        QSlider::add-page:horizontal {{
            border: none;
            background: {ui_border};
            border-radius: 5px;
        }}
        QSlider::handle:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {gs_blue},
                stop:0.25 {gs_indigo},
                stop:0.5 {gs_purple},
                stop:0.75 {gs_magenta},
                stop:1 {gs_orange});
            border: 1px solid {ui_border};
            width: 16px;
            height: 16px;
            min-width: 16px;
            min-height: 16px;
            max-width: 16px;
            max-height: 16px;
            margin: -6px 0;
            border-radius: 12px; /* force circle */
        }}
        QSlider::handle:horizontal:hover {{ border-color: {focus_ring}; }}

        QSlider[gradientRole="white_level"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:1 #3C3C3C);
        }}
        QSlider[gradientRole="black_level"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #4F4F4F);
        }}
        QSlider[gradientRole="saturation"]::groove {{
            background: qlineargradient( x1:0, y1:0, x2:1, y2:0, stop:0 #ff0000, stop:0.14 #ff7a2f, stop:0.28 #ffed2f, stop:0.42 #2bff2b, stop:0.56 #00a3ff, stop:0.7 #5b3cff, stop:0.84 #c12aff, stop:1 #ff0000);
        }}
        QSlider[gradientRole="vibrance"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #444444, stop:1 #ff9b36);
        }}
        QSlider[gradientRole="temperature"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E6DFF, stop:1 #FF7A2F);
        }}
        QSlider[gradientRole="tint"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2ecc71, stop:1 #c12aff);
        }}
        QSlider[gradientRole="exposure"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #050505, stop:1 #ffffff);
        }}
        QSlider[gradientRole="contrast"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #555555, stop:1 #f5f5f5);
        }}
        QSlider[gradientRole="shadows"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #2F2F2F);
        }}
        QSlider[gradientRole="highlights"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:1 #999999);
        }}
        QSlider[gradientRole="hsl_hue"]::groove {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #000000,
                stop:0.125 #ff0000,
                stop:0.25 #ff7a2f,
                stop:0.375 #ffed2f,
                stop:0.5 #2bff2b,
                stop:0.625 #00a3ff,
                stop:0.75 #5b3cff,
                stop:0.875 #c12aff,
                stop:1 #ff0000
            );
        }}
        QSlider[gradientRole="hsl_hue_red"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #ff0000);
        }}
        QSlider[gradientRole="hsl_hue_orange"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #ff7a2f);
        }}
        QSlider[gradientRole="hsl_hue_yellow"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #ffed2f);
        }}
        QSlider[gradientRole="hsl_hue_green"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #2bff2b);
        }}
        QSlider[gradientRole="hsl_hue_aqua"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #00a3ff);
        }}
        QSlider[gradientRole="hsl_hue_blue"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #5b3cff);
        }}
        QSlider[gradientRole="hsl_hue_purple"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #c12aff);
        }}
        QSlider[gradientRole="hsl_hue_magenta"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #ff00ff);
        }}
        QSlider[gradientRole="hsl_sat"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4c4c4c, stop:1 #ffffff);
        }}
        QSlider[gradientRole="hsl_lum"]::groove {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #ffffff);
        }}

        QSlider[gradientRole="white_level"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:1 #3A3A3A);
        }}
        QSlider[gradientRole="black_level"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #4F4F4F);
        }}
        QSlider[gradientRole="saturation"]::sub-page,
        QSlider[gradientRole="saturation"]::groove {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #ff0000, stop:0.14 #ff7a2f, stop:0.28 #ffed2f,
                stop:0.42 #2bff2b, stop:0.56 #00a3ff, stop:0.7 #5b3cff,
                stop:0.84 #c12aff, stop:1 #ff0000
            );
        }}
        QSlider[gradientRole="vibrance"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #444444, stop:1 #ff9b36);
        }}
        QSlider[gradientRole="temperature"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E6DFF, stop:1 #FF7A2F);
        }}
        QSlider[gradientRole="tint"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2ecc71, stop:1 #c12aff);
        }}
        QSlider[gradientRole="exposure"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #050505, stop:1 #ffffff);
        }}
        QSlider[gradientRole="contrast"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #555555, stop:1 #f5f5f5);
        }}
        QSlider[gradientRole="shadows"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #2F2F2F);
        }}
        QSlider[gradientRole="highlights"]::sub-page {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:1 #999999);
        }}

        QCheckBox {{
            spacing: 8px;
            font-weight: 500;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 3px;
            border: 1px solid #3A3A3A;
            background-color: #2A2A2A;
            image: none;
            background-image: none;
        }}
        QCheckBox::indicator:hover {{
            border: 1px solid #5A5A5A;
            background-color: #333333;
        }}
        QCheckBox::indicator:checked {{
            border: 1px solid #5A5A5A;
            background-color: #2A2A2A;
            {check_icon_css}
        }}
        QCheckBox::indicator:checked:hover {{
            border: 1px solid #6A6A6A;
            background-color: #343434;
        }}
        QCheckBox::indicator:disabled {{
            border: 1px solid #444444;
            background-color: #222222;
            opacity: 0.6;
        }}
        QCheckBox::indicator:checked:disabled {{
            border-color: #444444;
            opacity: 0.45;
        }}

        QPushButton, QToolButton {{
            background-color: {ui_control_bg};
            border: 1px solid {ui_outline};
            border-radius: 4px;
            padding: 4px 10px;
            font-weight: 500;
            font-size: 13px;
            color: #E4E4E4;
        }}
        QPushButton:hover, QToolButton:hover {{
            background-color: {hover_bg};
        }}
        QPushButton:pressed, QToolButton:pressed {{ background-color: {pressed_bg}; }}
        QPushButton:disabled, QToolButton:disabled {{ color: {text_disabled}; }}

        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {{
            background-color: {ui_control_bg};
            border: 1px solid {ui_outline};
            border-radius: 4px;
            padding: 4px 6px;
            color: {text_primary};
            font-weight: 400;
            font-size: 12px;
        }}

        QStatusBar {{
            background-color: {ui_panel_bg};
            color: {text_secondary};
            border-top: 1px solid {ui_border};
            font-size: 11px;
        }}

        #imageDisplay {{
            background-color: {ui_tray_bg};
            border: none;
            color: {text_secondary};
            font-weight: 400;
            font-size: 12px;
        }}
        QScrollArea#imageScroll {{
            background-color: {ui_tray_bg};
            border: 1px solid {ui_border};
        }}

        /* Scrollbars */
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: {ui_panel_bg};
            border: 1px solid {ui_border};
            padding: 2px;
        }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: #4B4B4B;
            border: 1px solid {ui_border};
            min-height: 20px;
            border-radius: 4px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            background: {ui_panel_bg};
            border: none;
            width: 0;
            height: 0;
        }}
        QScrollBar::add-page, QScrollBar::sub-page {{
            background: {ui_panel_bg};
        }}
        """)

    # ---------- simple UI actions ----------

    def _on_toggle_filmstrip(self, checked: bool):
        self.thumbs.setVisible(checked)

    def pick_color(self, parent=None):
        """Wrapper for consistent color picking with validation."""
        color = QColorDialog.getColor(parent=parent or self)
        if color.isValid():
            return color
        return None

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
            "lens_name": None,
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

        # Lens name
        lens_name = data.get("LensModel") or data.get("LensMake")
        if not lens_name:
            lm = data.get("Make", "")
            camera = data.get("Model", "")
            fl = exif_info.get("focal")
            if lm or camera or fl:
                lens_name = " ".join([str(x) for x in [lm, camera, fl] if x])
        exif_info["lens_name"] = lens_name or None

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
            linear_full = cache.get("linear_full")
            if linear_full is not None:
                h, w = linear_full.shape[:2]
                depth = 32
            else:
                w = h = depth = 0
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

        # Lens profile detection
        lens_name = exif.get("lens_name") or ""
        geom = self._current_params.get("geometry") or {}
        if isinstance(geom, dict):
            geom["lens_profile_name"] = lens_name
        self._current_params["geometry"] = geom
        if hasattr(self, "adjust_panel") and getattr(self.adjust_panel, "geometry_group", None):
            self.adjust_panel.geometry_group.update_detected_lens(lens_name)


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
        effect_key = data.get("effect_key")
        mask_name = data.get("mask")

        if not path:
            return

        is_parent = (self._image_parents.get(path) is item)

        menu = QMenu(self)
        act_copy = menu.addAction("Copy Values")
        act_paste = menu.addAction("Paste Values")

        act_remove_image = None
        act_remove_effect = None
        act_remove_mask = None

        if is_parent:
            menu.addSeparator()
            act_remove_image = menu.addAction("Remove From Active Edit")
        elif effect_key:
            menu.addSeparator()
            act_remove_effect = menu.addAction("Remove Effect")
        elif mask_name and not effect_key:
            menu.addSeparator()
            act_remove_mask = menu.addAction("Remove Mask Effects")

        chosen = menu.exec(self.history_tree.viewport().mapToGlobal(pos))
        if chosen is act_copy:
            self._copy_active_edit(path, params)
        elif chosen is act_paste:
            self._paste_active_edit(path)
        elif act_remove_image is not None and chosen is act_remove_image:
            self._remove_active_edit_image(path)
        elif act_remove_effect is not None and chosen is act_remove_effect:
            self._remove_active_effect(path, effect_key, mask_name or "Global", params)
        elif act_remove_mask is not None and chosen is act_remove_mask:
            if path in self._active_edits_by_path:
                effects = self._active_edits_by_path[path].get("effects", [])
                self._active_edits_by_path[path]["effects"] = [
                    e for e in effects if e.get("mask", "Global") != (mask_name or "Global")
                ]
                if path == self._current_path:
                    self._update_history_for_current_image()

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

        self._show_image_version(path, self._copied_params)
        self._update_history_for_current_image()
        self.export_lrc(autosave=True)

        self.statusBar().showMessage(
            f"Pasted Active Edit values to {os.path.basename(path)}", 2000
        )

    def _remove_active_edit_image(self, path: str):
        self._image_params.pop(path, None)
        self._edits.pop(path, None)
        self._active_edits_by_path.pop(path, None)

        parent_item = self._image_parents.pop(path, None)
        if parent_item is not None:
            idx = self.history_tree.indexOfTopLevelItem(parent_item)
            if idx >= 0:
                self.history_tree.takeTopLevelItem(idx)

        if self._current_path == path:
            self._show_image_version(path, {})

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
        dlg.setWindowTitle("Preferences - Gradience Studio")
        layout = QFormLayout(dlg)

        spin = QSpinBox(dlg)
        spin.setRange(0, 120)
        spin.setSuffix(" min")
        spin.setValue(self._autosave_interval_min)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        layout.addRow("Autosave interval:", spin)

        thumb_size_spin = QSpinBox(dlg)
        thumb_size_spin.setRange(32, 160)
        thumb_size_spin.setValue(getattr(self, "_thumb_icon_px", 80))
        thumb_size_spin.setSuffix(" px")
        layout.addRow("Thumbnail size:", thumb_size_spin)

        thumb_inflight_spin = QSpinBox(dlg)
        thumb_inflight_spin.setRange(4, 128)
        thumb_inflight_spin.setValue(getattr(self, "_max_thumb_inflight", 32))
        layout.addRow("Max concurrent thumbnails:", thumb_inflight_spin)

        debug_chk = QCheckBox("Enable debug overlay (FPS graph)", dlg)
        debug_chk.setChecked(getattr(self, "_debug_overlay_enabled", False))
        layout.addRow(debug_chk)

        cuda_basic_chk = QCheckBox("Use CUDA for adjustments (if available)", dlg)
        cuda_basic_chk.setChecked(get_cuda_basic_enabled())
        cuda_basic_chk.setEnabled(_CUDA_AVAILABLE)
        layout.addRow(cuda_basic_chk)

        cuda_sharp_chk = QCheckBox("Use CUDA for sharpening (if available)", dlg)
        cuda_sharp_chk.setChecked(get_cuda_sharpen_enabled())
        cuda_sharp_chk.setEnabled(_CUDA_AVAILABLE)
        layout.addRow(cuda_sharp_chk)

        cuda_thumb_chk = QCheckBox("Use CUDA for thumbnail scaling (if available)", dlg)
        cuda_thumb_chk.setChecked(get_cuda_thumbs_enabled())
        layout.addRow(cuda_thumb_chk)

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

            new_thumb_px = thumb_size_spin.value()
            if new_thumb_px != getattr(self, "_thumb_icon_px", 80):
                self._thumb_icon_px = new_thumb_px
                self.thumbs.setIconSize(QSize(new_thumb_px, new_thumb_px))
                self._ensure_visible_thumbs()

            self._max_thumb_inflight = thumb_inflight_spin.value()

            self._set_debug_overlay_enabled(debug_chk.isChecked())
            set_cuda_basic_enabled(cuda_basic_chk.isChecked())
            set_cuda_sharpen_enabled(cuda_sharp_chk.isChecked())
            set_cuda_thumbs_enabled(cuda_thumb_chk.isChecked())

    # ---------- new project ----------

    def new_project(self):
        self._current_folder = None
        self._current_path = None
        self._preview_base_linear = None
        self._current_pixmap = None
        self._project_path = None
        self._project_root = None

        self._image_cache.clear()
        self._image_params.clear()
        self._image_parents.clear()
        self._edits.clear()
        self._active_edits_by_path.clear()
        self._active_edit = {"effects": []}

        self.thumbs.clear()
        self.history_tree.clear()
        self.image_display.clear()
        self.image_display.setText("No folder open. File → Import Folder")

        self._current_params = copy.deepcopy(self._default_params_template)
        self._current_params.update({
            "levels_black": 0,
            "levels_white": 100,
            "levels_gamma": 1.0,
            "levels_out_black": 0,
            "levels_out_white": 100,
            "levels_color_model": "RGB",
            "levels_channel": "Master",
        })

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
        orig_icon = QIcon()
        if cache:
            preview_linear = cache.get("preview_linear")
            if preview_linear is not None:
                orig_icon = QIcon(QPixmap.fromImage(qimage_from_linear(preview_linear)))

        edited_icon = QIcon(
            self._current_pixmap.scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        combined_params = self._image_params.get(path, {})
        active_edit = self._active_edits_by_path.get(path)
        if active_edit is None:
            active_edit = self._ensure_active_edit_tree(path)
        self._active_edit = active_edit

        effects = list(active_edit.get("effects", []))

        parent.setIcon(0, edited_icon)
        parent.setData(0, Qt.UserRole, {"path": path, "params": dict(combined_params)})

        child_orig = QTreeWidgetItem(parent)
        child_orig.setText(0, "Original")
        child_orig.setIcon(0, orig_icon)
        child_orig.setData(0, Qt.UserRole, {"path": path, "params": {}})

        masks: dict[str, list] = {}
        for eff in effects:
            mask_name = eff.get("mask", "Global")
            masks.setdefault(mask_name, []).append(eff)

        if not masks:
            masks["Global"] = []

        def _sorted_masks():
            if "Global" in masks:
                yield "Global"
            for name in sorted(m for m in masks.keys() if m != "Global"):
                yield name

        for mask_name in _sorted_masks():
            eff_list = masks.get(mask_name, [])
            mask_item = QTreeWidgetItem(parent)
            mask_label = f"Mask: {mask_name}"
            if eff_list:
                mask_label += f"  ({len(eff_list)} effect{'s' if len(eff_list) != 1 else ''})"
            mask_item.setText(0, mask_label)
            # mask node shows the unedited preview for context
            mask_item.setIcon(0, orig_icon)
            mask_item.setData(0, Qt.UserRole, {"path": path, "params": {}, "mask": mask_name})
            mask_item.setToolTip(0, "Mask preview (base image)")

            for eff in eff_list:
                label = eff.get("label") or eff.get("key")
                label = self._format_effect_label(label, eff.get("params", {}))
                child = QTreeWidgetItem(mask_item)
                child.setText(0, label)
                child.setIcon(0, edited_icon)
                child_params = eff.get("params", {})
                child.setToolTip(0, self._format_effect_tooltip(child_params))
                child.setData(
                    0,
                    Qt.UserRole,
                    {
                        "path": path,
                        "params": child_params,
                        "mask": mask_name,
                        "effect_key": eff.get("key"),
                    },
                )

        parent.setExpanded(True)

    def _on_history_item_clicked(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path")
        params = data.get("params", {})
        if not path:
            return
        self._show_image_version(path, params)

    # ---------- Image pipeline ----------

    def _rebuild_preview_for_path(self, path: str):
        """
        Rebuild the preview image for a given path based on self._preview_scale.
        Keeps the full-resolution image untouched in the cache.
        """
        if not hasattr(self, "_preview_scale"):
            self._preview_scale = 1.0

        cache = self._image_cache.get(path)
        if not cache:
            return

        preview_base = cache.get("preview_source")
        if preview_base is None:
            preview_base = cache.get("linear_full")
        if preview_base is None:
            return

        scale = max(0.0, float(self._preview_scale or 1.0))
        if scale >= 0.999:
            preview_linear = preview_base
        else:
            preview_linear = self._resample_linear_preview(preview_base, scale)

        cache["preview_linear"] = preview_linear

    def _resample_linear_preview(self, linear: np.ndarray, scale: float) -> np.ndarray:
        """High-quality downscale using PIL with LANCZOS while preserving linear data."""
        h, w = linear.shape[:2]
        target_w = max(1, int(round(w * scale)))
        target_h = max(1, int(round(h * scale)))

        srgb = to_srgb(np.clip(linear, 0.0, 1.0))
        srgb_u8 = (srgb * 255.0).round().astype(np.uint8)
        pil_img = Image.fromarray(srgb_u8, mode="RGB")
        resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
        resized_arr = np.asarray(resized, dtype=np.float32) / 255.0
        return to_linear(resized_arr)

    def _build_mipmaps(self, base_image: np.ndarray) -> None:
        """
        Create downscaled versions of base_image and store them in
        self._mipmaps and self._mip_scales.
        """
        self._mipmaps = []
        self._mip_scales = []
        if base_image is None:
            return

        self._mipmaps.append(base_image)
        self._mip_scales.append(1.0)

        h, w = base_image.shape[:2]
        scale = 0.5
        target_min = 96  # build deeper/smaller levels for drag preview
        while min(h, w) * scale >= target_min and scale >= 0.02:
            mip = self._resample_linear_preview(base_image, scale)
            self._mipmaps.append(mip)
            self._mip_scales.append(scale)
            scale *= 0.5

    def _choose_preview_mip(self) -> int:
        """
        Return the index into self._mipmaps that is closest in size
        to the current on-screen display size (taking zoom into account).
        """
        if not self._mipmaps:
            return 0

        base_h, base_w = self._mipmaps[0].shape[:2]
        vp_size = self.image_scroll.viewport().size()
        if vp_size.width() <= 0 or vp_size.height() <= 0:
            return 0

        if getattr(self, "_zoom_mode", "fit") == "fit":
            factor = min(vp_size.width() / max(1.0, base_w), vp_size.height() / max(1.0, base_h))
            target_w = base_w * factor
            target_h = base_h * factor
        else:
            factor = getattr(self, "_zoom_factor", 1.0)
            target_w = base_w * factor
            target_h = base_h * factor

        target_w *= float(getattr(self, "_preview_scale", 1.0) or 1.0)
        target_h *= float(getattr(self, "_preview_scale", 1.0) or 1.0)

        best_idx = 0
        best_diff = float("inf")
        for idx, mip in enumerate(self._mipmaps):
            mh, mw = mip.shape[:2]
            diff = abs(mw - target_w) + abs(mh - target_h)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        return best_idx


    def _ensure_image_cached(self, path: str):
        """
        Ensure we have at least the full image cached,
        and a preview built for the current preview scale.
        """
        cache = self._image_cache.get(path)
        if cache is not None:
            # If preview is missing (e.g. after we changed scale), rebuild it
            if "preview_linear" not in cache or cache.get("preview_linear") is None:
                self._rebuild_preview_for_path(path)
            return

        linear_full = _load_image_full_linear(path)
        preview_src = _load_image_preview_linear(path)
        if preview_src is None:
            preview_src = linear_full
        if linear_full is None and preview_src is None:
            return

        cache_entry = {
            "linear_full": linear_full if linear_full is not None else preview_src,
            "preview_source": preview_src,
        }

        src_for_preview = preview_src if preview_src is not None else linear_full
        if src_for_preview is not None:
            scale = max(0.0, float(getattr(self, "_preview_scale", 1.0) or 1.0))
            if scale >= 0.999:
                cache_entry["preview_linear"] = src_for_preview
            else:
                cache_entry["preview_linear"] = self._resample_linear_preview(src_for_preview, scale)

        self._image_cache[path] = cache_entry


    def _show_image_version(self, path: str, params: dict, update_sliders: bool = False):
        self._ensure_image_cached(path)
        cache = self._image_cache.get(path)
        if not cache:
            return

        self._reset_render_cache()
        self._current_path = path
        self._update_header_filename(path)
        self._base_image = cache.get("linear_full")
        if self._base_image is None:
            return

        cached_mips = cache.get("mipmaps")
        cached_scales = cache.get("mip_scales")
        if cached_mips and cached_scales:
            self._mipmaps = cached_mips
            self._mip_scales = cached_scales
        else:
            self._build_mipmaps(self._base_image)
            cache["mipmaps"] = self._mipmaps
            cache["mip_scales"] = self._mip_scales

        if "preview_linear" not in cache or cache.get("preview_linear") is None:
            cache["preview_linear"] = self._resample_linear_preview(self._base_image, self._preview_scale)

        preview_linear = cache.get("preview_linear")
        if preview_linear is None:
            preview_linear = self._mipmaps[0] if self._mipmaps else None
        self._preview_base_linear = preview_linear
        self._full_render_timer.stop()
        self._render_timer.stop()
        self._prepare_base_arrays()

        def get(name, default=128):
            return params.get(name, default)

        self._current_params.update(
            {
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
                "levels_black": get("levels_black", 0),
                "levels_white": get("levels_white", 100),
                "levels_gamma": get("levels_gamma", 1.0),
                "levels_out_black": get("levels_out_black", 0),
                "levels_out_white": get("levels_out_white", 100),
                "levels_color_model": get("levels_color_model", "RGB"),
                "levels_channel": get("levels_channel", "Master"),
                "levels_linear": get("levels_linear", False),
                "view_exposure_bias": get("view_exposure_bias", 0.0),
            }
        )
        ot_val = params.get("output_transform", self._current_params.get("output_transform", OutputTransformMode.STANDARD.value))
        if isinstance(ot_val, OutputTransformMode):
            ot_val = ot_val.value
        self._current_params["output_transform"] = str(ot_val or OutputTransformMode.STANDARD.value)
        geom_dict = copy.deepcopy(self._geometry_defaults)
        geom_dict.update(params.get("geometry", {}))
        self._current_params["geometry"] = geom_dict
        color_state = merge_color_params(params.get("color"), params)
        self._current_params["color"] = color_state
        for _color in ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]:
            hsl = color_state.get("hsl", {})
            hue_map = hsl.get("hue", {})
            sat_map = hsl.get("sat", {})
            lum_map = hsl.get("lum", {})
            self._current_params[f"hsl_hue_{_color}"] = hue_map.get(_color, get(f"hsl_hue_{_color}", 0))
            self._current_params[f"hsl_sat_{_color}"] = sat_map.get(_color, get(f"hsl_sat_{_color}", 0))
            self._current_params[f"hsl_lum_{_color}"] = lum_map.get(_color, get(f"hsl_lum_{_color}", 0))

        try:
            if hasattr(self, "adjust_panel") and hasattr(self.adjust_panel, "fx_group"):
                combo = getattr(self.adjust_panel.fx_group, "output_combo", None)
                if combo is not None:
                    combo.blockSignals(True)
                    mode_lower = str(self._current_params.get("output_transform", OutputTransformMode.STANDARD.value)).lower()
                    combo.setCurrentIndex(1 if mode_lower == "filmic" else 0)
                    combo.blockSignals(False)
        except Exception:
            pass

        self._active_edit = self._ensure_active_edit_tree(path)

        if update_sliders and hasattr(self, "adjust_panel"):
            try:
                self.adjust_panel.light_group.reset_ui_to_defaults()
            except Exception:
                pass
            try:
                self.adjust_panel.color_group.reset_ui_to_defaults()
            except Exception:
                pass
            try:
                self.adjust_panel.tone_group.reset_ui_to_defaults()
            except Exception:
                pass

        self._update_metadata_for_current()

        self._render_preview()
        self._schedule_full_render()

    def _prepare_base_arrays(self):
        """Compute base RGB, luminance and tone masks once per base image."""
        if self._preview_base_linear is None:
            self._reset_render_cache()
            return

        rgb = np.clip(self._preview_base_linear, 0.0, 1.0)

        lum = (
            0.2126 * rgb[..., 0] +
            0.7152 * rgb[..., 1] +
            0.0722 * rgb[..., 2]
        )

        L = np.clip(lum, 0.0, 1.0)
        highlights_mask = np.clip((L - 0.5) / 0.5, 0.0, 1.0)
        shadows_mask    = np.clip((0.5 - L) / 0.5, 0.0, 1.0)
        whites_mask     = np.clip((L - 0.8) / 0.2, 0.0, 1.0)
        blacks_mask     = np.clip((0.2 - L) / 0.2, 0.0, 1.0)

        self._base_linear = rgb.copy()
        self._base_lum = lum
        self._L_norm = L
        self._tone_masks = {
            "highlights": highlights_mask,
            "shadows": shadows_mask,
            "whites": whites_mask,
            "blacks": blacks_mask,
        }


    def _current_params_for_render(self) -> dict:
        params = copy.deepcopy(self._current_params)
        params["color"] = merge_color_params(params.get("color"), params)
        self._current_params["color"] = params["color"]
        return params

    def _current_edit_params(self) -> EditParams:
        return _dict_to_edit_params(self._current_params_for_render())

    def _compute_display_scale(self) -> tuple[float, int, int]:
        """Return (scale, target_w, target_h) for mapping base pixels to the viewport."""
        if self._base_image is None:
            pm = self.image_display.pixmap()
            if pm is not None and not pm.isNull():
                return 1.0, pm.width(), pm.height()
            vp = self.image_scroll.viewport().size()
            return 1.0, vp.width(), vp.height()

        base_h, base_w = self._base_image.shape[:2]
        vp_size = self.image_scroll.viewport().size()
        if vp_size.width() <= 0 or vp_size.height() <= 0:
            return 1.0, base_w, base_h

        if getattr(self, "_zoom_mode", "fit") == "fit":
            scale = min(vp_size.width() / max(1.0, base_w), vp_size.height() / max(1.0, base_h))
        else:
            scale = getattr(self, "_zoom_factor", 1.0)

        scale = max(1e-3, float(scale))
        target_w = max(1, int(round(base_w * scale)))
        target_h = max(1, int(round(base_h * scale)))
        return scale, target_w, target_h

    def _paint_tile_on_display(self, painter: QPainter, tile_key: TileKey, tile_arr: np.ndarray, display_scale: float):
        """Draw a rendered tile into the display image respecting the current zoom."""
        if tile_arr is None or not self._mip_scales:
            return
        if tile_key.level >= len(self._mip_scales):
            return

        mip_scale = self._mip_scales[tile_key.level]
        tile_h, tile_w = tile_arr.shape[:2]
        base_tile_w = tile_w / mip_scale
        base_tile_h = tile_h / mip_scale

        x = int(round(tile_key.tx * self._tile_size / mip_scale * display_scale))
        y = int(round(tile_key.ty * self._tile_size / mip_scale * display_scale))
        w = max(1, int(round(base_tile_w * display_scale)))
        h = max(1, int(round(base_tile_h * display_scale)))

        qimg = qimage_from_linear(np.clip(tile_arr, 0.0, 1.0))
        painter.drawImage(QRect(x, y, w, h), qimg)

    def _compose_display_from_preview(self):
        """Create the composited display image (preview + any cached tiles)."""
        if self._preview_qimage is None:
            return

        display_scale, target_w, target_h = self._compute_display_scale()
        if target_w <= 0 or target_h <= 0:
            return

        display_img = self._preview_qimage.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        actual_scale = display_img.width() / max(1, self._base_image.shape[1]) if self._base_image is not None else display_scale

        painter = QPainter(display_img)
        try:
            for key, tile_arr in self._tile_cache.items():
                self._paint_tile_on_display(painter, key, tile_arr, actual_scale)
        finally:
            painter.end()

        self._display_scale = actual_scale
        self._display_qimage = display_img
        self._current_pixmap = QPixmap.fromImage(display_img)
        self.image_display.setPixmap(self._current_pixmap)
        self.image_display.resize(self._current_pixmap.size())
        self.image_display.setText("")
        self._update_zoom_label()
        self._mark_frame_presented()

    def _update_display_from_qimage(self, qimg: QImage):
        """Store a new preview qimage and composite it with any available tiles."""
        if qimg is None:
            return
        self._preview_qimage = qimg
        self._compose_display_from_preview()

    def _render_preview(self):
        """Run apply_pipeline() on the selected mipmap and show the result immediately."""
        force_smallest = self._force_smallest_preview or self._drag_preview_mode
        self._force_smallest_preview = False

        if self._base_image is None and self._preview_base_linear is None:
            return

        if not self._mipmaps and self._base_image is not None:
            self._build_mipmaps(self._base_image)

        if self._mipmaps:
            geometry_active = not _geometry_is_identity(_dict_to_geometry_params(self._current_params.get("geometry")))
            if geometry_active:
                level_idx = 0  # force highest quality when geometry is active to avoid blocky previews
            elif force_smallest:
                level_idx = len(self._mipmaps) - 1  # smallest mip for fastest slider drag preview
            else:
                level_idx = self._choose_preview_mip()
            level_idx = min(level_idx, len(self._mipmaps) - 1)
            base = self._mipmaps[level_idx]
            self._preview_level = level_idx
        else:
            base = self._preview_base_linear
            self._preview_level = 0

        if base is None:
            return

        # Downscale preview via OpenCV and force CPU pipeline for interactivity
        self._preview_base_linear = base
        params = self._current_edit_params()
        bh, bw = base.shape[:2]
        max_preview = 2048
        scale = min(1.0, max_preview / max(bh, bw)) if max(bh, bw) > 0 else 1.0
        preview_img = base
        if scale < 0.999:
            new_w = max(1, int(bw * scale))
            new_h = max(1, int(bh * scale))
            preview_img = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = apply_pipeline_cpu(preview_img, params, quality="preview")
        if result is None:
            return

        qimg = qimage_from_linear(result)
        self._preview_qimage = qimg
        self._update_display_from_qimage(qimg)
        if not self._drag_preview_mode:
            self._update_histogram()

        if self._current_path:
            params_copy = copy.deepcopy(self._current_params_for_render())
            self._image_params[self._current_path] = params_copy
            self._edits[self._current_path] = params_copy
            self._update_history_for_current_image()

        if not self._drag_preview_mode:
            self._schedule_background_tiles()
        else:
            # Keep tiles paused until the user releases slider; idle timer will resume.
            self._tile_render_timer.stop()

    def _render_full(self):
        """Run apply_pipeline() on the full-res image in a background worker, then swap into the viewer."""
        self._start_full_render_worker()

    def _schedule_background_tiles(self, delay_ms: int = 150):
        """Debounce tile rendering so we don't flood the pool during slider drags."""
        # When geometry is active, skip tile rendering to avoid misaligned tiles; rely on preview + full render.
        if not _geometry_is_identity(self._current_params.get("geometry")):
            self._tile_render_timer.stop()
            return
        self._tile_render_timer.stop()
        self._tile_render_timer.start(delay_ms)

    def _choose_tile_level(self) -> int:
        """Pick the mip level closest to the on-screen scale."""
        if not self._mip_scales:
            return 0
        display_scale, _, _ = self._compute_display_scale()
        best_idx = 0
        best_diff = float("inf")
        for idx, scale in enumerate(self._mip_scales):
            diff = abs(scale - display_scale)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        return best_idx

    def _visible_tiles_for_level(self, level_idx: int) -> list[TileKey]:
        """Return the list of tile keys that intersect the viewport at a mip level."""
        if level_idx >= len(self._mipmaps):
            return []
        mip = self._mipmaps[level_idx]
        lvl_h, lvl_w = mip.shape[:2]
        display_scale, _, _ = self._compute_display_scale()
        if display_scale <= 0:
            return []

        hbar = self.image_scroll.horizontalScrollBar()
        vbar = self.image_scroll.verticalScrollBar()
        vp = self.image_scroll.viewport().size()

        base_x = hbar.value() / max(1e-6, display_scale)
        base_y = vbar.value() / max(1e-6, display_scale)
        base_w = vp.width() / max(1e-6, display_scale)
        base_h = vp.height() / max(1e-6, display_scale)

        mip_scale = self._mip_scales[level_idx]
        vis_x = base_x * mip_scale
        vis_y = base_y * mip_scale
        vis_w = base_w * mip_scale
        vis_h = base_h * mip_scale

        tx0 = int(max(0, np.floor(vis_x / self._tile_size)))
        ty0 = int(max(0, np.floor(vis_y / self._tile_size)))
        tx1 = int(min(np.ceil((vis_x + vis_w) / self._tile_size), np.ceil(lvl_w / self._tile_size)))
        ty1 = int(min(np.ceil((vis_y + vis_h) / self._tile_size), np.ceil(lvl_h / self._tile_size)))

        keys: list[TileKey] = []
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                if tx * self._tile_size >= lvl_w or ty * self._tile_size >= lvl_h:
                    continue
                keys.append(TileKey(level_idx, tx, ty))
        return keys

    def _start_background_tiles(self):
        """Submit visible tiles to the render pool in priority order."""
        if not _geometry_is_identity(self._current_params.get("geometry")):
            return
        if self._shutting_down or not self._mipmaps:
            return
        if self._preview_qimage is None:
            return

        level_idx = self._choose_tile_level()
        visible = self._visible_tiles_for_level(level_idx)
        if not visible:
            return

        display_scale, _, _ = self._compute_display_scale()
        hbar = self.image_scroll.horizontalScrollBar()
        vbar = self.image_scroll.verticalScrollBar()
        vp = self.image_scroll.viewport().size()
        center_base_x = hbar.value() / max(1e-6, display_scale) + vp.width() / (2 * max(1e-6, display_scale))
        center_base_y = vbar.value() / max(1e-6, display_scale) + vp.height() / (2 * max(1e-6, display_scale))
        center_mip_x = center_base_x * self._mip_scales[level_idx]
        center_mip_y = center_base_y * self._mip_scales[level_idx]

        def _priority(key: TileKey) -> float:
            tile_cx = (key.tx + 0.5) * self._tile_size
            tile_cy = (key.ty + 0.5) * self._tile_size
            return abs(tile_cx - center_mip_x) + abs(tile_cy - center_mip_y)

        visible.sort(key=_priority)

        params = self._current_edit_params()
        version = self._render_version
        mip_level = self._mipmaps[level_idx]
        high_detail = (level_idx == 0) or (self._zoom_mode == "manual" and self._zoom_factor >= 1.0)
        quality = "final" if high_detail else "preview"
        max_queue = self._render_pool.maxThreadCount() * 2

        for key in visible:
            if key in self._tile_cache or key in self._tile_tasks_inflight:
                continue
            if len(self._tile_tasks_inflight) >= max_queue:
                break
            task = TileRenderTask(key, mip_level, params, version, self._tile_size, quality=quality)
            task.signals.finished.connect(self._on_tile_rendered)
            self._tile_tasks_inflight.add(key)
            self._render_pool.start(task)

    def _on_tile_rendered(self, tile_key: TileKey, tile_image: np.ndarray, version: int):
        if self._shutting_down:
            return
        if version != self._render_version:
            return
        self._tile_tasks_inflight.discard(tile_key)
        if tile_image is None:
            return

        self._tile_cache[tile_key] = tile_image
        if self._display_qimage is None or self._preview_qimage is None:
            self._compose_display_from_preview()
            return

        painter = QPainter(self._display_qimage)
        try:
            self._paint_tile_on_display(painter, tile_key, tile_image, self._display_scale)
        finally:
            painter.end()

        self._current_pixmap = QPixmap.fromImage(self._display_qimage)
        self.image_display.setPixmap(self._current_pixmap)
        self.image_display.update()
        self._mark_frame_presented()

    def _schedule_preview_render(self):
        """Debounce rendering so rapid slider moves don't re-render every tick."""
        self._render_timer.stop()
        self._force_smallest_preview = True
        delay = self._drag_render_timer_ms if self._drag_preview_mode else 5
        self._render_timer.start(delay)

    def _schedule_full_render(self, delay_ms: int = 200):
        """Start a delayed high-quality render unless another change arrives first."""
        self._full_render_timer.stop()
        self._full_render_timer.start(delay_ms)

    def _start_full_render_worker(self):
        if self._base_image is None or self._shutting_down:
            return

        params = self._current_params_for_render()
        self._full_render_token += 1
        token = self._full_render_token

        worker = _FullRenderWorker(self._base_image, copy.deepcopy(params), token)
        worker.signals.finished.connect(self._on_full_render_finished)
        self._last_full_worker = worker  # keep ref so signals stay alive
        self._render_pool.start(worker)

    def _on_full_render_finished(self, image_array: np.ndarray, params_used: dict, token: int):
        if token != self._full_render_token:
            return
        if params_used != self._current_params:
            return
        if image_array is None:
            return

        self._current_pixmap = QPixmap.fromImage(qimage_from_linear(np.clip(image_array, 0.0, 1.0)))
        self._rescale_preview()
        self._update_histogram()

    def _on_edit_params_changed(self, immediate_preview: bool = False):
        # Treat slider moves as "drag" until idle timer fires; show tiniest mip immediately
        self._drag_preview_mode = True
        self._idle_mip_timer.start(150)
        self._render_version += 1
        self._tile_cache.clear()
        self._tile_tasks_inflight.clear()
        if immediate_preview:
            self._render_timer.stop()
            self._force_smallest_preview = True
            self._render_preview()
        else:
            self._schedule_preview_render()
        self._schedule_background_tiles()

    def _apply_edit_params_to_current_image(self, params: dict):
        if params:
            self._current_params.update(params)
        if self._current_path:
            self._active_edit = self._ensure_active_edit_tree(self._current_path)
        self._on_edit_params_changed(immediate_preview=True)

    def _schedule_render(self):
        """Backward-compatible entry point for existing slider hooks."""
        self._on_edit_params_changed()

    def _render_current_params(self):
        self._render_preview()

    def _on_idle_preview_upgrade(self):
        """User stopped dragging; render a higher mip and restart tiles."""
        self._drag_preview_mode = False
        self._force_smallest_preview = False
        self._render_preview()

    def _rescale_preview(self):
        if not hasattr(self, "_zoom_mode"):
            self._zoom_mode = "fit"
        if not hasattr(self, "_zoom_factor"):
            self._zoom_factor = 1.0

        if self._preview_qimage is not None:
            self._compose_display_from_preview()
            return

        if self._current_pixmap is None:
            return

        if self._zoom_mode == "fit":
            vp_size = self.image_scroll.viewport().size()
            if vp_size.width() <= 0 or vp_size.height() <= 0:
                return
            scaled = self._current_pixmap.scaled(
                vp_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
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
        self._mark_frame_presented()


    # ---------- Thumbnail / folder handling ----------

    def _on_thumbnail_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        params = self._image_params.get(path, {})
        self._show_image_version(path, params)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", os.path.expanduser("~")
        )
        if folder:
            self.load_folder(folder)

    def _queue_thumb(self, path: str):
        if path in self._icon_cache or path in self._loading:
            return
        if len(self._loading) >= getattr(self, "_max_thumb_inflight", 32):
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
        self._preview_base_linear = None
        self._current_pixmap = None
        self._update_header_filename(None)

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
            self._schedule_background_tiles(0)
        self._update_debug_overlay_geometry()
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

    def closeEvent(self, event):
        """Ensure background renders don't emit after shutdown."""
        self._shutting_down = True
        self._full_render_token += 1
        self._full_render_timer.stop()
        self._idle_mip_timer.stop()
        self._tile_render_timer.stop()
        try:
            self._save_perf_graph()
            if self._debug_log_path:
                self.statusBar().showMessage(f"Saved FPS log to {self._debug_log_path}", 4000)
        except Exception:
            pass
        return super().closeEvent(event)



    # ---------- Export / import (project files) ----------

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

            self._reset_render_cache()
            self._current_path = path
            self._base_image = cache.get("linear_full")
            if self._base_image is None:
                continue

            self._build_mipmaps(self._base_image)
            if "preview_linear" not in cache or cache.get("preview_linear") is None:
                cache["preview_linear"] = self._resample_linear_preview(self._base_image, self._preview_scale)
            preview_linear = cache.get("preview_linear")
            if preview_linear is None:
                preview_linear = self._mipmaps[0] if self._mipmaps else None
            self._preview_base_linear = preview_linear
            self._active_edit = self._ensure_active_edit_tree(path)
            self._current_params = copy.deepcopy(self._default_params_template)
            if params:
                self._current_params.update(params)

            self._render_preview()
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
            "schema": "ECE 277 Gradience Studio Project",
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
                os.path.join(default_folder, f"project{PROJECT_EXTENSION}"),
                f"Gradience Studio Project (*{PROJECT_EXTENSION})",
            )
            if not path:
                return
            if not path.lower().endswith(PROJECT_EXTENSION):
                path += PROJECT_EXTENSION
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
            f"Gradience Studio Project (*{PROJECT_EXTENSION});;Legacy Lightroom Clone Project (*{LEGACY_PROJECT_EXTENSION})",
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

        schema = data.get("schema")
        valid_schemas = {"ECE 277 Gradience Studio Project", "ECE 277 LightRoom Project"}
        if schema not in valid_schemas:
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


