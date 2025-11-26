import os
from typing import Optional

import numpy as np
from PIL import Image

try:
    import rawpy
except ImportError:
    rawpy = None

from PySide6.QtGui import QImage, QImageReader

from .constants import RAW_EXTENSIONS


def relpath_or_same(path: str, start: str) -> str:
    """Return a relative path when possible; fall back to the original on failure."""
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path


def abspath_from_base(maybe_rel: str, base: str) -> str:
    """Resolve a path relative to *base* unless it is already absolute."""
    if os.path.isabs(maybe_rel):
        return os.path.abspath(maybe_rel)
    return os.path.abspath(os.path.join(base, maybe_rel))


def load_qimage_any(path: str) -> Optional[QImage]:
    """
    Load a QImage from a path, using rawpy for RAW formats when available.
    Returns None on failure.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in RAW_EXTENSIONS and rawpy is not None:
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    output_bps=8,
                    no_auto_bright=True,
                    gamma=(2.2, 4.5),
                )
            h, w, _ = rgb.shape
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            return img.copy()
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[RAW] Failed to decode {path}: {exc}")
            return None

    reader = QImageReader(path)
    reader.setAutoTransform(True)
    img = reader.read()
    if img.isNull():
        return None
    return img


def to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert gamma-encoded sRGB [0,1] into linear float32."""
    arr = np.asarray(rgb, dtype=np.float32)
    linear = np.where(
        arr <= 0.04045,
        arr / 12.92,
        np.power((arr + 0.055) / 1.055, 2.4),
    )
    return np.clip(linear, 0.0, 1.0)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1] into gamma-encoded sRGB float32."""
    arr = np.asarray(linear, dtype=np.float32)
    srgb = np.where(
        arr <= 0.0031308,
        arr * 12.92,
        1.055 * np.power(arr, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def qimage_from_linear(linear: np.ndarray) -> QImage:
    """Create an RGB888 QImage from a [0,1] linear float32 array."""
    arr = np.asarray(linear, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("linear array must be HxWx3")
    srgb = to_srgb(np.clip(arr, 0.0, 1.0))
    u8 = (srgb * 255.0).round().astype(np.uint8)
    h, w = u8.shape[:2]
    bytes_per_line = u8.strides[0]
    return QImage(u8.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def load_linear_image(path: str) -> Optional[np.ndarray]:
    """
    Load an image as a float32 linear RGB array in [0,1].
    Supports RAW via rawpy when available.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in RAW_EXTENSIONS and rawpy is not None:
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    output_bps=8,
                    no_auto_bright=True,
                    gamma=(2.2, 4.5),
                )
            arr = rgb.astype(np.float32) / 255.0
            return to_linear(arr)
        except Exception as exc:
            print(f"[RAW] Failed to decode {path}: {exc}")
            return None

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return to_linear(arr)
    except Exception:
        return None
