from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import colour
    _COLOUR_AVAILABLE = True
except Exception:  # pragma: no cover - runtime safeguard
    colour = None
    _COLOUR_AVAILABLE = False
    print("[Color] colour-science not installed; falling back to manual sRGB transfer functions.")

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    rawpy = None
    RAWPY_AVAILABLE = False

try:
    print("[RAW] rawpy available:", RAWPY_AVAILABLE, getattr(rawpy, "__version__", None))
except Exception:
    pass

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


_RAW_WARNING_SHOWN = False


def qimage_to_numpy(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w = img.width()
    h = img.height()
    bpl = img.bytesPerLine()
    buf = img.bits().tobytes()
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size < h * bpl:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = arr.reshape((h, bpl))
    arr = arr[:, : w * 3]
    return arr.reshape((h, w, 3)).copy()


def numpy_to_qimage_u8(rgb: np.ndarray) -> QImage:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected HxWx3 array")
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def _warn_raw_missing_once():
    global _RAW_WARNING_SHOWN
    if _RAW_WARNING_SHOWN:
        return
    _RAW_WARNING_SHOWN = True
    print("[RAW] rawpy not installed; RAW files unsupported.")


def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB-encoded float32 [0,1] to linear sRGB using colour-science.
    Falls back to manual transfer if colour is unavailable.
    """
    arr = np.asarray(img_srgb, dtype=np.float32)
    if _COLOUR_AVAILABLE and colour is not None:
        try:
            decoded = colour.cctf_decoding(arr, function="sRGB")
            return np.clip(np.asarray(decoded, dtype=np.float32), 0.0, 1.0)
        except Exception:
            pass

    linear = np.where(
        arr <= 0.04045,
        arr / 12.92,
        np.power((arr + 0.055) / 1.055, 2.4),
    )
    return np.clip(linear, 0.0, 1.0)


def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    """
    Convert linear sRGB float32 [0,1] to sRGB-encoded float32 using colour-science.
    Falls back to manual transfer if colour is unavailable.
    """
    arr = np.asarray(img_lin, dtype=np.float32)
    if _COLOUR_AVAILABLE and colour is not None:
        try:
            encoded = colour.cctf_encoding(arr, function="sRGB")
            return np.clip(np.asarray(encoded, dtype=np.float32), 0.0, 1.0)
        except Exception:
            pass

    srgb = np.where(
        arr <= 0.0031308,
        arr * 12.92,
        1.055 * np.power(arr, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def load_raw_thumbnail(path: str) -> Optional[np.ndarray]:
    if not RAWPY_AVAILABLE:
        _warn_raw_missing_once()
        return None
    try:
        with rawpy.imread(path) as raw:
            try:
                thumb = raw.extract_thumb()
            except Exception:
                rgb = raw.postprocess(
                    output_bps=8,
                    half_size=True,
                    no_auto_bright=True,
                    gamma=(2.2, 4.5),
                )
                return rgb

            if thumb.format == rawpy.ThumbFormat.JPEG:
                qimg = QImage.fromData(thumb.data, "JPG")
                if qimg.isNull():
                    return None
                return qimage_to_numpy(qimg)
            else:
                return thumb.data
    except Exception as e:
        print(f"[RAW] Thumbnail load failed for {path}: {e}")
        return None


def load_raw_preview(path: str, max_size: int = 2048) -> Optional[np.ndarray]:
    """
    Fast preview decode for RAW files.
    Returns uint8 RGB in [0,255], gamma-encoded sRGB, small resolution.
    """
    if not RAWPY_AVAILABLE:
        _warn_raw_missing_once()
        return None
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                output_bps=8,
                half_size=True,
                no_auto_bright=True,
                gamma=(2.2, 4.5),
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
            )
    except Exception as e:
        print(f"[RAW] Preview load failed for {path}: {e}")
        return None

    h, w, _ = rgb.shape
    scale = min(1.0, max_size / max(h, w)) if max(h, w) > 0 else 1.0
    if scale < 0.999:
        try:
            import cv2  # type: ignore
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            try:
                pil = Image.fromarray(rgb, mode="RGB")
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                pil = pil.resize((new_w, new_h), resample=Image.LANCZOS)
                rgb = np.asarray(pil, dtype=np.uint8)
            except Exception:
                pass
    return rgb


def load_raw_full(path: str) -> Optional[np.ndarray]:
    """Legacy alias for load_raw_full_linear."""
    return load_raw_full_linear(path)


def load_raw_full_linear(path: str) -> Optional[np.ndarray]:
    """
    Full-resolution RAW decode for the editing pipeline.
    Returns float32 RGB in [0,1], linear sRGB working space.
    """
    if not RAWPY_AVAILABLE:
        _warn_raw_missing_once()
        return None
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                output_bps=16,
                no_auto_bright=True,
                gamma=(1.0, 1.0),
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
            )
    except Exception as e:
        print(f"[RAW] Full load failed for {path}: {e}")
        return None
    rgb_float = rgb.astype(np.float32) / 65535.0
    return rgb_float


def load_image_thumbnail_qimage(path: str) -> Optional[QImage]:
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        rgb = load_raw_preview(path, max_size=512)
        if rgb is None:
            rgb = load_raw_thumbnail(path)
        if rgb is None:
            return None
        return numpy_to_qimage_u8(rgb)
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    img = reader.read()
    if img.isNull():
        return None
    return img


def load_image_preview_qimage(path: str) -> Optional[QImage]:
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        rgb = load_raw_preview(path)
        if rgb is None:
            return None
        return numpy_to_qimage_u8(rgb)
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    img = reader.read()
    if img.isNull():
        print("[IMG] Qt failed to load:", path, reader.errorString())
        return None
    return img


def load_qimage_any(path: str) -> Optional[QImage]:
    """Backwards-compatible loader used by thumbnail worker."""
    return load_image_preview_qimage(path)


def to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert gamma-encoded sRGB [0,1] into linear float32."""
    return srgb_to_linear(rgb)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1] into gamma-encoded sRGB float32."""
    return linear_to_srgb(linear)


def qimage_from_linear(linear: np.ndarray) -> QImage:
    """Create an RGB888 QImage from a [0,1] linear float32 array."""
    arr = np.asarray(linear, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("linear array must be HxWx3")
    srgb = linear_to_srgb(np.clip(arr, 0.0, 1.0))
    u8 = (srgb * 255.0).round().astype(np.uint8)
    h, w = u8.shape[:2]
    bytes_per_line = u8.strides[0]
    return QImage(u8.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def load_image_preview_linear(path: str, max_size: int = 2048) -> Optional[np.ndarray]:
    """
    Load a preview image as linear sRGB float32.
    RAW paths use load_raw_preview; non-RAW paths use QImageReader (auto-rotated).
    """
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        rgb = load_raw_preview(path, max_size=max_size)
        if rgb is None:
            return None
        return srgb_to_linear(rgb.astype(np.float32) / 255.0)

    reader = QImageReader(path)
    reader.setAutoTransform(True)
    qimg = reader.read()
    if qimg.isNull():
        print("[IMG] Qt failed to load:", path, reader.errorString())
        return None
    rgb = qimage_to_numpy(qimg)
    h, w, _ = rgb.shape
    scale = min(1.0, max_size / max(h, w)) if max(h, w) > 0 else 1.0
    if scale < 0.999:
        try:
            pil = Image.fromarray(rgb, mode="RGB")
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            pil = pil.resize((new_w, new_h), resample=Image.LANCZOS)
            rgb = np.asarray(pil, dtype=np.uint8)
        except Exception:
            pass
    return srgb_to_linear(rgb.astype(np.float32) / 255.0)


def load_image_full_linear(path: str) -> Optional[np.ndarray]:
    """
    Load a full-resolution image as float32 linear sRGB.
    RAW files go through rawpy; non-RAW use QImageReader to honor EXIF orientation.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTENSIONS:
        return load_raw_full_linear(path)

    reader = QImageReader(path)
    reader.setAutoTransform(True)
    qimg = reader.read()
    if qimg.isNull():
        print("[IMG] Qt failed to load:", path, reader.errorString())
        return None
    rgb = qimage_to_numpy(qimg).astype(np.float32) / 255.0
    return srgb_to_linear(rgb)


def load_linear_image(path: str) -> Optional[np.ndarray]:
    """
    Legacy helper kept for backwards compatibility.
    """
    return load_image_full_linear(path)
