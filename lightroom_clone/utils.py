import os
from typing import Optional

import numpy as np

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
