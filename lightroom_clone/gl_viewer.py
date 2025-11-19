from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QSurfaceFormat,
    QOpenGLContext,
    QOffscreenSurface,
)

try:
    from PySide6.QtGui import QOpenGLFunctions
    from PySide6.QtOpenGL import (
        QOpenGLBuffer,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLTexture,
    )
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except Exception:  # pragma: no cover
    QOpenGLWidget = None  # type: ignore
    QOpenGLFunctions = object  # type: ignore

import numpy as np

from .shaders import FRAGMENT_SHADER_SRC, VERTEX_SHADER_SRC

GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_COLOR_BUFFER_BIT = 0x00004000


if QOpenGLWidget is None:  # pragma: no cover
    GLImageView = None
else:

    class GLImageView(QOpenGLWidget, QOpenGLFunctions):
        """OpenGL preview widget with graceful CPU fallback."""

        zoomChanged = Signal(float)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMouseTracking(True)

            self._program: QOpenGLShaderProgram | None = None
            self._vbo: QOpenGLBuffer | None = None
            self._texture: QOpenGLTexture | None = None
            self._pending_image: QImage | None = None
            self._params: dict = self._default_params()
            self._image_size = QSize()
            self._zoom_factor = 1.0
            self._fit_mode = True

            self._fallback_pixmap = QPixmap()
            self._gl_ready = False
            self._init_error = ""

        def sizeHint(self):
            return QSize(600, 400)

        def has_image(self) -> bool:
            return (
                self._gl_ready and self._texture is not None and self._texture.isCreated()
            ) or (not self._fallback_pixmap.isNull())

        def clear_image(self):
            self._pending_image = None
            self._fallback_pixmap = QPixmap()
            self._image_size = QSize()
            if self._texture and self._texture.isCreated():
                self.makeCurrent()
                self._texture.destroy()
                self.doneCurrent()
            self._texture = None
            self._zoom_factor = 1.0
            self._fit_mode = True
            self.update()
            self.zoomChanged.emit(1.0)

        def set_image(self, image: QImage):
            if image is None or image.isNull():
                self.clear_image()
                return
            self._pending_image = image
            self._fallback_pixmap = QPixmap.fromImage(image)
            self._reset_view()
            if self._gl_ready:
                self.makeCurrent()
                self._upload_pending_image()
                self.doneCurrent()
            self.update()

        def update_cpu_pixmap(self, pixmap: QPixmap | None):
            if pixmap is None or pixmap.isNull():
                return
            self._fallback_pixmap = QPixmap(pixmap)
            self._pending_image = pixmap.toImage()
            if self._gl_ready:
                self.makeCurrent()
                self._upload_pending_image()
                self.doneCurrent()
            self.update()

        def set_params(self, params: dict):
            self._params = dict(params)
            self.update()

        def set_manual_zoom(self, factor: float) -> float:
            if not self.has_image():
                return self._zoom_factor
            factor = max(0.05, min(12.0, factor))
            self._fit_mode = False
            self._zoom_factor = factor
            self.update()
            self.zoomChanged.emit(self._zoom_factor)
            return self._zoom_factor

        def fit_to_window(self) -> float:
            if not self.has_image():
                return self._zoom_factor
            self._fit_mode = True
            self._zoom_factor = self._compute_fit_zoom()
            self.update()
            self.zoomChanged.emit(self._zoom_factor)
            return self._zoom_factor

        def current_zoom(self) -> float:
            return self._zoom_factor

        def capture_image(self, max_side: int | None = None) -> QImage:
            if self._gl_ready and self.context() is not None:
                img = self.grabFramebuffer()
            else:
                img = self._fallback_pixmap.toImage()
            if max_side and max(img.width(), img.height()) > max_side:
                img = img.scaled(
                    max_side,
                    max_side,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            return img

        # ----- Qt events -----

        def initializeGL(self):
            try:
                self.initializeOpenGLFunctions()
                self._init_program()
                self._init_geometry()
                self._gl_ready = True
                if self._pending_image is not None:
                    self._upload_pending_image()
            except Exception as exc:  # pragma: no cover
                self._gl_ready = False
                self._init_error = str(exc)
                print("[OpenGL] initialization failed:", exc)

        def resizeGL(self, w: int, h: int):
            if self._fit_mode and self.has_image():
                self._zoom_factor = self._compute_fit_zoom()
                self.zoomChanged.emit(self._zoom_factor)

        def paintGL(self):
            if not self._gl_ready or self._texture is None or not self._texture.isCreated():
                self._paint_fallback()
                return

            self.glViewport(0, 0, max(1, self.width()), max(1, self.height()))
            self.glClearColor(0.08, 0.08, 0.09, 1.0)
            self.glClear(GL_COLOR_BUFFER_BIT)

            self._program.bind()
            self._vbo.bind()

            stride = 4 * np.dtype(np.float32).itemsize
            tex_offset = 2 * np.dtype(np.float32).itemsize
            self._program.enableAttributeArray(0)
            self._program.setAttributeBuffer(0, GL_FLOAT, 0, 2, stride)
            self._program.enableAttributeArray(1)
            self._program.setAttributeBuffer(1, GL_FLOAT, tex_offset, 2, stride)

            self._apply_transform_uniforms()
            self._apply_param_uniforms()

            self._texture.bind(0)
            self._program.setUniformValue("u_image", 0)

            self.glDrawArrays(GL_TRIANGLES, 0, 6)

            self._texture.release(0)
            self._vbo.release()
            self._program.release()

        def paintEvent(self, event):
            if self._gl_ready:
                super().paintEvent(event)
            else:
                self._paint_fallback()

        def wheelEvent(self, event):
            if not self.has_image():
                event.ignore()
                return
            delta = event.angleDelta().y()
            if delta == 0:
                event.ignore()
                return
            factor = 1.0 + abs(delta) / 480.0
            if delta < 0:
                factor = 1.0 / factor
            self.set_manual_zoom(self._zoom_factor * factor)
            event.accept()

        # ----- internal helpers -----

        def _paint_fallback(self):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(8, 8, 9))
            if self._fallback_pixmap.isNull():
                text = self._init_error or "No Image Loaded"
                painter.setPen(QPen(QColor(200, 200, 200)))
                painter.drawText(self.rect(), Qt.AlignCenter, text)
            else:
                scaled = self._fallback_pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                painter.drawPixmap(x, y, scaled)
            painter.end()

        def _init_program(self):
            program = QOpenGLShaderProgram(self.context())
            if not program.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX_SHADER_SRC):
                raise RuntimeError("Failed to compile vertex shader")
            if not program.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAGMENT_SHADER_SRC):
                raise RuntimeError("Failed to compile fragment shader")
            if not program.link():
                raise RuntimeError("Failed to link shader program")
            self._program = program

        def _init_geometry(self):
            vertices = np.array(
                [
                    -1.0, -1.0, 0.0, 0.0,
                    1.0, -1.0, 1.0, 0.0,
                    1.0,  1.0, 1.0, 1.0,
                    -1.0, -1.0, 0.0, 0.0,
                    1.0,  1.0, 1.0, 1.0,
                    -1.0,  1.0, 0.0, 1.0,
                ],
                dtype=np.float32,
            )
            self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            if not self._vbo.create():
                raise RuntimeError("Failed to create VBO")
            self._vbo.bind()
            self._vbo.allocate(vertices.tobytes(), vertices.nbytes)
            self._vbo.release()

        def _upload_pending_image(self):
            if self._pending_image is None:
                return
            img = self._pending_image.convertToFormat(QImage.Format_RGBA8888)
            img = img.mirrored(False, True)

            if self._texture is None:
                self._texture = QOpenGLTexture(QOpenGLTexture.Target2D)
            elif self._texture.isCreated():
                self._texture.destroy()

            if not self._texture.create():
                raise RuntimeError("Failed to create GL texture")

            self._texture.setMinificationFilter(QOpenGLTexture.Linear)
            self._texture.setMagnificationFilter(QOpenGLTexture.Linear)
            self._texture.setWrapMode(QOpenGLTexture.ClampToEdge)
            self._texture.bind()
            self._texture.setData(img)
            self._texture.release()

            self._image_size = img.size()
            self._pending_image = None

        def _compute_fit_zoom(self) -> float:
            if self.width() <= 0 or self.height() <= 0 or self._image_size.isEmpty():
                return self._zoom_factor
            return min(
                self.width() / self._image_size.width(),
                self.height() / self._image_size.height(),
            )

        def _reset_view(self):
            if self._fit_mode:
                self._zoom_factor = self._compute_fit_zoom()
            else:
                self._zoom_factor = max(0.05, min(12.0, self._zoom_factor))
            self.zoomChanged.emit(self._zoom_factor)

        def _apply_transform_uniforms(self):
            if not self._program:
                return
            w = max(1, self.width())
            h = max(1, self.height())
            img_w = max(1, self._image_size.width())
            img_h = max(1, self._image_size.height())

            scale_x = (img_w * self._zoom_factor) / w
            scale_y = (img_h * self._zoom_factor) / h

            self._program.setUniformValue("u_scale", scale_x, scale_y)
            self._program.setUniformValue("u_pan", 0.0, 0.0)

        def _apply_param_uniforms(self):
            if not self._program:
                return

            def tone_strength(name: str, scale: float) -> float:
                value = self._params.get(name, 128)
                return (value - 128.0) / 128.0 * scale

            uniforms = {
                "u_temperature": (self._params.get("temperature", 128) - 128.0) / 128.0 * 0.5,
                "u_tint": (self._params.get("tint", 128) - 128.0) / 128.0 * 0.5,
                "u_exposure": (self._params.get("exposure", 128) - 128.0) / 128.0 * 2.0,
                "u_contrast": self._params.get("contrast", 128) / 128.0,
                "u_saturation": self._params.get("saturation", 128) / 128.0,
                "u_vibrance": (self._params.get("vibrance", 128) - 128.0) / 128.0 * 0.75,
                "u_highlights": tone_strength("highlights", 0.6),
                "u_shadows": tone_strength("shadows", 0.6),
                "u_whites": tone_strength("whites", 0.8),
                "u_blacks": tone_strength("blacks", 0.8),
            }

            for name, value in uniforms.items():
                self._program.setUniformValue(name, float(value))

        def _default_params(self) -> dict:
            return {
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

        @staticmethod
        def is_supported() -> bool:
            try:
                ctx = QOpenGLContext()
                fmt = QSurfaceFormat.defaultFormat()
                ctx.setFormat(fmt)
                if not ctx.create():
                    return False
                surface = QOffscreenSurface()
                surface.setFormat(fmt)
                surface.create()
                ok = ctx.makeCurrent(surface)
                ctx.doneCurrent()
                surface.destroy()
                return ok
            except Exception:
                return False
