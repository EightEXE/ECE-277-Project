import os
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QPixmap

from lightroom_clone.constants import PROJECT_EXTENSION, LEGACY_PROJECT_EXTENSION
from lightroom_clone.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # Determine if we are auto-loading a project file
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    valid_exts = (PROJECT_EXTENSION, LEGACY_PROJECT_EXTENSION)
    loading_project = (
        arg_path is not None
        and arg_path.lower().endswith(valid_exts)
        and os.path.isfile(arg_path)
    )

    # Splash/progress always shown; progress steps depend on whether we load a project
    card_path = os.path.join(os.path.dirname(__file__), "card.png")
    pixmap = QPixmap(card_path) if os.path.isfile(card_path) else QPixmap()
    if not pixmap.isNull():
        # Scale down slightly to avoid an oversized splash on large assets
        scaled = pixmap.scaled(
            int(pixmap.width() * 0.6),
            int(pixmap.height() * 0.6),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        if not scaled.isNull():
            pixmap = scaled

    splash_widget = QWidget(None, Qt.SplashScreen | Qt.FramelessWindowHint)
    layout = QVBoxLayout(splash_widget)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)
    layout.setAlignment(Qt.AlignCenter)

    if not pixmap.isNull():
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)

    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setTextVisible(False)
    progress.setFixedWidth(
        max(200, int(pixmap.width() * 0.5)) if not pixmap.isNull() else 240
    )
    layout.addWidget(progress, alignment=Qt.AlignCenter)

    status = QLabel("Loading...")
    status.setAlignment(Qt.AlignCenter)
    layout.addWidget(status)

    splash_widget.adjustSize()
    if app.primaryScreen():
        geo = app.primaryScreen().availableGeometry()
        frame = splash_widget.frameGeometry()
        frame.moveCenter(geo.center())
        splash_widget.move(frame.topLeft())

    splash_widget.show()
    app.processEvents()

    def _update_progress(msg: str, value: int, delay: float = 0.1):
        if splash_widget is None:
            return
        if status is not None:
            status.setText(msg)
        if progress is not None:
            progress.setValue(value)
        app.processEvents()
        if delay > 0:
            time.sleep(delay)

    window = MainWindow()

    if loading_project:
        _update_progress("Reading project file...", 20)
        try:
            window.import_project_from_path(arg_path)
            _update_progress("Applying edits...", 70)
            _update_progress("Finalizing UI...", 90)
        except Exception as exc:
            print(f"Failed to auto-load project: {exc}")
            _update_progress("Failed to load project", 100, delay=0)
        else:
            _update_progress("Ready", 100, delay=0.1)
    else:
        _update_progress("Starting Gradience Studio...", 20)
        _update_progress("Initializing UI...", 50)
        _update_progress("Ready", 100, delay=0.05)

    window.show()

    if splash_widget is not None:
        splash_widget.hide()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
