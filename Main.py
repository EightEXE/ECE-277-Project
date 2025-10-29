from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidget, QWidgetItem, QListWidgetItem, QScrollArea, QFileDialog, QStatusBar, QListView, QSizePolicy, QLayout, QAbstractItemView, QStyle, QDockWidget, QSlider, QHBoxLayout
from PySide6.QtCore import Qt, QSize, QRunnable, QThreadPool, Signal, QObject, QPoint
from PySide6.QtGui import QPixmap, QIcon, QImageReader, QFontMetrics
import sys
from qt_material import apply_stylesheet
import os

IMage_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LightRoom")
        self.resize(1200, 800)
        self.setMinimumSize(600, 400)
        central = QWidget(self)
        self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central)
        self.vbox.setContentsMargins(8,8,8,8)
        self.vbox.setSpacing(8)


        #large image aera
        self.image_display = QLabel("No Image Loaded")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumHeight(400)
        self.image_display.setStyleSheet("border: 1px solid gray;")
        self.vbox.addWidget(self.image_display, 1)
        self.vbox.setSizeConstraint(QLayout.SetDefaultConstraint)

        self.thumbs = QListWidget()
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(100, 100))
        self.thumbs.setResizeMode(QListWidget.Adjust)
        self.thumbs.setFlow(QListView.LeftToRight)
        self.thumbs.setWrapping(False)
        self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setSpacing(10)
        self.thumbs.setUniformItemSizes(True)
        self.thumbs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.thumbs.setWrapping(False)
        self.thumbs.setFixedHeight(130)
        self.thumbs.itemClicked.connect(self.display_image)
        self.vbox.addWidget(self.thumbs, 0)

        self.thumbs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.thumbs.setMinimumWidth(0)
        self.thumbs.setMinimumHeight(self.thumbs.height())

        self.thumbs.horizontalScrollBar().valueChanged.connect(lambda _: self._ensure_visible_thumbs())

        icon_w, icon_h = self.thumbs.iconSize().width(), self.thumbs.iconSize().height()
        fm = QFontMetrics(self.thumbs.font())
        text_h = fm.height()
        pad_h = 8
        cell_w = icon_w + 16
        cell_h = icon_h + text_h + pad_h

        self.thumbs.setGridSize(QSize(cell_w, cell_h))

        sb_h = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        self.thumbs.setFixedHeight(cell_h + sb_h + 2)

        self.thumbs.setResizeMode(QListView.Fixed)
        self.thumbs.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.thumbs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.thumbs.setMinimumWidth(0)

        self.setStatusBar(QStatusBar(self))

        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(min(8, (os.cpu_count() or 4)))
        self._icon_cache = {}
        self._loading = set()
        self._item_for_path = {}

        file_menu = self.menuBar().addMenu("&File")
        self.act_open = file_menu.addAction("&Open Folder")
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.triggered.connect(self.open_folder)
        self._placeholder_icon = QIcon(QPixmap(icon_w, icon_h))
        self._current_pixmap = None

         # saturation slider
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 255)
        self.saturation_slider.setValue(128)
        self.saturation_value = QLabel("Saturation: 128")
        row = QHBoxLayout()
        row.addWidget(QLabel("Saturation"))
        row.addWidget(self.saturation_slider, 1)
        row.addWidget(self.saturation_value)

        

    #sidebar dock
        sidebar_dock = QDockWidget("Sidebar", self)
        sidebar_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        sidebar_dock.setAllowedAreas(Qt.RightDockWidgetArea)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)

        sidebar_layout.addLayout(row)
        sidebar_layout.addStretch(1)
        sidebar_dock.setWidget(sidebar_content)

        self.addDockWidget(Qt.RightDockWidgetArea, sidebar_dock)


   


    def get_natural_pixmap(self, path: str) -> QPixmap:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        image = reader.read()
        pm = QPixmap(path)
        return pm.fromImage(image)

    def display_image(self, item):
        path = item.data(Qt.UserRole)
        pm = QPixmap(path)
        if pm.isNull():
            return
        self._current_pixmap = pm
        self._rescale_preview()
    
    def _rescale_preview(self):
        if not self._current_pixmap:
            return
        scaled = self._current_pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_display.setPixmap(scaled)
        self.image_display.setText("")

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", os.path.expanduser("~"))
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
        row_y = vp.height() // 2
        first = self.thumbs.indexAt(QPoint(0, row_y)).row()
        last = self.thumbs.indexAt(QPoint(vp.width()-1, row_y)).row()

        if first < 0:
            first = 0
        if last < 0:
            approx_per_view = max(1, vp.width() // (self.thumbs.iconSize().width() + self.thumbs.spacing()))
            last = min(self.thumbs.count() -1, first + approx_per_view + 4)
            
        first = max(0, first - 4)
        last = min(self.thumbs.count() -1, last + 8)

        for i in range(first, last + 1):
            it = self.thumbs.item(i)
            self._queue_thumb(it.data(Qt.UserRole))

    def load_folder(self, folder_path: str):
        self.thumbs.clear()
        self._icon_cache.clear()
        self._loading.clear()
        self._item_for_path.clear()
        count = 0
        for name in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in IMage_EXTENSIONS:
                continue

            item = QListWidgetItem(self._placeholder_icon, name)
            item.setData(Qt.UserRole, path)
            self.thumbs.addItem(item)
            self._item_for_path[path] = item
            count += 1
        
        self.statusBar().showMessage(f"Loading {count} images from {folder_path}")

        if count:
            self.thumbs.setCurrentRow(0)
            self.display_image(self.thumbs.item(0))
        self._ensure_visible_thumbs()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, '_current_pixmap', None) is not None:
            self._rescale_preview()
        self._ensure_visible_thumbs()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_cyan.xml')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())