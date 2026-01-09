"""
Dataset Visualization & Review Tool (PyQt Version)

Purpose:
--------
A professional-grade desktop application for reviewing annotated datasets.
Designed for smooth UX, clean window handling, and future extensibility.

Key Features:
-------------
- Multi-image grid display.
- GUI buttons + keyboard shortcuts.
- Single persistent window.
- Graceful application shutdown.
- Extendable architecture for filters, tagging, and logging.

Usage:
------
python visualize_pyqt.py

Dependencies:
-------------
- PyQt5 or PySide6
- datumaro
- numpy
- Pillow

Author:
-------
<Your Name / Team>
"""

import sys
import numpy as np
from PIL import Image
from datumaro.components.dataset import Dataset
from datumaro.components.visualizer import Visualizer

# Try PyQt5 first, fallback to PySide6
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QGridLayout
    )
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtCore import Qt
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QGridLayout
    )
    from PySide6.QtGui import QPixmap, QImage
    from PySide6.QtCore import Qt


# ============================================================
# DATA HANDLING
# ============================================================

class DatasetManager:
    """
    Responsible for loading the dataset and sampling batches.
    """

    def __init__(self, dataset_path: str, dataset_format: str, images_per_batch: int):
        self.dataset = Dataset.import_from(dataset_path, dataset_format)
        self.visualizer = Visualizer(self.dataset)
        self.images_per_batch = images_per_batch
        self.current_items = []

    def random_batch(self):
        """Return a random batch of dataset items."""
        self.current_items = self.visualizer.get_random_items(self.images_per_batch)
        return self.current_items


# ============================================================
# GUI COMPONENTS
# ============================================================

class ReviewerWindow(QMainWindow):
    """
    Main application window.
    Handles UI layout, user interactions, and dataset rendering.
    """

    def __init__(self, dataset_manager: DatasetManager, window_title: str):
        super().__init__()

        self.manager = dataset_manager
        self.setWindowTitle(window_title)
        self.setMinimumSize(1000, 700)

        self._init_ui()
        self.load_new_batch()

    def _init_ui(self):
        """Initialize all UI components and layouts."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)

        # Grid for images
        self.image_grid = QGridLayout()
        self.main_layout.addLayout(self.image_grid)

        self.image_labels = []
        for i in range(2):
            for j in range(2):
                label = QLabel("No Image")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid #999;")
                label.setMinimumSize(400, 300)
                self.image_grid.addWidget(label, i, j)
                self.image_labels.append(label)

        # Control buttons
        self._init_controls()

    def _init_controls(self):
        """Initialize control buttons and bind actions."""
        controls_layout = QHBoxLayout()

        self.prev_btn = QPushButton("⬅ Prev")
        self.shuffle_btn = QPushButton("🔀 Reshuffle")
        self.next_btn = QPushButton("Next ➡")
        self.quit_btn = QPushButton("❌ Quit")

        self.prev_btn.clicked.connect(self.load_new_batch)
        self.shuffle_btn.clicked.connect(self.load_new_batch)
        self.next_btn.clicked.connect(self.load_new_batch)
        self.quit_btn.clicked.connect(self.close)

        for btn in [self.prev_btn, self.shuffle_btn, self.next_btn, self.quit_btn]:
            controls_layout.addWidget(btn)

        self.main_layout.addLayout(controls_layout)

    # ========================================================
    # DATA RENDERING
    # ========================================================

    def load_new_batch(self):
        """Fetch a new random batch and update the display."""
        items = self.manager.random_batch()
        self._render_items(items)

    def _render_items(self, items):
        """
        Render dataset items into the grid.
        Converts Datumaro-rendered images into Qt-compatible images.
        """
        # Clear previous images
        for label in self.image_labels:
            label.clear()
            label.setText("")

        # Use Datumaro visualizer to render items
        figure = self.manager.visualizer.vis_gallery(items)

        # Convert Matplotlib figure to raw image
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(height, width, 3)

        image = Image.fromarray(img)
        qt_image = self._pil_to_qimage(image)
        pixmap = QPixmap.fromImage(qt_image)

        # Display same rendered grid in all slots (simpler, faster)
        for label in self.image_labels:
            label.setPixmap(
                pixmap.scaled(
                    label.width(),
                    label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    @staticmethod
    def _pil_to_qimage(pil_image: Image.Image) -> QImage:
        """Convert PIL Image to QImage."""
        rgb_image = pil_image.convert("RGB")
        w, h = rgb_image.size
        data = rgb_image.tobytes("raw", "RGB")
        return QImage(data, w, h, QImage.Format_RGB888)

    # ========================================================
    # KEYBOARD EVENTS
    # ========================================================

    def keyPressEvent(self, event):
        """
        Keyboard shortcuts for fast navigation.

        n / N      -> Next batch
        r / R      -> Reshuffle
        q / Esc    -> Quit
        """
        if event.key() in (Qt.Key_N,):
            self.load_new_batch()
        elif event.key() in (Qt.Key_R,):
            self.load_new_batch()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Application entry point.
    All configuration parameters are defined here.
    """

    # ---------------- CONFIGURATION ----------------

    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"
    IMAGES_PER_BATCH = 4
    WINDOW_TITLE = "Dataset Reviewer - YOLO Annotations"

    # ------------------------------------------------
    # DON'T MODIFY
    # ------------------------------------------------

    app = QApplication(sys.argv)

    dataset_manager = DatasetManager(
        dataset_path=DATASET_PATH,
        dataset_format=DATASET_FORMAT,
        images_per_batch=IMAGES_PER_BATCH
    )

    window = ReviewerWindow(
        dataset_manager=dataset_manager,
        window_title=WINDOW_TITLE
    )
    window.show()

    sys.exit(app.exec_())
