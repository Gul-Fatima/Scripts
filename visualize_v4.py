import os
import sys
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Try Datumaro (optional)
try:
    from datumaro.components.dataset import Dataset
    DATUMARO_AVAILABLE = True
except:
    DATUMARO_AVAILABLE = False


# ============================================================
# DATA LOADING UTILITIES
# ============================================================

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_images(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(root, f))
    return files


def load_image_cv(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# MATPLOTLIB IMAGE PANEL
# ============================================================

class ImagePanel(QWidget):
    """
    One tile: metadata ABOVE image, image + boxes below
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(4)

        # Metadata label (above image)
        self.meta_label = QLabel("")
        self.meta_label.setAlignment(Qt.AlignLeft)
        self.meta_label.setStyleSheet(
            "QLabel { background: #f2f2f2; padding: 4px; font-size: 10pt; }"
        )

        # Matplotlib canvas
        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")

        self.layout.addWidget(self.meta_label)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def clear(self):
        self.meta_label.setText("")
        self.ax.clear()
        self.ax.axis("off")
        self.canvas.draw()

    def show_image(self, img, title="", annotations=None):
        self.ax.clear()
        self.ax.axis("off")

        if img is None:
            self.meta_label.setText("Failed to load image")
            self.ax.text(0.5, 0.5, "NO IMAGE", ha="center", va="center", color="red")
            self.canvas.draw()
            return

        # Show image
        self.ax.imshow(img)

        # Draw annotations if any
        class_summary = {}
        if annotations:
            for ann in annotations:
                x1, y1, x2, y2 = ann["bbox"]
                label = ann.get("label", "obj")
                color = ann.get("color", (1, 0, 0))

                rect = matplotlib.patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
                )
                self.ax.add_patch(rect)

                class_summary[label] = class_summary.get(label, 0) + 1

        # Metadata text (above image)
        if class_summary:
            summary = " | ".join([f"{k}:{v}" for k, v in class_summary.items()])
            meta = f"{title}   |   {summary}"
        else:
            meta = f"{title}   |   No annotations"

        self.meta_label.setText(meta)
        self.canvas.draw()


# ============================================================
# MAIN VIEWER
# ============================================================

class DatasetViewer(QMainWindow):
    def __init__(self, dataset_path, batch_size=4):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = min(batch_size, 4)

        self.setWindowTitle("Dataset Visualization Tool")
        self.resize(1200, 800)

        # Try loading dataset with Datumaro first
        self.items = []
        self.use_datumaro = False

        if DATUMARO_AVAILABLE:
            try:
                self.dataset = Dataset.import_from(dataset_path, "yolo")
                subsets = list(self.dataset.subsets())
                if subsets:
                    subset = subsets[0]
                    self.items = list(self.dataset.get_subset(subset))
                    self.use_datumaro = True
            except:
                self.use_datumaro = False

        # Fallback: raw directory
        if not self.use_datumaro:
            self.image_files = find_images(dataset_path)
            self.items = self.image_files

        if not self.items:
            QMessageBox.critical(self, "Error", "No images found in dataset.")
            sys.exit(1)

        # =========================
        # CENTRAL WIDGET
        # =========================
        central = QWidget()
        main_layout = QVBoxLayout()

        # Image grid (2x2)
        grid_layout = QHBoxLayout()

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        self.panels = []
        for _ in range(2):
            panel = ImagePanel()
            left_col.addWidget(panel)
            self.panels.append(panel)

        for _ in range(2):
            panel = ImagePanel()
            right_col.addWidget(panel)
            self.panels.append(panel)

        grid_layout.addLayout(left_col)
        grid_layout.addLayout(right_col)

        # =========================
        # BOTTOM CONTROLS
        # =========================
        control_layout = QHBoxLayout()

        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setStyleSheet("QPushButton { font-size: 12pt; padding: 8px; }")
        self.btn_quit.clicked.connect(self.close)

        self.btn_refresh = QPushButton("RESHUFFLE")
        self.btn_refresh.setStyleSheet("QPushButton { font-size: 12pt; padding: 8px; }")
        self.btn_refresh.clicked.connect(self.show_batch)

        control_layout.addWidget(self.btn_quit)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_refresh)

        main_layout.addLayout(grid_layout)
        main_layout.addLayout(control_layout)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Show first batch
        self.show_batch()

    # ========================================================
    # DATA HANDLING
    # ========================================================

    def get_random_batch(self):
        n = min(self.batch_size, len(self.items))
        return random.sample(self.items, n)

    def load_item(self, item):
        """
        Returns: (image, title, annotations)
        annotations: list of {bbox: (x1,y1,x2,y2), label: str, color: (r,g,b)}
        """
        if self.use_datumaro:
            return self.load_from_datumaro(item)
        else:
            return self.load_from_path(item)

    def load_from_path(self, path):
        img = load_image_cv(path)
        title = os.path.basename(path)
        return img, title, None

    def load_from_datumaro(self, item):
        # Load image
        img = None
        if hasattr(item, "media") and hasattr(item.media, "path"):
            img = load_image_cv(item.media.path)

        title = str(item.id)
        annotations = []

        if hasattr(item, "annotations"):
            for ann in item.annotations:
                if hasattr(ann, "points") and len(ann.points) >= 4:
                    x1, y1, x2, y2 = ann.points[0], ann.points[1], ann.points[2], ann.points[3]
                    label = str(ann.label) if hasattr(ann, "label") else "obj"
                    annotations.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "color": (1, 0, 0)
                    })

        return img, title, annotations

    # ========================================================
    # DISPLAY
    # ========================================================

    def show_batch(self):
        batch = self.get_random_batch()

        # Clear all panels
        for panel in self.panels:
            panel.clear()

        # Display items
        for panel, item in zip(self.panels, batch):
            img, title, annotations = self.load_item(item)
            panel.show_image(img, title=title, annotations=annotations)

    def closeEvent(self, event):
        # Clean shutdown
        QApplication.quit()
        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    DATASET_PATH = r"/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    BATCH_SIZE = 4

    app = QApplication(sys.argv)
    viewer = DatasetViewer(DATASET_PATH, BATCH_SIZE)
    viewer.show()
    sys.exit(app.exec_())
