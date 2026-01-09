"""
Simple Dataset Visualization & Review Tool
Fixed Datumaro Visualizer usage with robust rendering
"""

import sys
import numpy as np
from PIL import Image

from datumaro.components.dataset import Dataset
from datumaro.components.visualizer import Visualizer

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt

# ============================================================
# SIMPLE DATASET MANAGER
# ============================================================

class DatasetManager:
    def __init__(self, dataset_path, dataset_format):
        self.dataset = Dataset.import_from(dataset_path, dataset_format)
        self.visualizer = Visualizer(self.dataset)
        self.items = list(self.dataset)
        
    def get_random_batch(self, n=4):
        if not self.items:
            return []
        indices = np.random.choice(len(self.items), min(n, len(self.items)), replace=False)
        return [self.items[i] for i in indices]


# ============================================================
# SIMPLE RENDERING THREAD
# ============================================================

class RenderThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, items, visualizer):
        super().__init__()
        self.items = items
        self.visualizer = visualizer
        
    def run(self):
        try:
            images = []

            for idx, item in enumerate(self.items):
                try:
                    # Render item individually
                    fig, ax = plt.subplots(figsize=(6, 4))
                    self.visualizer.draw_item(item, ax)
                    ax.axis('off')
                    fig.tight_layout(pad=0)

                    # Convert to PIL safely
                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()
                    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    img_array = img_array.reshape(h, w, 4)[:, :, :3]  # RGB only

                    pil_img = Image.fromarray(img_array)
                    images.append(pil_img)
                    plt.close(fig)
                except Exception as e:
                    # fallback: try to use raw image
                    try:
                        if hasattr(item, 'image') and item.image is not None:
                            if hasattr(item.image, 'data'):
                                img_data = item.image.data
                                if img_data is not None:
                                    pil_img = Image.fromarray(img_data)
                                    images.append(pil_img)
                                    continue
                            elif hasattr(item.image, 'path'):
                                import cv2
                                img = cv2.imread(item.image.path)
                                if img is not None:
                                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                    images.append(pil_img)
                                    continue
                        # If everything fails
                        images.append(Image.new('RGB', (400, 300), color=(255, 0, 0)))
                    except:
                        images.append(Image.new('RGB', (400, 300), color=(255, 0, 0)))

            self.finished.emit(images)

        except Exception as e:
            import traceback
            self.error.emit(f"Rendering failed: {str(e)}\n{traceback.format_exc()}")


# ============================================================
# MAIN WINDOW - SIMPLE
# ============================================================

class ReviewerWindow(QMainWindow):
    def __init__(self, dataset_path, dataset_format):
        super().__init__()
        
        # Setup
        self.manager = DatasetManager(dataset_path, dataset_format)
        self.batch_size = 4
        self.image_labels = []
        self.render_thread = None
        
        # UI
        self.setWindowTitle(f"Dataset Reviewer - {dataset_format}")
        self.setMinimumSize(900, 600)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)
        
        # Image grid
        self.grid = QGridLayout()
        layout.addLayout(self.grid)
        self._setup_image_grid()
        
        # Buttons
        btn_layout = QVBoxLayout()
        
        self.shuffle_btn = QPushButton("🔀 Load New Batch (N, R, Space)")
        self.shuffle_btn.clicked.connect(self.load_batch)
        self.shuffle_btn.setMinimumHeight(40)
        self.shuffle_btn.setStyleSheet("font-weight: bold;")
        
        self.quit_btn = QPushButton("Quit (Q, Esc)")
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setMinimumHeight(40)
        
        btn_layout.addWidget(self.shuffle_btn)
        btn_layout.addWidget(self.quit_btn)
        layout.addLayout(btn_layout)
        
        # Load first batch
        self.load_batch()
        
    def _setup_image_grid(self):
        """Create 2x2 image grid."""
        for i in range(2):
            for j in range(2):
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #ccc; 
                        background: #f8f8f8;
                        border-radius: 5px;
                    }
                    QLabel:hover {
                        border: 2px solid #0078d7;
                    }
                """)
                label.setMinimumSize(400, 300)
                self.grid.addWidget(label, i, j)
                self.image_labels.append(label)
    
    def load_batch(self):
        """Load and display a new batch."""
        self.shuffle_btn.setEnabled(False)
        self.shuffle_btn.setText("Loading...")
        
        items = self.manager.get_random_batch(self.batch_size)
        
        if not items:
            QMessageBox.warning(self, "No Items", "Dataset is empty or failed to load.")
            self.shuffle_btn.setEnabled(True)
            self.shuffle_btn.setText("🔀 Load New Batch (N, R, Space)")
            return
        
        for label in self.image_labels:
            label.clear()
            label.setText("Loading...")
        
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.terminate()
            
        self.render_thread = RenderThread(items, self.manager.visualizer)
        self.render_thread.finished.connect(self.on_images_ready)
        self.render_thread.error.connect(self.on_render_error)
        self.render_thread.start()
    
    def on_images_ready(self, images):
        for i, label in enumerate(self.image_labels):
            if i < len(images):
                img = images[i]
                qimage = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(pixmap)
            else:
                label.clear()
                label.setText("No Image")
        
        self.shuffle_btn.setEnabled(True)
        self.shuffle_btn.setText("🔀 Load New Batch (N, R, Space)")
    
    def on_render_error(self, error_msg):
        print(f"Render error: {error_msg}")
        for label in self.image_labels:
            label.clear()
            label.setText("Error")
        
        self.shuffle_btn.setEnabled(True)
        self.shuffle_btn.setText("🔀 Load New Batch (N, R, Space)")
        
        QMessageBox.warning(self, "Rendering Error", 
                          "Failed to render images. Check console for details.")
    
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_N, Qt.Key_R, Qt.Key_Space):
            self.load_batch()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
    
    def closeEvent(self, event):
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.terminate()
            self.render_thread.wait()
        event.accept()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"
    
    if len(sys.argv) > 1:
        DATASET_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        DATASET_FORMAT = sys.argv[2]
    
    app = QApplication(sys.argv)
    
    try:
        window = ReviewerWindow(DATASET_PATH, DATASET_FORMAT)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", 
                           f"Failed to start application:\n{str(e)}")
        sys.exit(1)
