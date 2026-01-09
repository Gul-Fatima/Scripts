"""
Simple Dataset Visualization & Review Tool
Fixed Datumaro Visualizer usage
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
            
            # Use Datumaro's visualizer to create a gallery
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create a figure with subplots
            n_items = len(self.items)
            n_cols = min(2, n_items)  # Max 2 columns
            n_rows = (n_items + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8 * n_rows / 2))
            if n_items == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            # Plot each item
            for idx, (item, ax) in enumerate(zip(self.items, axes)):
                try:
                    # Draw the item using Datumaro's visualizer
                    self.visualizer.draw_item(item, ax)
                    ax.set_title(f"Item {idx}", fontsize=10)
                    ax.axis('off')
                except Exception as e:
                    # If draw_item fails, try to just show the image
                    try:
                        if hasattr(item, 'image') and item.image is not None:
                            if hasattr(item.image, 'data'):
                                img_data = item.image.data
                                if img_data is not None:
                                    ax.imshow(img_data)
                            elif hasattr(item.image, 'path'):
                                # Try to load from path
                                import cv2
                                img = cv2.imread(item.image.path)
                                if img is not None:
                                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        ax.set_title(f"Item {idx} (no annotations)", fontsize=10)
                        ax.axis('off')
                    except:
                        ax.text(0.5, 0.5, "Failed to load", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
            
            # Hide empty subplots
            for idx in range(len(self.items), len(axes)):
                axes[idx].axis('off')
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Convert the entire figure to an image
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Split the combined image into individual images
            width, height = img_array.shape[1], img_array.shape[0]
            
            if n_items > 0:
                # Calculate individual image dimensions
                img_height = height // n_rows
                img_width = width // n_cols
                
                for row in range(n_rows):
                    for col in range(n_cols):
                        idx = row * n_cols + col
                        if idx < n_items:
                            y_start = row * img_height
                            y_end = min((row + 1) * img_height, height)
                            x_start = col * img_width
                            x_end = min((col + 1) * img_width, width)
                            
                            sub_img = img_array[y_start:y_end, x_start:x_end]
                            images.append(Image.fromarray(sub_img))
            
            plt.close(fig)
            self.finished.emit(images[:n_items])  # Return only needed images
            
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
        
        # Shuffle button
        self.shuffle_btn = QPushButton("🔀 Load New Batch (N, R, Space)")
        self.shuffle_btn.clicked.connect(self.load_batch)
        self.shuffle_btn.setMinimumHeight(40)
        self.shuffle_btn.setStyleSheet("font-weight: bold;")
        
        # Quit button
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
        # Disable button during loading
        self.shuffle_btn.setEnabled(False)
        self.shuffle_btn.setText("Loading...")
        
        # Get items
        items = self.manager.get_random_batch(self.batch_size)
        
        if not items:
            QMessageBox.warning(self, "No Items", "Dataset is empty or failed to load.")
            self.shuffle_btn.setEnabled(True)
            self.shuffle_btn.setText("🔀 Load New Batch (N, R, Space)")
            return
        
        # Clear current images
        for label in self.image_labels:
            label.clear()
            label.setText("Loading...")
        
        # Start rendering thread
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.terminate()
            
        self.render_thread = RenderThread(items, self.manager.visualizer)
        self.render_thread.finished.connect(self.on_images_ready)
        self.render_thread.error.connect(self.on_render_error)
        self.render_thread.start()
    
    def on_images_ready(self, images):
        """Display rendered images."""
        for i, label in enumerate(self.image_labels):
            if i < len(images):
                # Convert PIL to QPixmap
                img = images[i]
                qimage = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(pixmap)
            else:
                label.clear()
                label.setText("No Image")
        
        # Re-enable button
        self.shuffle_btn.setEnabled(True)
        self.shuffle_btn.setText("🔀 Load New Batch (N, R, Space)")
    
    def on_render_error(self, error_msg):
        """Handle rendering errors."""
        print(f"Render error: {error_msg}")
        for label in self.image_labels:
            label.clear()
            label.setText("Error")
        
        self.shuffle_btn.setEnabled(True)
        self.shuffle_btn.setText("🔀 Load New Batch (N, R, Space)")
        
        QMessageBox.warning(self, "Rendering Error", 
                          f"Failed to render images. Check console for details.")
    
    def keyPressEvent(self, event):
        """Keyboard shortcuts."""
        if event.key() in (Qt.Key_N, Qt.Key_R, Qt.Key_Space):
            self.load_batch()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.terminate()
            self.render_thread.wait()
        event.accept()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"
    
    # Optional: Use command line arguments
    if len(sys.argv) > 1:
        DATASET_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        DATASET_FORMAT = sys.argv[2]
    
    # Run app
    app = QApplication(sys.argv)
    
    try:
        window = ReviewerWindow(DATASET_PATH, DATASET_FORMAT)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", 
                           f"Failed to start application:\n{str(e)}")
        sys.exit(1)
