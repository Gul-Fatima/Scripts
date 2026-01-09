"""
Enhanced Dataset Visualization & Review Tool (PyQt5 Version)

Purpose:
--------
A professional-grade desktop application for reviewing annotated datasets.
Features: Memory-efficient rendering, error handling, adjustable grid, and navigation.

Dependencies:
-------------
- PyQt5
- datumaro
- numpy
- Pillow
- matplotlib (with Agg backend for rendering)
"""

import sys
import os
import traceback
from typing import List, Optional
import numpy as np
from PIL import Image, ImageQt

# Use Agg backend for matplotlib to avoid memory issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datumaro.components.dataset import Dataset
from datumaro.components.visualizer import Visualizer

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox,
    QStatusBar, QSpinBox, QComboBox, QGroupBox, QCheckBox,
    QFileDialog, QShortcut
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QFont, QKeySequence,
    QPalette, QColor
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QThread, pyqtSlot


# ============================================================
# DATA HANDLING
# ============================================================

class DatasetManager:
    """Responsible for loading the dataset and sampling batches."""

    def __init__(self, dataset_path: str, dataset_format: str):
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.dataset: Optional[Dataset] = None
        self.visualizer: Optional[Visualizer] = None
        self.current_items = []
        self.all_items = []
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset with error handling."""
        try:
            print(f"Loading dataset from {self.dataset_path}...")
            self.dataset = Dataset.import_from(self.dataset_path, self.dataset_format)
            self.visualizer = Visualizer(self.dataset)
            self.all_items = list(self.dataset)
            print(f"Loaded {len(self.all_items)} items.")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def random_batch(self, batch_size: int) -> List:
        """Return a random batch of dataset items."""
        if not self.all_items:
            return []
        
        indices = np.random.choice(len(self.all_items), 
                                 min(batch_size, len(self.all_items)), 
                                 replace=False)
        self.current_items = [self.all_items[idx] for idx in indices]
        return self.current_items

    def get_item_image(self, item):
        """Extract image data from dataset item."""
        try:
            # Check if item has image data
            if hasattr(item, 'image') and item.image is not None:
                if hasattr(item.image, 'data'):
                    return item.image.data
                elif hasattr(item.image, 'path'):
                    return item.image.path
            return None
        except:
            return None


# ============================================================
# WORKER THREAD FOR IMAGE RENDERING
# ============================================================

class RenderingWorker(QThread):
    """Worker thread for rendering images to prevent UI freeze."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, items, visualizer, grid_size):
        super().__init__()
        self.items = items
        self.visualizer = visualizer
        self.grid_size = grid_size
        
    def run(self):
        """Render images in a separate thread."""
        try:
            rendered_images = []
            for item in self.items:
                # Create figure with Agg backend
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                # Render the item
                self.visualizer.vis_item(item, ax=ax)
                ax.axis('off')
                
                # Convert to numpy array
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(height, width, 3)
                
                # Convert to PIL Image
                pil_img = Image.fromarray(img_array)
                
                rendered_images.append(pil_img)
                
                # Clean up
                plt.close(fig)
                del fig, ax
            
            self.finished.emit(rendered_images)
            
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()


# ============================================================
# IMAGE DISPLAY WIDGET
# ============================================================

class ImageDisplayWidget(QLabel):
    """Custom QLabel for displaying images with annotations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            ImageDisplayWidget {
                border: 2px solid #ccc;
                border-radius: 4px;
                background-color: #f8f8f8;
            }
            ImageDisplayWidget:hover {
                border: 2px solid #0078d7;
            }
        """)
        self.setText("No Image")
        self.setFont(QFont("Arial", 12))
        
    def display_image(self, pil_image: Image.Image):
        """Display a PIL image in the widget."""
        if pil_image:
            # Convert PIL Image to QPixmap
            qt_image = ImageQt.ImageQt(pil_image)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale pixmap to fit widget while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.size() - QSize(20, 20),  # Padding
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.setPixmap(scaled_pixmap)
        else:
            self.clear()
            self.setText("No Image")
            
    def clear_display(self):
        """Clear the display."""
        self.clear()
        self.setText("No Image")


# ============================================================
# MAIN APPLICATION WINDOW
# ============================================================

class ReviewerWindow(QMainWindow):
    """Main application window with enhanced features."""
    
    def __init__(self, dataset_path: str, dataset_format: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.dataset_manager: Optional[DatasetManager] = None
        self.current_batch_size = 4
        self.image_widgets = []
        self.rendering_worker = None
        
        self._init_dataset()
        self._init_ui()
        self._setup_shortcuts()
        
    def _init_dataset(self):
        """Initialize dataset manager."""
        self.dataset_manager = DatasetManager(self.dataset_path, self.dataset_format)
        if self.dataset_manager.dataset is None:
            QMessageBox.critical(
                self, 
                "Dataset Error", 
                f"Failed to load dataset from:\n{self.dataset_path}\n\n"
                f"Format: {self.dataset_format}\n"
                f"Check if the path exists and format is correct."
            )
            sys.exit(1)
            
    def _init_ui(self):
        """Initialize all UI components."""
        self.setWindowTitle(f"Dataset Reviewer - {os.path.basename(self.dataset_path)}")
        self.setMinimumSize(1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create image display area
        self._create_image_display()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Loaded {len(self.dataset_manager.all_items)} items")
        
        # Load initial batch
        self.load_new_batch()
        
    def _create_toolbar(self):
        """Create toolbar with controls."""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_widget.setLayout(toolbar_layout)
        
        # Batch size control
        batch_group = QGroupBox("Batch Settings")
        batch_layout = QHBoxLayout()
        
        batch_size_label = QLabel("Images per batch:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 16)
        self.batch_size_spin.setValue(self.current_batch_size)
        self.batch_size_spin.valueChanged.connect(self.on_batch_size_changed)
        
        batch_layout.addWidget(batch_size_label)
        batch_layout.addWidget(self.batch_size_spin)
        batch_group.setLayout(batch_layout)
        
        # Display options
        display_group = QGroupBox("Display")
        display_layout = QHBoxLayout()
        
        self.show_annos_check = QCheckBox("Show Annotations")
        self.show_annos_check.setChecked(True)
        self.show_annos_check.stateChanged.connect(self.load_new_batch)
        
        display_layout.addWidget(self.show_annos_check)
        display_group.setLayout(display_layout)
        
        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()
        
        self.shuffle_btn = QPushButton("🔀 New Batch")
        self.shuffle_btn.clicked.connect(self.load_new_batch)
        self.shuffle_btn.setToolTip("Load new random batch (Shortcut: N, R, Space)")
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.load_previous_batch)
        self.prev_btn.setToolTip("Load previous batch (Shortcut: P, Left Arrow)")
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.load_next_batch)
        self.next_btn.setToolTip("Load next batch (Shortcut: Right Arrow)")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.shuffle_btn)
        nav_layout.addWidget(self.next_btn)
        nav_group.setLayout(nav_layout)
        
        # Add all groups to toolbar
        toolbar_layout.addWidget(batch_group)
        toolbar_layout.addWidget(display_group)
        toolbar_layout.addWidget(nav_group)
        
        # Add spacer and quit button
        toolbar_layout.addStretch()
        
        self.quit_btn = QPushButton("🚪 Quit")
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        toolbar_layout.addWidget(self.quit_btn)
        
        self.main_layout.addWidget(toolbar_widget)
        
    def _create_image_display(self):
        """Create the image grid display area."""
        # Container for the image grid
        self.image_container = QWidget()
        self.image_container_layout = QVBoxLayout()
        self.image_container.setLayout(self.image_container_layout)
        
        # Scroll area for many images
        from PyQt5.QtWidgets import QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_container)
        
        self.main_layout.addWidget(self.scroll_area, 1)  # 1 = stretch factor
        
        # Initial grid setup
        self.update_image_grid()
        
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Navigation
        QShortcut(QKeySequence("N"), self).activated.connect(self.load_new_batch)
        QShortcut(QKeySequence("R"), self).activated.connect(self.load_new_batch)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.load_new_batch)
        QShortcut(QKeySequence("P"), self).activated.connect(self.load_previous_batch)
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.load_previous_batch)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.load_next_batch)
        
        # Application
        QShortcut(QKeySequence("Q"), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        
        # Zoom
        QShortcut(QKeySequence("Ctrl++"), self).activated.connect(self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(self.zoom_reset)
        
    def update_image_grid(self):
        """Update the image grid based on current batch size."""
        # Clear existing widgets
        for widget in self.image_widgets:
            widget.setParent(None)
        self.image_widgets.clear()
        
        # Clear container layout
        while self.image_container_layout.count():
            child = self.image_container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create new grid layout
        grid_layout = QGridLayout()
        
        # Calculate grid dimensions
        cols = min(4, self.current_batch_size)
        rows = (self.current_batch_size + cols - 1) // cols
        
        # Create image widgets
        for i in range(self.current_batch_size):
            row = i // cols
            col = i % cols
            
            image_widget = ImageDisplayWidget()
            self.image_widgets.append(image_widget)
            grid_layout.addWidget(image_widget, row, col)
        
        self.image_container_layout.addLayout(grid_layout)
        
    def on_batch_size_changed(self, value):
        """Handle batch size change."""
        self.current_batch_size = value
        self.update_image_grid()
        self.load_new_batch()
        
    # ========================================================
    # BATCH NAVIGATION
    # ========================================================
    
    def load_new_batch(self):
        """Load a new random batch of images."""
        self.status_bar.showMessage("Loading new batch...")
        
        # Cancel any ongoing rendering
        if self.rendering_worker and self.rendering_worker.isRunning():
            self.rendering_worker.terminate()
            self.rendering_worker.wait()
        
        # Get new batch
        items = self.dataset_manager.random_batch(self.current_batch_size)
        
        if not items:
            self.status_bar.showMessage("No items to display")
            return
            
        # Clear current displays
        for widget in self.image_widgets:
            widget.clear_display()
            
        # Start rendering in background thread
        self.rendering_worker = RenderingWorker(
            items, 
            self.dataset_manager.visualizer,
            (len(self.image_widgets),)
        )
        self.rendering_worker.finished.connect(self.on_rendering_finished)
        self.rendering_worker.error.connect(self.on_rendering_error)
        self.rendering_worker.start()
        
    def load_previous_batch(self):
        """Load previous batch (placeholder for actual implementation)."""
        QMessageBox.information(self, "Info", "Previous batch functionality not implemented yet")
        
    def load_next_batch(self):
        """Load next batch (placeholder for actual implementation)."""
        self.load_new_batch()
        
    @pyqtSlot(list)
    def on_rendering_finished(self, rendered_images):
        """Handle completion of image rendering."""
        for i, (widget, pil_img) in enumerate(zip(self.image_widgets, rendered_images)):
            if i < len(rendered_images):
                widget.display_image(pil_img)
            else:
                widget.clear_display()
                
        # Update status
        self.status_bar.showMessage(
            f"Displaying {len(rendered_images)}/{self.current_batch_size} images "
            f"| Total items: {len(self.dataset_manager.all_items)}"
        )
        
    @pyqtSlot(str)
    def on_rendering_error(self, error_msg):
        """Handle rendering errors."""
        self.status_bar.showMessage(f"Error: {error_msg}")
        QMessageBox.warning(self, "Rendering Error", 
                          f"Failed to render images:\n{error_msg}")
        
    # ========================================================
    # ZOOM CONTROLS
    # ========================================================
    
    def zoom_in(self):
        """Zoom in on images."""
        for widget in self.image_widgets:
            if widget.pixmap():
                current_size = widget.size()
                new_size = current_size * 1.2
                widget.setMinimumSize(new_size)
                
    def zoom_out(self):
        """Zoom out on images."""
        for widget in self.image_widgets:
            if widget.pixmap():
                current_size = widget.size()
                new_size = current_size * 0.8
                widget.setMinimumSize(max(new_size, QSize(200, 150)))
                
    def zoom_reset(self):
        """Reset zoom to default."""
        for widget in self.image_widgets:
            widget.setMinimumSize(400, 300)
            
    # ========================================================
    # WINDOW EVENTS
    # ========================================================
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any running worker threads
        if self.rendering_worker and self.rendering_worker.isRunning():
            self.rendering_worker.terminate()
            self.rendering_worker.wait()
            
        event.accept()
        
    def resizeEvent(self, event):
        """Handle window resize to update image scaling."""
        super().resizeEvent(event)
        # Update images on resize
        if hasattr(self, 'image_widgets'):
            for widget in self.image_widgets:
                if widget.pixmap():
                    widget.display_image(
                        ImageQt.ImageQt(widget.pixmap().toImage())
                    )


# ============================================================
# APPLICATION LAUNCHER
# ============================================================

def main():
    """Main entry point."""
    # Configuration
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"
    
    # Alternatively, use command line arguments
    if len(sys.argv) > 1:
        DATASET_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        DATASET_FORMAT = sys.argv[2]
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Dataset Reviewer")
    app.setOrganizationName("MLTools")
    
    # Set dark theme option (optional)
    # app.setStyle('Fusion')
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.WindowText, Qt.white)
    # app.setPalette(palette)
    
    # Create and show main window
    try:
        window = ReviewerWindow(DATASET_PATH, DATASET_FORMAT)
        window.show()
        
        # Center window on screen
        screen_geometry = app.primaryScreen().availableGeometry()
        window_geometry = window.frameGeometry()
        window.move(
            (screen_geometry.width() - window_geometry.width()) // 2,
            (screen_geometry.height() - window_geometry.height()) // 2
        )
        
        return app.exec_()
        
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", 
                           f"Failed to start application:\n{str(e)}\n\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
