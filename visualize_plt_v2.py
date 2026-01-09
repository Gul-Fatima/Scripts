"""
Dataset Viewer for YOLO Format
Improved with class names, colors, and better organization
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datumaro.components.dataset import Dataset
import traceback
import random


# ============================================================
# YOLO DATASET VIEWER
# ============================================================

class YOLODatasetViewer:
    """Viewer specifically for YOLO format datasets with class names and colors."""
    
    def __init__(self, dataset_path, batch_size=4, class_names=None, colors=None):
        """
        Initialize viewer.
        
        Args:
            dataset_path: Path to YOLO dataset
            batch_size: Number of images per batch
            class_names: List of class names (optional, will try to read from data.yaml)
            colors: List of colors for each class (optional)
        """
        print(f"Loading YOLO dataset from: {dataset_path}")
        
        # Store parameters
        self.dataset_path = dataset_path
        self.batch_size = min(batch_size, 4)
        
        # Load class names from data.yaml if available
        self.class_names = class_names or self._load_class_names(dataset_path)
        print(f"Class names: {self.class_names}")
        
        # Generate colors for each class
        self.colors = colors or self._generate_class_colors(len(self.class_names))
        
        # Load dataset
        self.dataset = Dataset.import_from(dataset_path, "yolo")
        self.subsets = list(self.dataset.subsets())
        
        if not self.subsets:
            print("No subsets found!")
            self.items = []
        else:
            subset_name = self.subsets[0]
            print(f"Using subset: {subset_name}")
            self.items = list(self.dataset.get_subset(subset_name))
        
        print(f"Loaded {len(self.items)} items")
        
        if not self.items:
            print("No items to display!")
            return
            
        # Setup figure
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.canvas.manager.set_window_title(f"YOLO Dataset Viewer - {len(self.items)} items")
        self.axs = self.axs.flatten()
        
        # Add info text
        self.info_text = self.fig.text(0.02, 0.98, "", fontsize=10, verticalalignment='top')
        
        # Setup buttons
        self._setup_buttons()
        
        # Load and show first batch
        self.show_batch()
        
    def _load_class_names(self, dataset_path):
        """Load class names from data.yaml file."""
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        class_names = []
        
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Extract class names
                if 'names' in data:
                    class_names = data['names']
                elif 'nc' in data:
                    # If only number of classes, create generic names
                    nc = data['nc']
                    class_names = [f'Class_{i}' for i in range(nc)]
                    
                print(f"✓ Loaded {len(class_names)} classes from data.yaml")
            except Exception as e:
                print(f"✗ Could not parse data.yaml: {e}")
        
        return class_names if class_names else ['Class_0', 'Class_1', 'Class_2']
    
    def _generate_class_colors(self, num_classes):
        """Generate distinct colors for each class."""
        # Predefined color palette
        color_palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 0, 128),  # Pink
            (0, 128, 255),  # Light Blue
        ]
        
        # If more classes than colors, generate random colors
        if num_classes > len(color_palette):
            colors = []
            for _ in range(num_classes):
                colors.append((random.randint(50, 200), 
                              random.randint(50, 200), 
                              random.randint(50, 200)))
            return colors
        
        return color_palette[:num_classes]
    
    def _setup_buttons(self):
        """Setup control buttons."""
        # Next button
        ax_next = plt.axes([0.4, 0.02, 0.2, 0.05])
        self.btn_next = Button(ax_next, 'Next Batch (N)')
        self.btn_next.on_clicked(self.show_batch)
        
        # Quit button
        ax_quit = plt.axes([0.65, 0.02, 0.2, 0.05])
        self.btn_quit = Button(ax_quit, 'Quit (Q)')
        self.btn_quit.on_clicked(self.quit)
        
        # Connect keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def on_key(self, event):
        """Keyboard shortcuts."""
        if event.key in ['n', 'N', ' ']:
            self.show_batch()
        elif event.key in ['q', 'Q']:
            self.quit()
            
    def get_random_batch(self):
        """Get random batch of items."""
        n = min(self.batch_size, len(self.items))
        indices = np.random.choice(len(self.items), n, replace=False)
        return [self.items[i] for i in indices]
    
    def load_yolo_image(self, item):
        """Load image for YOLO format."""
        try:
            item_id = str(item.id) if hasattr(item, 'id') else ""
            
            # Method 1: Check media attribute
            if hasattr(item, 'media'):
                media = item.media
                if hasattr(media, 'data') and media.data is not None:
                    return media.data
                elif hasattr(media, 'path') and media.path:
                    return self._load_image_from_path(media.path)
            
            # Method 2: Check image attribute
            if hasattr(item, 'image'):
                img_attr = item.image
                if isinstance(img_attr, str):
                    return self._load_image_from_path(img_attr)
                elif hasattr(img_attr, 'path'):
                    return self._load_image_from_path(img_attr.path)
                elif hasattr(img_attr, 'data') and img_attr.data is not None:
                    return img_attr.data
            
            # Method 3: Construct path from ID
            if item_id:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    images_dir = os.path.join(self.dataset_path, 'images')
                    if os.path.exists(images_dir):
                        img_path = os.path.join(images_dir, item_id + ext)
                        if os.path.exists(img_path):
                            return self._load_image_from_path(img_path)
            
            return None
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _load_image_from_path(self, path):
        """Load image from file path."""
        if not path or not isinstance(path, str):
            return None
            
        # Check if path exists
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Try relative to dataset path
        rel_path = os.path.join(os.path.dirname(self.dataset_path), path)
        if os.path.exists(rel_path):
            img = cv2.imread(rel_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return None
    
    def draw_annotations(self, ax, item, image_shape):
        """Draw annotations with class names and colors."""
        if not hasattr(item, 'annotations'):
            return
            
        annos = item.annotations
        annotation_count = {i: 0 for i in range(len(self.class_names))}
        
        for anno in annos:
            if hasattr(anno, 'points') and len(anno.points) >= 4:
                x1, y1, x2, y2 = anno.points[0], anno.points[1], anno.points[2], anno.points[3]
                
                # Get class label
                class_id = 0
                if hasattr(anno, 'label'):
                    class_id = anno.label
                
                # Ensure class_id is within bounds
                if class_id >= len(self.class_names):
                    class_id = 0
                
                # Get color for this class
                color = self.colors[class_id]
                color_normalized = (color[0]/255, color[1]/255, color[2]/255)
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor=color_normalized, 
                                    facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
                # Add label with class name
                class_name = self.class_names[class_id]
                label_text = f"{class_name}"
                
                # Add confidence if available
                if hasattr(anno, 'attributes') and 'conf' in anno.attributes:
                    conf = anno.attributes['conf']
                    label_text += f" {conf:.2f}"
                
                # Add label background
                ax.text(x1, y1-5, label_text, 
                       color='white', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=color_normalized, 
                                edgecolor=color_normalized, 
                                alpha=0.8))
                
                # Count annotations per class
                annotation_count[class_id] += 1
        
        # Display annotation summary in corner
        summary_parts = []
        for class_id, count in annotation_count.items():
            if count > 0:
                color = self.colors[class_id]
                color_normalized = (color[0]/255, color[1]/255, color[2]/255)
                class_name = self.class_names[class_id]
                summary_parts.append(f"{class_name}: {count}")
        
        if summary_parts:
            summary_text = " | ".join(summary_parts)
            ax.text(0.02, 0.98, summary_text,
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.7),
                   verticalalignment='top')
    
    def show_batch(self, event=None):
        """Show a batch of images with annotations."""
        print("\n" + "=" * 60)
        print("LOADING NEW BATCH")
        print("=" * 60)
        
        # Get batch
        batch = self.get_random_batch()
        if not batch:
            print("No items in batch!")
            return
            
        # Clear axes
        for ax in self.axs:
            ax.clear()
            ax.axis('off')
            
        # Display each image
        for i, (item, ax) in enumerate(zip(batch, self.axs)):
            # Load image
            img = self.load_yolo_image(item)
            
            if img is not None:
                # Display image
                ax.imshow(img)
                ax.set_title(f"Image {i+1}", fontsize=12, fontweight='bold')
                
                # Draw annotations
                self.draw_annotations(ax, item, img.shape)
            else:
                # Show placeholder
                ax.text(0.5, 0.5, f"No Image\nItem {i+1}", 
                       ha='center', va='center', fontsize=14, color='red')
            
            ax.axis('off')
            
        # Hide unused axes
        for i in range(len(batch), len(self.axs)):
            self.axs[i].set_visible(False)
            
        # Update info
        info = f"Showing {len(batch)} images | Total: {len(self.items)} | Classes: {len(self.class_names)}"
        self.info_text.set_text(info)
        
        # Update display
        plt.draw()
        print(f"✓ Displayed batch of {len(batch)} images")
        
    def quit(self, event=None):
        """Close the viewer."""
        plt.close()
        
    def run(self):
        """Run the viewer."""
        if self.items:
            plt.show()
        else:
            print("Cannot run: No items to display!")


# ============================================================
# SIMPLE DIRECTORY VIEWER
# ============================================================

class SimpleDirectoryViewer:
    """
    Simple viewer that displays images directly from directory.
    This is a fallback option when Datumaro doesn't work.
    Shows that you can browse images without YOLO annotations.
    """
    
    def __init__(self, dataset_path, batch_size=4):
        print(f"Simple Directory Viewer - browsing: {dataset_path}")
        
        self.dataset_path = dataset_path
        self.batch_size = min(batch_size, 4)
        
        # Find all images in directory
        self.image_files = self._find_images(dataset_path)
        print(f"Found {len(self.image_files)} image files")
        
        if not self.image_files:
            print("No images found!")
            return
            
        # Setup figure
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(f"Directory Viewer - {len(self.image_files)} images")
        self.axs = self.axs.flatten()
        
        # Add info
        self.info_text = self.fig.text(0.02, 0.98, "", fontsize=10, verticalalignment='top')
        
        # Setup buttons
        self._setup_buttons()
        
        # Show first batch
        self.show_batch()
        
    def _find_images(self, directory):
        """Find all image files in directory and subdirectories."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def _setup_buttons(self):
        """Setup control buttons."""
        ax_next = plt.axes([0.4, 0.02, 0.2, 0.05])
        btn_next = Button(ax_next, 'Next Batch (N)')
        btn_next.on_clicked(self.show_batch)
        
        ax_quit = plt.axes([0.65, 0.02, 0.2, 0.05])
        btn_quit = Button(ax_quit, 'Quit (Q)')
        btn_quit.on_clicked(self.quit)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def on_key(self, event):
        if event.key in ['n', 'N', ' ']:
            self.show_batch()
        elif event.key in ['q', 'Q']:
            self.quit()
    
    def get_random_batch(self):
        """Get random batch of image files."""
        n = min(self.batch_size, len(self.image_files))
        indices = np.random.choice(len(self.image_files), n, replace=False)
        return [self.image_files[i] for i in indices]
    
    def show_batch(self, event=None):
        """Show a batch of images."""
        batch_files = self.get_random_batch()
        
        # Clear axes
        for ax in self.axs:
            ax.clear()
            ax.axis('off')
            
        # Display images
        for i, (file_path, ax) in enumerate(zip(batch_files, self.axs)):
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    filename = os.path.basename(file_path)
                    ax.set_title(f"{filename[:20]}...", fontsize=10)
                else:
                    ax.text(0.5, 0.5, "Failed to load", 
                           ha='center', va='center', color='red')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error", 
                       ha='center', va='center', color='red')
            
            ax.axis('off')
            
        # Hide unused axes
        for i in range(len(batch_files), len(self.axs)):
            self.axs[i].set_visible(False)
            
        # Update info
        info = f"Showing {len(batch_files)} images | Total: {len(self.image_files)}"
        self.info_text.set_text(info)
        
        plt.draw()
        
    def quit(self, event=None):
        plt.close()
    
    def run(self):
        if self.image_files:
            plt.show()
        else:
            print("No images to display!")


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # CONFIGURATION SECTION
    # ========================================================
    
    # Dataset path (change this to your dataset location)
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    
    # Batch size (number of images to show at once)
    BATCH_SIZE = 4
    
    # Optional: Specify class names if not in data.yaml
    # CLASS_NAMES = ["pothole", "crack", "patch"]  # Example
    CLASS_NAMES = None  # Will be loaded from data.yaml
    
    # Optional: Specify custom colors for each class
    # Colors should be RGB tuples (0-255)
    # CUSTOM_COLORS = [
    #     (255, 0, 0),    # Red for class 0
    #     (0, 255, 0),    # Green for class 1
    #     (0, 0, 255),    # Blue for class 2
    # ]
    CUSTOM_COLORS = None  # Will be auto-generated
    
    # Viewer mode
    # Options: "datumaro" (with annotations) or "directory" (simple image browser)
    VIEWER_MODE = "datumaro"
    
    # ========================================================
    # RUN VIEWER (Don't modify below this line)
    # ========================================================
    
    print("=" * 60)
    print("YOLO DATASET VIEWER")
    print("=" * 60)
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"✗ Dataset path does not exist: {DATASET_PATH}")
        print("Please check the DATASET_PATH in the configuration section.")
        exit(1)
    
    try:
        if VIEWER_MODE == "datumaro":
            print("Starting YOLO Dataset Viewer with annotations...")
            viewer = YOLODatasetViewer(
                dataset_path=DATASET_PATH,
                batch_size=BATCH_SIZE,
                class_names=CLASS_NAMES,
                colors=CUSTOM_COLORS
            )
            viewer.run()
            
        elif VIEWER_MODE == "directory":
            print("Starting Simple Directory Viewer (images only)...")
            viewer = SimpleDirectoryViewer(
                dataset_path=DATASET_PATH,
                batch_size=BATCH_SIZE
            )
            viewer.run()
            
        else:
            print(f"✗ Unknown viewer mode: {VIEWER_MODE}")
            print("Please set VIEWER_MODE to 'datumaro' or 'directory'")
            
    except Exception as e:
        print(f"✗ Error starting viewer: {e}")
        traceback.print_exc()
        
        # Try fallback to simple directory viewer
        print("\n" + "=" * 60)
        print("FALLING BACK TO DIRECTORY VIEWER")
        print("=" * 60)
        
        try:
            viewer = SimpleDirectoryViewer(DATASET_PATH, BATCH_SIZE)
            viewer.run()
        except:
            print("✗ Could not start any viewer.")
