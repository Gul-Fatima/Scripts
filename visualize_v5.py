"""
Universal Dataset Viewer
Supports Datumaro formats (YOLO, COCO, VOC, etc.) and plain image directories.
Improved with class names, colors, and reshuffle support.
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
# UNIVERSAL DATASET VIEWER
# ============================================================

class UniversalDatasetViewer:
    """Universal viewer for Datumaro datasets and directories."""

    def __init__(self, dataset_path, dataset_format="yolo", batch_size=4, class_names=None, colors=None):
        """
        Args:
            dataset_path: Path to dataset
            dataset_format: 'yolo', 'coco', 'voc', or 'directory'
            batch_size: Number of images per batch
            class_names: Optional class names list (for annotation formats)
            colors: Optional RGB colors for each class
        """
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format.lower()
        self.batch_size = min(batch_size, 4)
        self.class_names = class_names
        self.colors = colors

        self.items = []
        self.image_files = []

        # Load dataset depending on format
        if self.dataset_format == "directory":
            self._load_directory()
        else:
            self._load_datumaro()

        # Setup figure
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.canvas.manager.set_window_title(f"Dataset Viewer - {len(self.items) or len(self.image_files)} items")
        self.axs = self.axs.flatten()
        self.info_text = self.fig.text(0.02, 0.98, "", fontsize=10, verticalalignment='top')

        # Setup buttons
        self._setup_buttons()

        # Show first batch
        self.show_batch()

    # --------------------------
    # Dataset Loading
    # --------------------------
    def _load_directory(self):
        """Load images from directory."""
        print(f"Loading images from directory: {self.dataset_path}")
        self.image_files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    self.image_files.append(os.path.join(root, file))
        print(f"✓ Found {len(self.image_files)} image files")

    def _load_datumaro(self):
        """Load dataset via Datumaro."""
        print(f"Loading Datumaro dataset ({self.dataset_format}) from: {self.dataset_path}")
        try:
            self.dataset = Dataset.import_from(self.dataset_path, self.dataset_format)
            subsets = list(self.dataset.subsets())
            if subsets:
                subset_name = subsets[0]
                self.items = list(self.dataset.get_subset(subset_name))
            else:
                print("✗ No subsets found!")
                self.items = []

            # Load class names and colors
            if not self.class_names:
                self.class_names = self._load_class_names()
            if not self.colors:
                self.colors = self._generate_class_colors(len(self.class_names))

            print(f"✓ Loaded {len(self.items)} items | {len(self.class_names)} classes")
        except Exception as e:
            print(f"✗ Failed to load dataset via Datumaro: {e}")
            traceback.print_exc()
            self.items = []

    def _load_class_names(self):
        """Load class names from data.yaml if YOLO, otherwise generic."""
        if self.dataset_format == "yolo":
            yaml_path = os.path.join(self.dataset_path, 'data.yaml')
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    if 'names' in data:
                        return data['names']
                    elif 'nc' in data:
                        return [f"Class_{i}" for i in range(data['nc'])]
                except:
                    pass
        # Default fallback
        return ['Class_0', 'Class_1', 'Class_2']

    def _generate_class_colors(self, num_classes):
        """Generate distinct colors for classes."""
        palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (255, 0, 128), (0, 128, 255)
        ]
        if num_classes <= len(palette):
            return palette[:num_classes]
        else:
            colors = []
            for _ in range(num_classes):
                colors.append((random.randint(50,200), random.randint(50,200), random.randint(50,200)))
            return colors

    # --------------------------
    # Buttons & Keyboard
    # --------------------------
    def _setup_buttons(self):
        ax_next = plt.axes([0.4, 0.02, 0.2, 0.05])
        self.btn_next = Button(ax_next, 'Next Batch (N)')
        self.btn_next.on_clicked(self.show_batch)

        ax_quit = plt.axes([0.65, 0.02, 0.2, 0.05])
        self.btn_quit = Button(ax_quit, 'Quit (Q)')
        self.btn_quit.on_clicked(self.quit)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key in ['n', 'N', ' ']:
            self.show_batch()
        elif event.key in ['q', 'Q']:
            self.quit()

    # --------------------------
    # Batch Selection
    # --------------------------
    def get_random_batch(self):
        """Return random batch of images or items."""
        n = self.batch_size
        if self.dataset_format == "directory":
            n = min(n, len(self.image_files))
            indices = np.random.choice(len(self.image_files), n, replace=False)
            return [self.image_files[i] for i in indices]
        else:
            n = min(n, len(self.items))
            indices = np.random.choice(len(self.items), n, replace=False)
            return [self.items[i] for i in indices]

    # --------------------------
    # Image Loading
    # --------------------------
    def load_image(self, source):
        """Load image from file path or Datumaro item."""
        if self.dataset_format == "directory":
            img = cv2.imread(source)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        else:
            item = source
            item_id = str(item.id) if hasattr(item, 'id') else ""
            if hasattr(item, 'media'):
                media = item.media
                if hasattr(media, 'data') and media.data is not None:
                    return media.data
                elif hasattr(media, 'path') and media.path:
                    return self._load_image_from_path(media.path)
            if hasattr(item, 'image'):
                img_attr = item.image
                if isinstance(img_attr, str):
                    return self._load_image_from_path(img_attr)
                elif hasattr(img_attr, 'path'):
                    return self._load_image_from_path(img_attr.path)
                elif hasattr(img_attr, 'data') and img_attr.data is not None:
                    return img_attr.data
            # Fallback to ID-based path
            if item_id:
                for ext in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']:
                    images_dir = os.path.join(self.dataset_path,'images')
                    if os.path.exists(images_dir):
                        img_path = os.path.join(images_dir,item_id+ext)
                        if os.path.exists(img_path):
                            return self._load_image_from_path(img_path)
            return None

    def _load_image_from_path(self, path):
        if not path or not isinstance(path, str):
            return None
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    # --------------------------
    # Drawing
    # --------------------------
    def draw_annotations(self, ax, item):
        if not hasattr(item, 'annotations'):
            return
        annos = item.annotations
        annotation_count = {i: 0 for i in range(len(self.class_names))}
        for anno in annos:
            if hasattr(anno, 'points') and len(anno.points) >= 4:
                x1, y1, x2, y2 = anno.points[:4]
                class_id = getattr(anno,'label',0)
                if class_id >= len(self.class_names):
                    class_id = 0
                color = self.colors[class_id]
                color_norm = tuple(c/255 for c in color)
                rect = plt.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,edgecolor=color_norm,facecolor='none',alpha=0.8)
                ax.add_patch(rect)
                label_text = self.class_names[class_id]
                if hasattr(anno,'attributes') and 'conf' in anno.attributes:
                    label_text += f" {anno.attributes['conf']:.2f}"
                ax.text(x1, y1-5, label_text, color='white', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',facecolor=color_norm,edgecolor=color_norm,alpha=0.8))
                annotation_count[class_id] += 1
        if annotation_count:
            summary = " | ".join(f"{self.class_names[i]}:{c}" for i,c in annotation_count.items() if c>0)
            ax.text(0.02,0.98,summary,transform=ax.transAxes,fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='gray',alpha=0.7),
                    verticalalignment='top')

    # --------------------------
    # Display Batch
    # --------------------------
    def show_batch(self, event=None):
        print("\n" + "="*50)
        print("LOADING NEW BATCH")
        print("="*50)

        batch = self.get_random_batch()
        if not batch:
            print("No items/images to display!")
            return

        for ax in self.axs:
            ax.clear()
            ax.axis('off')

        for i, (source, ax) in enumerate(zip(batch, self.axs)):
            img = self.load_image(source)
            if img is not None:
                ax.imshow(img)
                if self.dataset_format != "directory":
                    self.draw_annotations(ax, source)
                title = f"Image {i+1}" if self.dataset_format != "directory" else os.path.basename(source)
                ax.set_title(title, fontsize=12, fontweight='bold')
            else:
                ax.text(0.5,0.5,"Failed to load",ha='center',va='center',color='red',fontsize=14)
            ax.axis('off')

        for i in range(len(batch), len(self.axs)):
            self.axs[i].set_visible(False)

        info = f"Showing {len(batch)} images | Total: {len(self.items) or len(self.image_files)}"
        if self.dataset_format != "directory":
            info += f" | Classes: {len(self.class_names)}"
        self.info_text.set_text(info)

        plt.draw()
        print(f"✓ Displayed batch of {len(batch)} images")

    def quit(self, event=None):
        plt.close()

    def run(self):
        plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    # -------------------------
    # CONFIGURATION
    # -------------------------
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"  # options: "yolo", "coco", "voc", "directory"
    BATCH_SIZE = 4
    CLASS_NAMES = None
    CUSTOM_COLORS = None

    if not os.path.exists(DATASET_PATH):
        print(f"✗ Dataset path does not exist: {DATASET_PATH}")
        exit(1)

    try:
        viewer = UniversalDatasetViewer(
            dataset_path=DATASET_PATH,
            dataset_format=DATASET_FORMAT,
            batch_size=BATCH_SIZE,
            class_names=CLASS_NAMES,
            colors=CUSTOM_COLORS
        )
        viewer.run()
    except Exception as e:
        print(f"✗ Failed to start viewer: {e}")
        traceback.print_exc()
