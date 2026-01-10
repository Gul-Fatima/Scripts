"""
Universal Dataset Viewer
- Supports Datumaro formats: YOLO, COCO, VOC, etc.
- Falls back to simple directory view if format is "directory"
- Batch viewing, reshuffle, next, quit
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datumaro.components.dataset import Dataset
import random
import traceback

# ============================================================
# UNIVERSAL DATASET VIEWER
# ============================================================

class UniversalDatasetViewer:
    """Universal dataset viewer with optional annotations."""

    def __init__(self, dataset_path, dataset_format="yolo", batch_size=4,
                 class_names=None, colors=None):

        self.dataset_path = dataset_path
        self.dataset_format = dataset_format.lower()
        self.batch_size = min(batch_size, 4)
        self.class_names = class_names
        self.colors = colors

        print(f"Initializing viewer: {dataset_path} | Format: {dataset_format}")

        if self.dataset_format == "directory":
            # Simple directory viewer
            self.image_files = self._find_images(dataset_path)
            if not self.image_files:
                raise RuntimeError("No images found in directory!")
            print(f"Found {len(self.image_files)} images.")
            self.items = None  # no Datumaro items
        else:
            # Datumaro-based viewer
            self.dataset = Dataset.import_from(dataset_path, dataset_format)
            subsets = list(self.dataset.subsets())
            if not subsets:
                raise RuntimeError("No subsets found in dataset!")
            subset_name = subsets[0]
            self.items = list(self.dataset.get_subset(subset_name))
            if not self.items:
                raise RuntimeError("Dataset has no items!")
            print(f"Loaded {len(self.items)} items from subset '{subset_name}'.")

            # Load class names
            if self.class_names is None:
                self.class_names = self._infer_class_names()
            if self.colors is None:
                self.colors = self._generate_class_colors(len(self.class_names))

        # Setup figure
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        self.axs = self.axs.flatten()
        self.fig.canvas.manager.set_window_title(
            f"Universal Viewer ({dataset_format})"
        )
        self.info_text = self.fig.text(0.02, 0.98, "", fontsize=10, va='top')

        # Buttons
        self._setup_buttons()

        # Show first batch
        self.show_batch()

    # ========================================================
    # HELPERS
    # ========================================================

    def _find_images(self, directory):
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        files = []
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if any(f.lower().endswith(ext) for ext in exts):
                    files.append(os.path.join(root, f))
        return files

    def _infer_class_names(self):
        try:
            cats = self.dataset.categories().get('label', None)
            if cats:
                return [c.name for c in cats.items]
        except:
            pass
        return ['Class_0', 'Class_1', 'Class_2']

    def _generate_class_colors(self, num_classes):
        palette = [
            (255,0,0), (0,255,0), (0,0,255),
            (255,255,0), (255,0,255), (0,255,255),
            (255,128,0), (128,0,255), (255,0,128), (0,128,255)
        ]
        if num_classes > len(palette):
            return [(random.randint(50,200), random.randint(50,200), random.randint(50,200)) for _ in range(num_classes)]
        return palette[:num_classes]

    def _setup_buttons(self):
        ax_next = plt.axes([0.2,0.02,0.2,0.05])
        btn_next = Button(ax_next, "Next Batch (N)")
        btn_next.on_clicked(self.show_batch)

        ax_shuffle = plt.axes([0.45,0.02,0.2,0.05])
        btn_shuffle = Button(ax_shuffle, "Reshuffle (R)")
        btn_shuffle.on_clicked(self.show_batch)

        ax_quit = plt.axes([0.7,0.02,0.2,0.05])
        btn_quit = Button(ax_quit, "Quit (Q)")
        btn_quit.on_clicked(self.quit)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key in ['n','N',' ','r','R']:
            self.show_batch()
        elif event.key in ['q','Q']:
            self.quit()

    def get_random_batch(self):
        if self.dataset_format == "directory":
            n = min(self.batch_size, len(self.image_files))
            indices = np.random.choice(len(self.image_files), n, replace=False)
            return [self.image_files[i] for i in indices]
        else:
            n = min(self.batch_size, len(self.items))
            indices = np.random.choice(len(self.items), n, replace=False)
            return [self.items[i] for i in indices]

    # ========================================================
    # IMAGE LOADING & ANNOTATIONS
    # ========================================================

    def load_image(self, source):
        if self.dataset_format == "directory":
            # source is file path
            if os.path.exists(source):
                img = cv2.imread(source)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        else:
            # Datumaro item
            try:
                item = source
                if hasattr(item, 'media'):
                    media = item.media
                    if hasattr(media,'data') and media.data is not None:
                        return media.data
                    if hasattr(media,'path') and media.path:
                        return self._load_from_path(media.path)
                if hasattr(item,'image'):
                    img = item.image
                    if hasattr(img,'data') and img.data is not None:
                        return img.data
                    if hasattr(img,'path'):
                        return self._load_from_path(img.path)
            except:
                return None
        return None

    def _load_from_path(self, path):
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def draw_annotations(self, ax, item):
        if self.dataset_format == "directory":
            return  # no annotations
        if not hasattr(item,'annotations'):
            return
        for anno in item.annotations:
            if hasattr(anno,'points') and len(anno.points)>=4:
                x1,y1,x2,y2 = anno.points[:4]
                class_id = getattr(anno,'label',0)
                if class_id>=len(self.class_names):
                    class_id=0
                color = self.colors[class_id]
                color_norm = tuple(c/255 for c in color)
                rect = plt.Rectangle((x1,y1),x2-x1,y2-y1, linewidth=2, edgecolor=color_norm, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1,y1-5, self.class_names[class_id], color='white', fontsize=9,
                        bbox=dict(facecolor=color_norm, alpha=0.8, pad=2))

    # ========================================================
    # SHOW BATCH
    # ========================================================

    def show_batch(self, event=None):
        batch = self.get_random_batch()
        for ax in self.axs:
            ax.cla()
            ax.axis('off')
            ax.set_visible(True)

        for i, (source, ax) in enumerate(zip(batch, self.axs)):
            img = self.load_image(source)
            if img is not None:
                ax.imshow(img)
                title = os.path.basename(source) if self.dataset_format=="directory" else f"Image {i+1}"
                ax.set_title(title[:30], fontsize=11)
                self.draw_annotations(ax, source)
            else:
                ax.text(0.5,0.5,"No Image", ha='center', va='center', color='red')
            ax.axis('off')

        # hide unused axes
        for i in range(len(batch), len(self.axs)):
            self.axs[i].set_visible(False)

        # info
        info = f"Showing {len(batch)} | "
        if self.dataset_format=="directory":
            info += f"Total: {len(self.image_files)}"
        else:
            info += f"Total: {len(self.items)} | Classes: {len(self.class_names)}"
        self.info_text.set_text(info)

        # force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def quit(self, event=None):
        plt.close()

    def run(self):
        plt.show()


# ============================================================
# MAIN CONFIG
# ============================================================

if __name__=="__main__":

    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"

    # Supported: yolo, coco, voc, directory
    DATASET_FORMAT = "yolo"

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
