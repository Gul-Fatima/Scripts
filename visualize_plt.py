"""
Dataset Viewer for YOLO Format
Fixes the "no image attribute" issue
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


# ============================================================
# YOLO DATASET FIX
# ============================================================

class YOLODatasetViewer:
    """Viewer specifically for YOLO format datasets."""
    
    def __init__(self, dataset_path, batch_size=4):
        print(f"Loading YOLO dataset from: {dataset_path}")
        
        # Store dataset path
        self.dataset_path = dataset_path
        
        # Load dataset
        self.dataset = Dataset.import_from(dataset_path, "yolo")
        self.subsets = list(self.dataset.subsets())  # Get subsets (train, val, test)
        
        if not self.subsets:
            print("No subsets found!")
            self.items = []
        else:
            # Use first subset (usually 'train')
            subset_name = self.subsets[0]
            print(f"Using subset: {subset_name}")
            self.items = list(self.dataset.get_subset(subset_name))
        
        print(f"Loaded {len(self.items)} items")
        
        if not self.items:
            print("No items to display!")
            return
            
        self.batch_size = min(batch_size, 4)
        
        # Setup figure
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(f"YOLO Dataset Viewer - {len(self.items)} items")
        self.axs = self.axs.flatten()
        
        # Add info text
        self.info_text = self.fig.text(0.02, 0.98, "", fontsize=10, verticalalignment='top')
        
        # Setup buttons
        self._setup_buttons()
        
        # Load and show first batch
        self.show_batch()
        
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
        """
        Load image for YOLO format.
        YOLO datasets store images in a separate 'images' folder.
        """
        try:
            # Get item ID (usually the image filename without extension)
            item_id = str(item.id) if hasattr(item, 'id') else ""
            
            print(f"Processing item ID: {item_id}")
            print(f"Item type: {type(item)}")
            print(f"Item attributes: {[a for a in dir(item) if not a.startswith('_')]}")
            
            # Method 1: Check if item has media attribute
            if hasattr(item, 'media'):
                print(f"  Has media attribute")
                media = item.media
                print(f"  Media type: {type(media)}")
                print(f"  Media attributes: {[a for a in dir(media) if not a.startswith('_')]}")
                
                if hasattr(media, 'data') and media.data is not None:
                    print(f"  ✓ Got image data from media.data")
                    return media.data
                elif hasattr(media, 'path') and media.path:
                    path = media.path
                    print(f"  Media path: {path}")
                    return self._load_image_from_path(path)
            
            # Method 2: Check for image attribute (might have different structure)
            if hasattr(item, 'image'):
                print(f"  Has image attribute")
                img_attr = item.image
                print(f"  Image attribute type: {type(img_attr)}")
                
                # It might be a string path or an object
                if isinstance(img_attr, str):
                    print(f"  Image is string path: {img_attr}")
                    return self._load_image_from_path(img_attr)
                elif hasattr(img_attr, 'path'):
                    path = img_attr.path
                    print(f"  Image has path: {path}")
                    return self._load_image_from_path(path)
                elif hasattr(img_attr, 'data') and img_attr.data is not None:
                    print(f"  ✓ Got image data from image.data")
                    return img_attr.data
            
            # Method 3: Try to construct path from ID
            if item_id:
                # Common YOLO image extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    # Try in images folder relative to dataset
                    images_dir = os.path.join(self.dataset_path, 'images')
                    if os.path.exists(images_dir):
                        img_path = os.path.join(images_dir, item_id + ext)
                        if os.path.exists(img_path):
                            print(f"  ✓ Found image at: {img_path}")
                            return self._load_image_from_path(img_path)
                    
                    # Try item_id directly as path
                    if os.path.exists(item_id + ext):
                        print(f"  ✓ Found image at: {item_id + ext}")
                        return self._load_image_from_path(item_id + ext)
            
            # Method 4: Try to get the image from the item's annotations
            if hasattr(item, 'annotations'):
                # Some datasets store image data with annotations
                for ann in item.annotations:
                    if hasattr(ann, 'image') and ann.image is not None:
                        if hasattr(ann.image, 'data') and ann.image.data is not None:
                            print(f"  ✓ Got image data from annotation")
                            return ann.image.data
            
            print(f"  ✗ Could not find image for item {item_id}")
            return None
            
        except Exception as e:
            print(f"  ✗ Error loading image: {e}")
            traceback.print_exc()
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
            else:
                print(f"    ✗ OpenCV failed to load: {path}")
                return None
        else:
            # Try relative to dataset path
            rel_path = os.path.join(os.path.dirname(self.dataset_path), path)
            if os.path.exists(rel_path):
                img = cv2.imread(rel_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    print(f"    ✗ OpenCV failed to load relative: {rel_path}")
                    return None
            else:
                print(f"    ✗ Path does not exist: {path}")
                print(f"    ✗ Relative path does not exist: {rel_path}")
                return None
    
    def show_batch(self, event=None):
        """Show a batch of images."""
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
            print(f"\n--- Processing item {i} ---")
            
            # Load image
            img = self.load_yolo_image(item)
            
            if img is not None:
                # Display image
                ax.imshow(img)
                ax.set_title(f"Image {i+1}", fontsize=12)
                
                # Try to draw annotations if they exist
                if hasattr(item, 'annotations'):
                    annos = item.annotations
                    print(f"  Found {len(annos)} annotations")
                    
                    # Draw bounding boxes (simplified)
                    for anno in annos:
                        if hasattr(anno, 'points'):
                            points = anno.points
                            if len(points) >= 4:
                                # Assuming [xmin, ymin, xmax, ymax]
                                x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
                                
                                # Draw rectangle
                                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                                    linewidth=2, edgecolor='red', facecolor='none')
                                ax.add_patch(rect)
                                
                                # Add label if available
                                if hasattr(anno, 'label'):
                                    ax.text(x1, y1-5, f"Class {anno.label}", 
                                           color='red', fontsize=8, backgroundcolor='white')
                else:
                    print(f"  No annotations found")
            else:
                # Show placeholder
                ax.text(0.5, 0.5, f"No Image\nItem {i+1}", 
                       ha='center', va='center', fontsize=14, color='red')
                print(f"  ✗ No image loaded")
                
            ax.axis('off')
            
        # Hide unused axes
        for i in range(len(batch), len(self.axs)):
            self.axs[i].set_visible(False)
            
        # Update info
        info = f"Showing {len(batch)} images | Total: {len(self.items)}"
        self.info_text.set_text(info)
        
        # Update display
        plt.draw()
        print(f"\n✓ Displayed batch of {len(batch)} images")
        
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
# DIRECT YOLO DATASET INSPECTION
# ============================================================

def inspect_yolo_dataset(dataset_path):
    """
    Directly inspect YOLO dataset structure without Datumaro.
    """
    print("\n" + "=" * 60)
    print("DIRECT YOLO DATASET INSPECTION")
    print("=" * 60)
    
    # Check directory structure
    print(f"Dataset path: {dataset_path}")
    print(f"Directory contents:")
    
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    
    # Look for YOLO structure
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(yaml_path):
        print(f"\n✓ Found data.yaml file")
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            print(f"YAML contents:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        except:
            print(f"  Could not parse YAML")
    
    # Look for images directory
    images_dir = os.path.join(dataset_path, 'images')
    if os.path.exists(images_dir):
        print(f"\n✓ Found images directory")
        # Count images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        print(f"  Found {len(image_files)} image files")
        if image_files:
            print(f"  First 3 images:")
            for img in image_files[:3]:
                print(f"    {img}")
    
    # Look for labels directory
    labels_dir = os.path.join(dataset_path, 'labels')
    if os.path.exists(labels_dir):
        print(f"\n✓ Found labels directory")
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        print(f"  Found {len(label_files)} label files")
        if label_files:
            print(f"  First 3 label files:")
            for lbl in label_files[:3]:
                print(f"    {lbl}")
    
    # Look for train.txt, val.txt, test.txt
    for split in ['train', 'val', 'test']:
        txt_path = os.path.join(dataset_path, f'{split}.txt')
        if os.path.exists(txt_path):
            print(f"\n✓ Found {split}.txt")
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                print(f"  Contains {len(lines)} image paths")
                if lines:
                    print(f"  First 3 paths:")
                    for line in lines[:3]:
                        print(f"    {line.strip()}")
            except:
                print(f"  Could not read file")


# ============================================================
# SIMPLE IMAGE DISPLAY FROM DIRECTORY
# ============================================================

def display_images_from_directory(dataset_path):
    """
    Simple function to display images directly from directory.
    """
    print("\n" + "=" * 60)
    print("SIMPLE DIRECTORY IMAGE DISPLAY")
    print("=" * 60)
    
    # Find images in the dataset directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    # Search for images
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    
    if not image_files:
        print("No images found!")
        return
    
    # Take first 4 images
    display_files = image_files[:4]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (img_path, ax) in enumerate(zip(display_files, axes)):
        print(f"\nLoading: {img_path}")
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(os.path.basename(img_path), fontsize=10)
                print(f"  ✓ Loaded: {img.shape}")
            else:
                ax.text(0.5, 0.5, "Failed to load", 
                       ha='center', va='center', color='red')
                print(f"  ✗ Failed to load")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:30]}", 
                   ha='center', va='center', color='red')
            print(f"  ✗ Error: {e}")
        
        ax.axis('off')
    
    # Hide unused axes
    for i in range(len(display_files), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    
    print("YOLO Dataset Viewer")
    print("=" * 60)
    
    # First, inspect the dataset structure
    inspect_yolo_dataset(DATASET_PATH)
    
    print("\n" + "=" * 60)
    print("CHOOSE VIEWER MODE:")
    print("1. YOLO Dataset Viewer (with Datumaro)")
    print("2. Simple Directory Viewer (no Datumaro)")
    print("=" * 60)
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Try with Datumaro
        try:
            viewer = YOLODatasetViewer(DATASET_PATH)
            viewer.run()
        except Exception as e:
            print(f"Error with Datumaro viewer: {e}")
            traceback.print_exc()
            print("\nFalling back to simple directory viewer...")
            display_images_from_directory(DATASET_PATH)
    elif choice == "2":
        # Simple directory viewer
        display_images_from_directory(DATASET_PATH)
    else:
        print("Invalid choice. Using simple directory viewer...")
        display_images_from_directory(DATASET_PATH)
