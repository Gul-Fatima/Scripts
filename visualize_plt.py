"""
QUICK TEST: Just load and display one image
"""

import cv2
import matplotlib.pyplot as plt
from datumaro.components.dataset import Dataset
import os

DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
DATASET_FORMAT = "yolo"

# Load dataset
dataset = Dataset.import_from(DATASET_PATH, DATASET_FORMAT)
items = list(dataset)
print(f"Total items: {len(items)}")

if items:
    # Take first item
    item = items[0]
    print(f"\nFirst item type: {type(item)}")
    print(f"Attributes: {[a for a in dir(item) if not a.startswith('_')]}")
    
    if hasattr(item, 'image'):
        print(f"\nImage attributes: {[a for a in dir(item.image) if not a.startswith('_')]}")
        
        # Try different ways to get image
        img = None
        
        # Method 1: Direct data
        if hasattr(item.image, 'data') and item.image.data is not None:
            print(f"Got data attribute")
            img = item.image.data
            
        # Method 2: Path
        elif hasattr(item.image, 'path') and item.image.path:
            path = item.image.path
            print(f"Got path: {path}")
            
            # Check if exists
            print(f"Exists as absolute: {os.path.exists(path)}")
            
            # Try relative
            base_dir = os.path.dirname(DATASET_PATH)
            rel_path = os.path.join(base_dir, path)
            print(f"Relative path: {rel_path}")
            print(f"Exists relative: {os.path.exists(rel_path)}")
            
            # Try to read
            if os.path.exists(path):
                img = cv2.imread(path)
            elif os.path.exists(rel_path):
                img = cv2.imread(rel_path)
                
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display
        if img is not None:
            print(f"Image shape: {img.shape}")
            plt.imshow(img)
            plt.title("First image from dataset")
            plt.axis('off')
            plt.show()
        else:
            print("Failed to load image!")
    else:
        print("Item has no image attribute!")
else:
    print("Dataset is empty!")
