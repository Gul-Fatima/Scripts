"""
Interactive Dataset Reviewer (Matplotlib Version)
Fixed Datumaro rendering issue
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for better interactivity
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from datumaro.components.dataset import Dataset
from datumaro.components.visualizer import Visualizer


# ============================================================
# REVIEWER CLASS
# ============================================================

class DatasetReviewer:
    """
    Interactive reviewer for a Datumaro dataset using matplotlib.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 4, n_rows: int = 2, n_cols: int = 2):
        """
        Initialize the reviewer.

        Args:
            dataset: Datumaro Dataset object
            batch_size: Number of images to display per batch
            n_rows: Number of rows in gallery
            n_cols: Number of columns in gallery
        """
        self.dataset = dataset
        self.items = list(dataset)
        self.viz = Visualizer(dataset)
        self.batch_size = batch_size
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Create figure with proper layout
        self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=(12, 8))
        if batch_size == 1:
            self.axs = np.array([self.axs])
        self.axs = self.axs.flatten()
        
        # Adjust layout for buttons
        plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
        
        # Status text
        self.status_text = plt.figtext(0.5, 0.02, f"Total items: {len(self.items)}", 
                                      ha='center', fontsize=10)

        # Buttons
        self._setup_buttons()
        
        # Track current batch
        self.current_batch = []

    # --------------------------------------------------------
    # BUTTON SETUP
    # --------------------------------------------------------
    def _setup_buttons(self):
        """Setup matplotlib buttons for interaction."""
        # Next button
        ax_next = plt.axes([0.3, 0.05, 0.2, 0.05])
        self.btn_next = Button(ax_next, 'Next Batch (N)', color='lightblue', hovercolor='skyblue')
        self.btn_next.on_clicked(self.next_batch)
        
        # Quit button
        ax_quit = plt.axes([0.55, 0.05, 0.2, 0.05])
        self.btn_quit = Button(ax_quit, 'Quit (Q)', color='lightcoral', hovercolor='red')
        self.btn_quit.on_clicked(self.quit)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    # --------------------------------------------------------
    # KEYBOARD EVENTS
    # --------------------------------------------------------
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key in ['n', 'N', ' ']:
            self.next_batch()
        elif event.key in ['q', 'Q', 'escape']:
            self.quit()

    # --------------------------------------------------------
    # RANDOM SELECTION
    # --------------------------------------------------------
    def get_random_items(self):
        """Return a random batch of dataset items."""
        if not self.items:
            return []
        n = min(self.batch_size, len(self.items))
        indices = np.random.choice(len(self.items), n, replace=False)
        return [self.items[i] for i in indices]

    # --------------------------------------------------------
    # RENDERING - FIXED VERSION
    # --------------------------------------------------------
    def render_gallery(self, selected_items):
        """Render the selected items in the grid axes."""
        # Clear all axes first
        for ax in self.axs:
            ax.clear()
            ax.axis('off')
            ax.set_title('')

        self.current_batch = selected_items
        
        for i, (item, ax) in enumerate(zip(selected_items, self.axs)):
            try:
                # Method 1: Try to get image data from item
                img_data = None
                
                # Check if item has image data
                if hasattr(item, 'image') and item.image is not None:
                    # Try to get data directly
                    if hasattr(item.image, 'data') and item.image.data is not None:
                        img_data = item.image.data
                    
                    # Try to load from path
                    elif hasattr(item.image, 'path') and item.image.path:
                        try:
                            img_cv = cv2.imread(item.image.path)
                            if img_cv is not None:
                                img_data = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        except:
                            pass
                
                if img_data is not None:
                    # Display the image
                    ax.imshow(img_data)
                    
                    # Try to draw annotations using Datumaro visualizer
                    try:
                        self.viz.draw_item(item, ax)
                    except:
                        # If draw_item fails, just show the image
                        pass
                    
                    # Set title
                    title = f"Item {i+1}"
                    if hasattr(item, 'id'):
                        title += f"\nID: {item.id[:20]}..."
                    ax.set_title(title, fontsize=9)
                    
                else:
                    # No image data found
                    ax.text(0.5, 0.5, "No image data", 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color='red')
                    ax.set_title(f"Item {i+1} - Failed", fontsize=9)
                
                ax.axis('off')
                
            except Exception as e:
                # Error handling
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}...", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
                ax.axis('off')
                print(f"Error rendering item {i}: {e}")

        # Hide unused axes
        for i in range(len(selected_items), len(self.axs)):
            self.axs[i].set_visible(False)

        # Update status
        if hasattr(self, 'status_text'):
            self.status_text.set_text(f"Showing {len(selected_items)} items | Total: {len(self.items)}")
        
        plt.draw()

    # --------------------------------------------------------
    # BUTTON CALLBACKS
    # --------------------------------------------------------
    def next_batch(self, event=None):
        """Load and display the next batch of images."""
        items = self.get_random_items()
        if items:
            self.render_gallery(items)
        else:
            print("No items to display!")

    def quit(self, event=None):
        """Close the figure and exit."""
        plt.close(self.fig)

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    def run(self):
        """Start the reviewer."""
        if not self.items:
            print("Dataset is empty. Exiting.")
            return

        # Display initial batch
        self.next_batch()
        plt.show()


# ============================================================
# SIMPLER VERSION WITHOUT DATUMARO VISUALIZER
# ============================================================

class SimpleDatasetReviewer:
    """
    Even simpler version that just shows images without annotations.
    """
    
    def __init__(self, dataset_path: str, dataset_format: str, batch_size: int = 4):
        """Load dataset and setup viewer."""
        self.dataset = Dataset.import_from(dataset_path, dataset_format)
        self.items = list(self.dataset)
        self.batch_size = batch_size
        
        # Calculate grid
        n_cols = min(2, batch_size)
        n_rows = (batch_size + n_cols - 1) // n_cols
        
        # Create figure
        self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=(10, 6 * n_rows / 2))
        if batch_size == 1:
            self.axs = np.array([self.axs])
        self.axs = self.axs.flatten()
        
        plt.subplots_adjust(bottom=0.15)
        
        # Setup buttons
        self._setup_buttons()
        
    def _setup_buttons(self):
        """Simple button setup."""
        ax_next = plt.axes([0.4, 0.05, 0.2, 0.05])
        ax_quit = plt.axes([0.65, 0.05, 0.2, 0.05])
        
        btn_next = Button(ax_next, 'Next Batch')
        btn_quit = Button(ax_quit, 'Quit')
        
        btn_next.on_clicked(self.next_batch)
        btn_quit.on_clicked(self.quit)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def on_key(self, event):
        """Keyboard shortcuts."""
        if event.key in ['n', 'N', ' ']:
            self.next_batch()
        elif event.key in ['q', 'Q']:
            self.quit()
    
    def get_random_items(self):
        """Get random batch."""
        if not self.items:
            return []
        n = min(self.batch_size, len(self.items))
        indices = np.random.choice(len(self.items), n, replace=False)
        return [self.items[i] for i in indices]
    
    def next_batch(self, event=None):
        """Show next batch."""
        items = self.get_random_items()
        
        # Clear axes
        for ax in self.axs:
            ax.clear()
            ax.axis('off')
        
        # Display images
        for i, (item, ax) in enumerate(zip(items, self.axs)):
            try:
                # Try to get image
                img = None
                if hasattr(item, 'image') and item.image:
                    if hasattr(item.image, 'data') and item.image.data is not None:
                        img = item.image.data
                    elif hasattr(item.image, 'path'):
                        img_cv = cv2.imread(item.image.path)
                        if img_cv is not None:
                            img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                if img is not None:
                    ax.imshow(img)
                    ax.set_title(f"Image {i+1}", fontsize=10)
                else:
                    ax.text(0.5, 0.5, "No image", ha='center', va='center')
                
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error", ha='center', va='center', color='red')
                print(f"Error: {e}")
        
        # Hide unused axes
        for i in range(len(items), len(self.axs)):
            self.axs[i].set_visible(False)
        
        plt.draw()
    
    def quit(self, event=None):
        plt.close()
    
    def run(self):
        self.next_batch()
        plt.show()


# ============================================================
# MAIN BLOCK - CONFIGURATION
# ============================================================

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"
    
    # Try the simple version first
    print("Loading dataset...")
    
    try:
        # Option 1: Simple version (just images)
        print("Using simple viewer (images only)...")
        dataset = Dataset.import_from(DATASET_PATH, DATASET_FORMAT)
        print(f"Loaded {len(list(dataset))} items")
        
        reviewer = SimpleDatasetReviewer(DATASET_PATH, DATASET_FORMAT, batch_size=4)
        reviewer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Option 2: Try with basic error display
        import traceback
        print("Falling back to error display...")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, 
               f"Failed to load dataset\n\nError: {str(e)[:100]}...\n\n"
               f"Check path: {DATASET_PATH}\nFormat: {DATASET_FORMAT}",
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.show()
