"""
Interactive Dataset Reviewer (Matplotlib Version)
- Shows N random images per batch in a grid gallery
- Uses Datumaro's Visualizer for annotations
- Fully interactive with Next/Quit buttons
"""

import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.visualizer import Visualizer
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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

        # Figure and axes
        self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=(10, 8))
        self.axs = self.axs.flatten()
        plt.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.3)

        # Buttons
        self._setup_buttons()

    # --------------------------------------------------------
    # BUTTON SETUP
    # --------------------------------------------------------
    def _setup_buttons(self):
        """Setup matplotlib buttons for interaction."""
        ax_next = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_quit = plt.axes([0.81, 0.05, 0.1, 0.075])

        self.btn_next = Button(ax_next, 'Next Batch', color='lightblue', hovercolor='skyblue')
        self.btn_next.on_clicked(self.next_batch)

        self.btn_quit = Button(ax_quit, 'Quit', color='lightcoral', hovercolor='red')
        self.btn_quit.on_clicked(self.quit)

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
    # RENDERING
    # --------------------------------------------------------
    def render_gallery(self, selected_items):
        """Render the selected items in the grid axes."""
        for ax in self.axs:
            ax.clear()
            ax.axis('off')

        for i, (item, ax) in enumerate(zip(selected_items, self.axs)):
            try:
                img_array = self.viz.render_item(item)  # RGB numpy array
                if img_array is None:
                    raise ValueError("Visualizer returned None")
                ax.imshow(img_array)
                ax.set_title(f"Item {i}")
                ax.axis('off')
            except Exception:
                ax.text(0.5, 0.5, "Failed to load",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        plt.draw()

    # --------------------------------------------------------
    # BUTTON CALLBACKS
    # --------------------------------------------------------
    def next_batch(self, event=None):
        """Load and display the next batch of images."""
        items = self.get_random_items()
        self.render_gallery(items)

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
# MAIN BLOCK - CONFIGURATION AT THE BOTTOM
# ============================================================

if __name__ == "__main__":
    # -----------------------------
    # User Configuration
    # -----------------------------
    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"
    BATCH_SIZE = 4  # Number of images per batch
    GRID_ROWS = 2
    GRID_COLS = 2

    # -----------------------------
    # Load Dataset
    # -----------------------------
    try:
        dataset = Dataset.import_from(DATASET_PATH, DATASET_FORMAT)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        exit(1)

    # -----------------------------
    # Run Reviewer
    # -----------------------------
    reviewer = DatasetReviewer(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        n_rows=GRID_ROWS,
        n_cols=GRID_COLS
    )
    reviewer.run()
