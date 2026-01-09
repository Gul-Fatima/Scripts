"""
Dataset Visualization & Review Tool (Matplotlib Version)

Purpose:
--------
Provides a lightweight, keyboard-driven interface for visually reviewing
annotated datasets (YOLO format) using Datumaro. This tool is intended for
dataset QA and inspection prior to model training.
"""

import sys
import matplotlib.pyplot as plt
from datumaro.components.dataset import Dataset
from datumaro.components.visualizer import Visualizer


# ============================================================
# APPLICATION LOGIC
# ============================================================

class DatasetReviewer:
    """
    Handles dataset loading, batch sampling, and figure updates.
    """

    def __init__(self, dataset_path: str, dataset_format: str, images_per_batch: int):
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.images_per_batch = images_per_batch

        # Load dataset once (avoid reloading on every refresh)
        self.dataset = Dataset.import_from(self.dataset_path, self.dataset_format)
        self.visualizer = Visualizer(self.dataset, figsize=(12, 8))

        self.current_items = []
        self.figure = None

    def get_random_batch(self):
        """Fetch a random batch of dataset items."""
        self.current_items = self.visualizer.get_random_items(self.images_per_batch)

    def render(self):
        """
        Render the current batch in the existing Matplotlib window.
        Clears previous content instead of opening new windows.
        """
        if self.figure:
            plt.clf()

        self.figure = self.visualizer.vis_gallery(self.current_items)
        self.figure.canvas.draw_idle()

    def refresh(self):
        """Load a new random batch and re-render."""
        self.get_random_batch()
        self.render()


# ============================================================
# EVENT HANDLERS
# ============================================================

def on_key_press(event, reviewer: DatasetReviewer):
    """
    Global keyboard handler for the Matplotlib window.

    Controls:
    ---------
    n / N      -> Next random batch
    r / R      -> Reshuffle (same as next)
    q / Esc    -> Quit application cleanly
    """
    if event.key in ("n", "N", "r", "R"):
        reviewer.refresh()

    elif event.key in ("q", "escape"):
        plt.close("all")
        sys.exit(0)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """
    Application entry point.
    """

    # ---------------- CONFIGURATION (MODIFY THIS) ----------------

    DATASET_PATH = "/mnt/Training/MLTraining/Projects/Script_testing/phash4/yolo/Potholes"
    DATASET_FORMAT = "yolo"
    IMAGES_PER_BATCH = 4
    
    # ============================================================
    # DON'T MODIFY
    # ============================================================
    reviewer = DatasetReviewer(
        dataset_path=DATASET_PATH,
        dataset_format=DATASET_FORMAT,
        images_per_batch=IMAGES_PER_BATCH,
    )

    reviewer.get_random_batch()
    reviewer.render()

    # Bind keyboard events
    reviewer.figure.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key_press(event, reviewer)
    )

    plt.show()


if __name__ == "__main__":
    main()
