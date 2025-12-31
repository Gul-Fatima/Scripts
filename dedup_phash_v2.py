"""
Image Deduplication Utility (Perceptual Hashing)

Purpose
-------
Provides a utility for identifying and removing visually duplicate images
using perceptual hashing (pHash). The tool is intended for dataset hygiene and
preprocessing prior to training.

Supported Use Cases
-------------------
- YOLO-format datasets (images and labels with automatic split regeneration)
- Plain directories containing raw image files

Limitations
-----------
- Duplicate detection uses pairwise pHash comparisons (O(N²))
"""

import os
import json
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copy2


# ============================================================
# Helper functions 
# ============================================================

def compute_phash(path):
    """Compute perceptual hash for an image."""
    with Image.open(path) as img:
        return imagehash.phash(img)


def collect_images_from_folder(folder):
    """Collect all images recursively from a folder."""
    exts = {".jpg", ".png"}
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]


def collect_yolo_images(images_root):
    """Collect YOLO images from images/ directory."""
    return collect_images_from_folder(images_root)


def show_side_by_side(group, save_path):
    """Save side-by-side preview of duplicate images."""
    n = len(group)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, group):
        img = Image.open(item["path"])
        ax.imshow(img)
        ax.set_title(item["path"].name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# Deduplication logic (shared by YOLO + RAW)
# ============================================================

def deduplicate_images(
    images,
    phash_threshold,
    dry_run,
    log_dir,
    preview_dir,
    all_duplicates_dir,
    labels_root=None,
    images_root=None,
    regenerate_txt=False,
    dataset_path=None
):
    """
    Deduplicates a list of images using perceptual hashing (pHash).

    Parameters
    ----------
    images : list[Path]
        List of image file paths to process. These can come from a YOLO dataset
        (images/train, images/val, etc.) or from a plain folder of raw images.

    phash_threshold : int
        Maximum allowed perceptual hash (pHash) distance to consider two images
        as duplicates. Lower values are stricter (fewer false positives),
        higher values are more tolerant to small visual changes.

    dry_run : bool
        If True, no files are deleted. The script only generates logs and
        preview images so results can be reviewed safely.
        If False, duplicate images (and labels if applicable) are deleted.

    log_dir : str or Path
        Directory where logs are written:
        - dedup_report.json (full summary)
        - kept_files.txt
        - deleted_files.txt

    preview_dir : str or Path
        Directory where side-by-side preview images of duplicate groups
        are saved. Each group is saved as group_X.png.

    all_duplicates_dir : str or Path
        Directory where all duplicate images (excluding the kept one)
        are copied for manual inspection or backup.

    labels_root : Path, optional
        Path to the YOLO labels directory (dataset/labels).
        If provided, the corresponding .txt label files for deleted images
        are also removed. Should be None for raw image folders.

    images_root : Path, optional
        Path to the YOLO images directory (dataset/images).
        Required only when labels_root is provided, so the script can map
        image paths to their matching label paths.

    regenerate_txt : bool, optional
        If True, regenerates train.txt / val.txt / test.txt files after
        deduplication. This is only relevant for YOLO datasets.

    dataset_path : Path, optional
        Root path of the YOLO dataset.
        Required only when regenerate_txt is True, so regenerated split
        files are written to the correct location.
    """

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(all_duplicates_dir, exist_ok=True)

    records = []
    for img in tqdm(images, desc="Hashing images"):
        try:
            records.append({
                "path": img,
                "hash": compute_phash(img)
            })
        except Exception as e:
            print(f"Skipping {img}: {e}")

    visited = set()
    keep = []
    remove = []
    duplicate_groups = []

    print("Grouping duplicates...")
    for i, r1 in enumerate(records):
        if r1["path"] in visited:
            continue

        group = [r1]
        visited.add(r1["path"])

        for r2 in records[i + 1:]:
            if r2["path"] in visited:
                continue
            if r1["hash"] - r2["hash"] <= phash_threshold:
                group.append(r2)
                visited.add(r2["path"])

        if len(group) == 1:
            keep.append(group[0])
        else:
            # Keep the first encountered image in the group.
            # (This can be replaced with a resolution-based choice if needed.)
            keep.append(group[0])         
            remove.extend(group[1:])      
            duplicate_groups.append(group)

    # ---------------- Logging ----------------
    report = {
        "total_images": len(images),
        "kept": [str(r["path"]) for r in keep],
        "removed": [str(r["path"]) for r in remove],
        "duplicate_groups": [
            [str(r["path"]) for r in g] for g in duplicate_groups
        ]
    }

    with open(Path(log_dir) / "dedup_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(Path(log_dir) / "deleted_files.txt", "w") as f:
        for r in remove:
            f.write(str(r["path"]) + "\n")

    with open(Path(log_dir) / "kept_files.txt", "w") as f:
        for r in keep:
            f.write(str(r["path"]) + "\n")

    print(f"Logs written to {log_dir}")
    print(f"Total: {len(images)} | Kept: {len(keep)} | Removed: {len(remove)}")

    # ---------------- Previews & duplicate copies ----------------
    for i, group in enumerate(duplicate_groups):
        preview_path = Path(preview_dir) / f"group_{i+1}.png"
        show_side_by_side(group, preview_path)

        group_dir = Path(all_duplicates_dir) / f"group_{i+1}"
        os.makedirs(group_dir, exist_ok=True)

        for r in group[1:]:
            copy2(r["path"], group_dir / r["path"].name)

    # ---------------- Deletion ----------------
    if dry_run:
        print("Dry-run enabled: no files were deleted.")
        return

    print("Deleting duplicate images...")
    for r in remove:
        img_path = r["path"]
        img_path.unlink(missing_ok=True)

        if labels_root and images_root:
            # Map image path to its corresponding YOLO label file
            lbl_path = labels_root / img_path.relative_to(images_root)
            lbl_path = lbl_path.with_suffix(".txt")
            lbl_path.unlink(missing_ok=True)

    # ---------------- Regenerate YOLO txt files ----------------
    if regenerate_txt and dataset_path and images_root:
        for split in ["train", "val", "test"]:
            split_dir = images_root / split
            if not split_dir.exists():
                continue

            txt_path = dataset_path / f"{split}.txt"
            imgs = collect_images_from_folder(split_dir)

            with open(txt_path, "w") as f:
                for img in imgs:
                    f.write(str(img.relative_to(dataset_path)) + "\n")

        print("train/val/test txt files regenerated.")


# ============================================================
# Mode-specific wrappers
# ============================================================

def deduplicate_yolo_dataset(dataset_path, **kwargs):
    dataset_path = Path(dataset_path)
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"

    images = collect_yolo_images(images_root)
    print(f"Found {len(images)} YOLO images")

    deduplicate_images(
        images=images,
        images_root=images_root,
        labels_root=labels_root,
        regenerate_txt=True,
        dataset_path=dataset_path,
        **kwargs
    )


def deduplicate_raw_images(folder_path, **kwargs):
    images = collect_images_from_folder(folder_path)
    print(f"Found {len(images)} raw images")

    deduplicate_images(
        images=images,
        **kwargs
    )

def run_deduplication(
    mode,
    yolo_dataset_path,
    raw_images_path,
    phash_threshold,
    dry_run,
    log_dir,
    preview_dir,
    all_duplicates_dir
):
    """
    Executes deduplication based on selected mode.

    This function contains all execution logic so `main` stays
    clean and configuration-only.
    """

    if mode == "yolo":
        if not yolo_dataset_path:
            raise ValueError("YOLO_DATASET_PATH must be set for YOLO mode")

        deduplicate_yolo_dataset(
            dataset_path=yolo_dataset_path,
            phash_threshold=phash_threshold,
            dry_run=dry_run,
            log_dir=log_dir,
            preview_dir=preview_dir,
            all_duplicates_dir=all_duplicates_dir
        )

    elif mode == "raw":
        if not raw_images_path:
            raise ValueError("RAW_IMAGES_PATH must be set for RAW mode")

        deduplicate_raw_images(
            folder_path=raw_images_path,
            phash_threshold=phash_threshold,
            dry_run=dry_run,
            log_dir=log_dir,
            preview_dir=preview_dir,
            all_duplicates_dir=all_duplicates_dir
        )

    else:
        raise ValueError("MODE must be either 'yolo' or 'raw'")

# ============================================================
# Example Usage 
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # CONFIGURATION (EDIT ONLY THIS SECTION)
    # ========================================================

    MODE = "raw"  # "yolo" or "raw"

    YOLO_DATASET_PATH = r"D:\Datasets\my_yolo_dataset" # Set this, if MODE = "yolo"
    RAW_IMAGES_PATH = r"D:\Datasets\raw_images" # Set this, if MODE = "raw"

    PHASH_THRESHOLD = 6
    DRY_RUN = True

    LOG_DIR = "dedup_logs"
    PREVIEW_DIR = "duplicate_previews"
    ALL_DUPLICATES_DIR = "all_duplicates"

    # ========================================================
    # EXECUTION (DO NOT MODIFY)
    # ========================================================

    run_deduplication(
        mode=MODE,
        yolo_dataset_path=YOLO_DATASET_PATH,
        raw_images_path=RAW_IMAGES_PATH,
        phash_threshold=PHASH_THRESHOLD,
        dry_run=DRY_RUN,
        log_dir=LOG_DIR,
        preview_dir=PREVIEW_DIR,
        all_duplicates_dir=ALL_DUPLICATES_DIR
    )

    print("Deduplication process completed.")

