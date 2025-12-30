import os
import json
from pathlib import Path
from PIL import Image
import fastdup
import matplotlib.pyplot as plt
from shutil import copy2

# ================= CONFIGURATION =================
# CHANGE THIS to point to your YOLO dataset folder
# Your dataset structure should be:
# dataset/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
#     ├── train/
#     ├── val/
#     └── test/
DATASET_PATH = "/full/path/to/your/yolo_dataset"  # <-- CHANGE THIS

DRY_RUN = True  # Set False to actually delete duplicates
LOG_DIR = "dedup_logs"
PREVIEW_DIR = "duplicate_previews"
ALL_DUPLICATES_DIR = "all_duplicates"
FASTDUP_WORK_DIR = "fastdup_work"
SIMILARITY_THRESHOLD = 0.85  # Lower to 0.8 if you want more aggressive dedup
# ==================================================

# ---------------- Helper Functions ----------------
def collect_images(images_root):
    exts = {".jpg", ".png"}
    return [p.resolve() for p in images_root.rglob("*") if p.suffix.lower() in exts]

def show_side_by_side(group, save_path=None):
    n = len(group)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, item in zip(axes, group):
        img = Image.open(item["path"])
        ax.imshow(img)
        ax.set_title(item["path"].name)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# ---------------- Deduplication ----------------
def deduplicate_with_fastdup():
    dataset_path = Path(DATASET_PATH).resolve()
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    os.makedirs(ALL_DUPLICATES_DIR, exist_ok=True)
    os.makedirs(FASTDUP_WORK_DIR, exist_ok=True)

    images = collect_images(images_root)
    print(f"Found {len(images)} images")

    # -------- Run FastDup --------
    print("Running FastDup analysis...")
    try:
        fd = fastdup.create(work_dir=FASTDUP_WORK_DIR, input_dir=str(images_root))
        fd.run(annotations=None, overwrite=True, threshold=SIMILARITY_THRESHOLD)
    except Exception as e:
        print(f"Error running FastDup: {e}")
        return

    # Get duplicate clusters
    print("Finding duplicate clusters...")
    components = fd.connected_components()
    if not components:
        print("No duplicates found!")
        return

    print(f"Found {len(components)} duplicate clusters")

    # Map all images for easy lookup
    records = {img: {"path": img} for img in images}

    keep, remove, duplicate_groups = [], [], []

    for i, component in enumerate(components):
        group = []
        for img_path_str in component:
            # Convert FastDup relative paths to absolute
            img_path = (images_root / img_path_str).resolve()
            if img_path in records:
                group.append(records[img_path])

        if len(group) <= 1:
            keep.extend(group)
            continue

        keep.append(group[0])
        remove.extend(group[1:])
        duplicate_groups.append(group)

    # -------- Logging --------
    report = {
        "dataset_path": str(dataset_path),
        "total_images": len(images),
        "kept": [str(r["path"]) for r in keep],
        "removed": [str(r["path"]) for r in remove],
        "duplicate_groups": [[str(r["path"]) for r in g] for g in duplicate_groups],
        "fastdup_settings": {
            "work_dir": FASTDUP_WORK_DIR,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "components_found": len(components)
        },
        "dry_run": DRY_RUN
    }

    with open(Path(LOG_DIR)/"dedup_report_fastdup.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(Path(LOG_DIR)/"deleted_files.txt", "w") as f:
        for r in remove:
            f.write(str(r["path"]) + "\n")

    with open(Path(LOG_DIR)/"kept_files.txt", "w") as f:
        for r in keep:
            f.write(str(r["path"]) + "\n")

    print(f"Logs written to {LOG_DIR}/")
    print(f"Total images: {len(images)}, Kept: {len(keep)}, To delete: {len(remove)}")

    # -------- Previews & copy duplicates --------
    print("Generating previews for duplicate groups...")
    for i, group in enumerate(duplicate_groups):
        save_path = Path(PREVIEW_DIR)/f"group_{i+1}.png"
        show_side_by_side(group, save_path=save_path)

        group_dir = Path(ALL_DUPLICATES_DIR)/f"group_{i+1}"
        os.makedirs(group_dir, exist_ok=True)
        for r in group[1:]:
            copy2(r["path"], group_dir / r["path"].name)

    # FastDup visualization
    try:
        fd.vis.duplicates_gallery()
        fd.vis.component_gallery()
        print(f"HTML galleries created in {FASTDUP_WORK_DIR}")
    except Exception as e:
        print(f"Could not create FastDup galleries: {e}")

    # -------- Delete duplicates if DRY_RUN=False --------
    if DRY_RUN:
        print("DRY-RUN mode active: no files were deleted.")
        print("Set DRY_RUN=False to delete duplicates and labels.")
        return

    print("Deleting duplicate images & labels...")
    for r in remove:
        img_path = r["path"]
        lbl_path = labels_root / img_path.relative_to(images_root)
        lbl_path = lbl_path.with_suffix(".txt")

        if img_path.exists():
            img_path.unlink()
        if lbl_path.exists():
            lbl_path.unlink()
    print("Duplicate images & labels removed successfully")

    # -------- Regenerate train/val/test txt files --------
    for split in ["train", "val", "test"]:
        split_dir = images_root / split
        if not split_dir.exists():
            continue
        txt_path = dataset_path / f"{split}.txt"
        imgs = collect_images(split_dir)
        with open(txt_path, "w") as f:
            for img in imgs:
                f.write(str(img.relative_to(dataset_path)) + "\n")
    print("train/val/test txt files regenerated successfully")

# ---------------- Main ----------------
if __name__ == "__main__":
    print("="*50)
    print("YOLO Dataset Deduplicator with FastDup")
    print("="*50)

    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        exit(1)

    deduplicate_with_fastdup()
    print("Deduplication process completed.")
