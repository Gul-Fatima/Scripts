import os
import json
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copy2

# ---------------- Config ----------------
PHASH_THRESHOLD = 6          # max perceptual hash distance to consider images duplicates
DRY_RUN = False              # set True for dry-run
LOG_DIR = "dedup_logs"
PREVIEW_DIR = "duplicate_previews"
ALL_DUPLICATES_DIR = "all_duplicates"

# --------------- Helper Functions ----------------
def compute_phash(path):
    with Image.open(path) as img:
        return imagehash.phash(img)

def resolution(path):
    with Image.open(path) as img:
        w, h = img.size
    return w * h

def collect_images(images_root):
    # exts = {".jpg", ".jpeg", ".png", ".bmp"}
    exts = {".jpg",".png"}
    return [p for p in images_root.rglob("*") if p.suffix.lower() in exts]

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

# --------------- Deduplication ----------------
def deduplicate_yolo_dataset(dataset_path, dry_run=DRY_RUN):
    dataset_path = Path(dataset_path)
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    os.makedirs(ALL_DUPLICATES_DIR, exist_ok=True)

    images = collect_images(images_root)
    print(f"Found {len(images)} images")

    records = []
    for img in tqdm(images, desc="Hashing images"):
        try:
            records.append({"path": img, "hash": compute_phash(img)})
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

        for r2 in records[i+1:]:
            if r2["path"] in visited:
                continue
            if r1["hash"] - r2["hash"] <= PHASH_THRESHOLD:
                group.append(r2)
                visited.add(r2["path"])

        if len(group) == 1:
            keep.append(group[0])
            continue

        # Keep first image, remove the rest
        keep.append(group[0])
        remove.extend(group[1:])
        duplicate_groups.append(group)

    # -------- LOGGING --------
    report = {
        "total_images": len(images),
        "kept": [str(r["path"]) for r in keep],
        "removed": [str(r["path"]) for r in remove],
        "duplicate_groups": [[str(r["path"]) for r in g] for g in duplicate_groups]
    }

    with open(Path(LOG_DIR)/"dedup_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(Path(LOG_DIR)/"deleted_files.txt", "w") as f:
        for r in remove:
            f.write(str(r["path"]) + "\n")

    with open(Path(LOG_DIR)/"kept_files.txt", "w") as f:
        for r in keep:
            f.write(str(r["path"]) + "\n")

    print(f"Logs written to `{LOG_DIR}/`")
    print(f"Total images: {len(images)}, Kept: {len(keep)}, To delete: {len(remove)}")

    # -------- Preview duplicates --------
    print("Generating side-by-side previews for duplicate groups...")
    for i, group in enumerate(duplicate_groups):
        save_path = Path(PREVIEW_DIR)/f"group_{i+1}.png"
        show_side_by_side(group, save_path=save_path)

        # copy duplicates (except first image) to all_duplicates/group_X/
        group_dir = Path(ALL_DUPLICATES_DIR)/f"group_{i+1}"
        os.makedirs(group_dir, exist_ok=True)
        for r in group[1:]:
            copy2(r["path"], group_dir / r["path"].name)

    # -------- DELETE --------
    if dry_run:
        print("🟡 Dry-run mode: No files deleted. Inspect logs and previews first.")
        return

    print("🗑 Deleting duplicate images & labels...")
    for r in remove:
        img_path = r["path"]
        lbl_path = labels_root / img_path.relative_to(images_root)
        lbl_path = lbl_path.with_suffix(".txt")

        if img_path.exists():
            img_path.unlink()
        if lbl_path.exists():
            lbl_path.unlink()

    print("✅ Duplicate images & labels removed successfully")

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
    DATASET_PATH = "D:/Datasets/my_yolo_dataset"
    DRY_RUN = True  # change to False to actually delete files

    deduplicate_yolo_dataset(DATASET_PATH, dry_run=DRY_RUN)
    print("🎉 Deduplication process completed!")


