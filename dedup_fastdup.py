import os
import json
from pathlib import Path
from PIL import Image
import fastdup
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import copy2

# ================ HARDCODED CONFIGURATION ================
# Modify these values to match your setup
DATASET_PATH = "/path/to/your/yolo_dataset"  # CHANGE THIS to your dataset path
DRY_RUN = True                               # Set to False when ready to actually delete files
LOG_DIR = "dedup_logs"
PREVIEW_DIR = "duplicate_previews"
ALL_DUPLICATES_DIR = "all_duplicates"
FASTDUP_WORK_DIR = "fastdup_work"
SIMILARITY_THRESHOLD = 0.9                   # Fastdup similarity threshold (0.0 to 1.0)
# ========================================================

# --------------- Helper Functions ----------------
def collect_images(images_root):
    # Same as your original
    exts = {".jpg", ".png"}
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

# --------------- FastDup Deduplication ----------------
def deduplicate_with_fastdup():
    dataset_path = Path(DATASET_PATH)
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    os.makedirs(ALL_DUPLICATES_DIR, exist_ok=True)
    os.makedirs(FASTDUP_WORK_DIR, exist_ok=True)

    images = collect_images(images_root)
    print(f"📸 Found {len(images)} images")

    # -------- Run FastDup --------
    print("🚀 Running fastdup analysis...")
    print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Work directory: {FASTDUP_WORK_DIR}")
    
    try:
        # Configure and run fastdup
        fd = fastdup.create(work_dir=FASTDUP_WORK_DIR, 
                           input_dir=str(images_root))
        
        # Run the analysis - generates embeddings and finds similarities
        fd.run(annotations=None,    # Not using annotations for deduplication
               overwrite=True,      # Overwrite previous runs
               threshold=SIMILARITY_THRESHOLD)
    except Exception as e:
        print(f"❌ Error running fastdup: {e}")
        print("   Make sure fastdup is installed: pip install fastdup")
        return

    # Get connected components (clusters of similar images)
    print("🔍 Finding duplicate clusters...")
    components = fd.connected_components()
    
    if components is None or len(components) == 0:
        print("✅ No duplicates found by fastdup!")
        return
    
    # -------- Process Results --------
    print(f"📊 Found {len(components)} duplicate clusters")
    
    # Create a simple mapping for easier processing
    records = {}
    for img_path in images:
        records[img_path] = {"path": img_path}

    keep = []
    remove = []
    duplicate_groups = []

    for i, component in enumerate(components):
        # Convert component (list of paths) to our record format
        group = []
        for img_path_str in component:
            img_path = Path(img_path_str)
            if img_path in records:
                group.append(records[img_path])
        
        # Skip single-image clusters
        if len(group) <= 1:
            keep.extend(group)  # Unique images are kept
            continue
        
        # Keep FIRST image, remove the rest (exactly like your phash logic)
        keep.append(group[0])
        remove.extend(group[1:])
        duplicate_groups.append(group)

    # -------- LOGGING --------
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

    report_path = Path(LOG_DIR)/"dedup_report_fastdup.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(Path(LOG_DIR)/"deleted_files.txt", "w") as f:
        for r in remove:
            f.write(str(r["path"]) + "\n")

    with open(Path(LOG_DIR)/"kept_files.txt", "w") as f:
        for r in keep:
            f.write(str(r["path"]) + "\n")

    print(f"📄 Logs written to `{LOG_DIR}/`")
    print(f"📊 Total images: {len(images)}, Kept: {len(keep)}, To delete: {len(remove)}")

    # -------- Preview duplicates --------
    print("🖼 Generating side-by-side previews for duplicate groups...")
    for i, group in enumerate(duplicate_groups):
        save_path = Path(PREVIEW_DIR)/f"group_{i+1}.png"
        show_side_by_side(group, save_path=save_path)

        # Copy duplicates (except first image) to all_duplicates/group_X/
        group_dir = Path(ALL_DUPLICATES_DIR)/f"group_{i+1}"
        os.makedirs(group_dir, exist_ok=True)
        for r in group[1:]:
            copy2(r["path"], group_dir / r["path"].name)

    # Also create fastdup's own visualization
    print("📊 Creating fastdup visualization galleries...")
    try:
        fd.vis.duplicates_gallery()
        fd.vis.component_gallery()
        print(f"   HTML galleries created in '{FASTDUP_WORK_DIR}'")
    except Exception as e:
        print(f"⚠️ Could not create fastdup galleries: {e}")

    # -------- DELETE --------
    if DRY_RUN:
        print("\n🟡 DRY-RUN MODE: No files will be deleted.")
        print("   Set DRY_RUN = False in the script to enable deletion.")
        print(f"   Check '{LOG_DIR}/' for the report and '{PREVIEW_DIR}/' for previews.")
        print(f"   Check '{FASTDUP_WORK_DIR}/' for detailed fastdup outputs.")
        return

    print("\n🗑 Deleting duplicate images & labels...")
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
    print("📄 train/val/test txt files regenerated successfully")

# ---------------- Main ----------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO Dataset Deduplicator with FastDup")
    print("=" * 50)
    
    # Validate the dataset path
    if not Path(DATASET_PATH).exists():
        print(f"❌ Error: Dataset path does not exist: {DATASET_PATH}")
        print(f"Please update DATASET_PATH in the script to point to your YOLO dataset.")
        exit(1)
    
    deduplicate_with_fastdup()
    print("🎉 Deduplication process completed!")
