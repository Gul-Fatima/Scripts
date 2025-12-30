import os
import json
import shutil
from pathlib import Path
from PIL import Image
import fastdup
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================ HARDCODED CONFIGURATION ================
# Modify these values to match your setup
DATASET_PATH = "/path/to/your/yolo_dataset"  # CHANGE THIS to your dataset path
PHASH_THRESHOLD = 6
AUG_KEYWORDS = ["aug", "rot", "flip", "color", "bright", "dark", "hsv", "adjusted", "modified"]
DRY_RUN = True  # Set to False when ready to actually delete files
LOG_DIR = "dedup_logs"
PREVIEW_DIR = "duplicate_previews"
FASTDUP_WORK_DIR = "fastdup_work"
SIMILARITY_THRESHOLD = 0.9  # Fastdup similarity threshold (0.0 to 1.0)
# ========================================================

# --------------- Helper Functions ----------------
def is_augmented(name):
    name = name.lower()
    return any(k in name for k in AUG_KEYWORDS)

def resolution(path):
    with Image.open(path) as img:
        w, h = img.size
    return w * h

def collect_images(images_root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return [p for p in images_root.rglob("*") if p.suffix.lower() in exts]

def show_side_by_side(group, save_path=None):
    n = len(group)
    if n == 0:
        return
    
    fig, axes = plt.subplots(1, min(n, 5), figsize=(4*min(n, 5), 4))
    if n == 1:
        axes = [axes]
    
    for idx, (ax, item) in enumerate(zip(axes, group[:5])):
        try:
            img = Image.open(item["path"])
            ax.imshow(img)
            ax.set_title(f"{item['path'].name}\n{img.size[0]}x{img.size[1]}")
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{item['path'].name}", ha='center', va='center')
            ax.axis('off')
    
    plt.suptitle(f"Duplicate Group ({n} images)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_keep_priority(record):
    """Determine priority for keeping an image (lower = better to keep)."""
    path = record["path"]
    name = path.name.lower()
    score = 0
    
    # Original images get highest priority
    if "original" in name or "orig" in name or not is_augmented(name):
        score -= 100
    
    # Augmented images get penalty
    if is_augmented(name):
        score += 50
        
        # Additional penalties for specific augmentations
        if "rot" in name or "flip" in name:
            score += 30
        if "color" in name or "hsv" in name:
            score += 20
    
    # Higher resolution gets priority
    try:
        res = resolution(path)
        score -= res / 1000000  # Higher resolution = lower score (better)
    except:
        pass
    
    # Shorter filenames might be originals
    score += len(name) * 2
    
    return score

# --------------- FastDup Deduplication ----------------
def deduplicate_with_fastdup():
    dataset_path = Path(DATASET_PATH)
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"
    
    print(f"📁 Processing dataset at: {dataset_path}")
    print(f"📸 Images directory: {images_root}")
    print(f"🏷️ Labels directory: {labels_root}")
    
    if not images_root.exists():
        print(f"❌ Error: Images directory not found: {images_root}")
        return
    
    if not labels_root.exists():
        print(f"⚠️ Warning: Labels directory not found: {labels_root}")
    
    # Create necessary directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    os.makedirs(FASTDUP_WORK_DIR, exist_ok=True)
    
    # Collect all images
    images = collect_images(images_root)
    print(f"📸 Found {len(images)} images")
    
    if len(images) == 0:
        print("⚠️ No images found. Exiting.")
        return
    
    # -------- Run FastDup --------
    print("🚀 Running fastdup analysis...")
    print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Work directory: {FASTDUP_WORK_DIR}")
    
    # Configure and run fastdup
    try:
        fd = fastdup.create(work_dir=FASTDUP_WORK_DIR, 
                           input_dir=str(images_root))
        
        # Run the analysis
        fd.run(annotations=None,  # We're not using annotations for deduplication
               overwrite=True,    # Overwrite previous runs
               threshold=SIMILARITY_THRESHOLD)  # Similarity threshold
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
    
    # Create a mapping from image path to record
    records = {}
    for img_path in images:
        records[img_path] = {
            "path": img_path,
            "name": img_path.name
        }
    
    # Process each cluster
    keep = []
    remove = []
    duplicate_groups = []
    
    for i, component in enumerate(components):
        if len(component) <= 1:
            # Single image - keep it
            for img_path_str in component:
                img_path = Path(img_path_str)
                if img_path in records:
                    keep.append(records[img_path])
            continue
        
        # Create group from component
        group = []
        for img_path_str in component:
            img_path = Path(img_path_str)
            if img_path in records:
                group.append(records[img_path])
        
        if len(group) <= 1:
            continue
            
        # Sort group by priority (best to keep first)
        group.sort(key=lambda r: get_keep_priority(r))
        
        # Keep the best one, remove the rest
        keep.append(group[0])
        remove.extend(group[1:])
        duplicate_groups.append(group)
    
    # Add any images not in components (unique images)
    all_processed = set([r["path"] for r in keep + remove])
    for img_path, record in records.items():
        if img_path not in all_processed:
            keep.append(record)
    
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
    
    print(f"🧾 Deduplication report written to `{report_path}`")
    print(f"📊 Total images: {len(images)}")
    print(f"💾 Images kept: {len(keep)}")
    print(f"🗑️ Images to delete: {len(remove)}")
    
    # -------- Preview duplicates --------
    print("🖼 Generating side-by-side previews for duplicate groups...")
    preview_count = min(10, len(duplicate_groups))
    for i in range(preview_count):
        save_path = Path(PREVIEW_DIR)/f"group_{i+1}.png"
        show_side_by_side(duplicate_groups[i], save_path=save_path)
    print(f"   Created {preview_count} preview images in '{PREVIEW_DIR}'")
    
    # Also create fastdup's own visualization
    print("📊 Creating fastdup visualization galleries...")
    try:
        # Create similarity gallery
        fd.vis.duplicates_gallery()
        
        # Create components gallery
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
    deleted_count = 0
    for r in tqdm(remove, desc="Deleting files"):
        img_path = r["path"]
        lbl_path = labels_root / img_path.relative_to(images_root)
        lbl_path = lbl_path.with_suffix(".txt")
        
        if img_path.exists():
            img_path.unlink()
            deleted_count += 1
        if lbl_path.exists():
            lbl_path.unlink()
    
    print(f"✅ Removed {deleted_count} duplicate images and their labels")
    
    # -------- Regenerate train/val/test txt files --------
    print("📄 Regenerating dataset index files...")
    splits_updated = 0
    for split in ["train", "val", "test"]:
        split_dir = images_root / split
        if not split_dir.exists():
            continue
        txt_path = dataset_path / f"{split}.txt"
        imgs = collect_images(split_dir)
        with open(txt_path, "w") as f:
            for img in imgs:
                # Write path relative to dataset root for YOLO
                rel_path = img.relative_to(dataset_path)
                f.write(str(rel_path) + "\n")
        splits_updated += 1
        print(f"   Updated {split}.txt with {len(imgs)} images")
    
    print(f"📝 Updated {splits_updated} dataset split files")
    print("🎉 Deduplication with fastdup completed successfully!")
    print(f"📈 Check {FASTDUP_WORK_DIR}/atrain_stats.csv for detailed statistics")

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
