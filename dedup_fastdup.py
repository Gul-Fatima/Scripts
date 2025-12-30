import os
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import fastdup
import matplotlib.pyplot as plt
from shutil import copy2

# ================= CONFIG =================
DATASET_PATH = "/full/path/to/your/yolo_dataset"  # CHANGE THIS
DRY_RUN = True  # Set False to actually delete
SIMILARITY_THRESHOLD = 0.7  # Recommended for YOLO

WORK_DIR = "fastdup_work"
LOG_DIR = "dedup_logs"
PREVIEW_DIR = "duplicate_previews"
# ==========================================


def collect_images(root):
    return [p.resolve() for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]


def label_count(img_path, images_root, labels_root):
    lbl = labels_root / img_path.relative_to(images_root)
    lbl = lbl.with_suffix(".txt")
    if not lbl.exists():
        return 0
    return sum(1 for _ in open(lbl))


def show_side_by_side(paths, save_path):
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, paths):
        img = Image.open(p)
        ax.imshow(img)
        ax.set_title(p.name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    dataset = Path(DATASET_PATH).resolve()
    images_root = dataset / "images"
    labels_root = dataset / "labels"

    assert images_root.exists(), "images/ not found"
    assert labels_root.exists(), "labels/ not found"

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    images = collect_images(images_root)
    print(f"Found {len(images)} images")

    image_set = {p.resolve() for p in images}

    # ---------- Run FastDup ----------
    fd = fastdup.create(
        input_dir=str(images_root),
        work_dir=WORK_DIR
    )
    fd.run(overwrite=True, threshold=SIMILARITY_THRESHOLD)

    # ---------- Collect duplicates ----------
    duplicate_groups = []

    # Exact duplicates (byte identical)
    exact = fd.exact_duplicates()
    print(f"Exact duplicate groups: {len(exact)}")

    for group in exact:
        paths = [Path(p).resolve() for p in group if Path(p).resolve() in image_set]
        if len(paths) > 1:
            duplicate_groups.append(paths)

    # Near duplicates (similarity)
    components = fd.connected_components()
    print(f"Similarity duplicate clusters: {len(components)}")

    for comp in components:
        paths = [Path(p).resolve() for p in comp if Path(p).resolve() in image_set]
        if len(paths) > 1:
            duplicate_groups.append(paths)

    # ---------- Decide keep/remove ----------
    keep = set()
    remove = set()
    report_groups = []

    for group in duplicate_groups:
        scored = sorted(
            group,
            key=lambda p: label_count(p, images_root, labels_root),
            reverse=True
        )
        keep.add(scored[0])
        for p in scored[1:]:
            remove.add(p)
        report_groups.append([str(p) for p in scored])

    # ---------- Logging ----------
    report = {
        "total_images": len(images),
        "duplicates_found": len(report_groups),
        "to_delete": len(remove),
        "dry_run": DRY_RUN,
        "groups": report_groups
    }

    with open(Path(LOG_DIR) / "dedup_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Total duplicates to delete: {len(remove)}")

    # ---------- Previews ----------
    for i, group in enumerate(report_groups):
        paths = [Path(p) for p in group]
        show_side_by_side(paths, Path(PREVIEW_DIR) / f"group_{i+1}.png")

    # ---------- FastDup galleries ----------
    fd.vis.duplicates_gallery()
    fd.vis.component_gallery()
    print(f"HTML galleries saved in {WORK_DIR}/galleries")

    # ---------- Delete ----------
    if DRY_RUN:
        print("DRY-RUN enabled — no files deleted")
        return

    print("Deleting duplicates...")
    for img in remove:
        lbl = labels_root / img.relative_to(images_root)
        lbl = lbl.with_suffix(".txt")

        if img.exists():
            img.unlink()
        if lbl.exists():
            lbl.unlink()

    print("Deduplication completed")


if __name__ == "__main__":
    main()
