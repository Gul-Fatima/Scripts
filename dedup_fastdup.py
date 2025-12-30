import os
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
from PIL import Image
import fastdup
import matplotlib.pyplot as plt

# ================= CONFIG =================
DATASET_PATH = "/full/path/to/your/yolo_dataset"  # CHANGE
DRY_RUN = True
SIMILARITY_THRESHOLD = 0.7

WORK_DIR = "fastdup_work"
LOG_DIR = "dedup_logs"
PREVIEW_DIR = "duplicate_previews"
# =========================================


def collect_images(root):
    return {p.resolve() for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}}


def label_count(img_path, images_root, labels_root):
    lbl = labels_root / img_path.relative_to(images_root)
    lbl = lbl.with_suffix(".txt")
    if not lbl.exists():
        return 0
    return sum(1 for _ in open(lbl))


def show_group(paths, save_path):
    fig, axes = plt.subplots(1, len(paths), figsize=(4*len(paths), 4))
    if len(paths) == 1:
        axes = [axes]
    for ax, p in zip(axes, paths):
        ax.imshow(Image.open(p))
        ax.set_title(p.name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    dataset = Path(DATASET_PATH).resolve()
    images_root = dataset / "images"
    labels_root = dataset / "labels"

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    images = collect_images(images_root)
    print(f"Total images: {len(images)}")

    # -------- Run FastDup --------
    fd = fastdup.create(input_dir=str(images_root), work_dir=WORK_DIR)
    fd.run(overwrite=True, threshold=SIMILARITY_THRESHOLD)

    # -------- Load CSVs --------
    sim_csv = Path(WORK_DIR) / "similarity.csv"
    dup_csv = Path(WORK_DIR) / "duplicates.csv"

    groups = defaultdict(set)

    if dup_csv.exists():
        df = pd.read_csv(dup_csv)
        for _, r in df.iterrows():
            groups[r["from"]].add(r["to"])
            groups[r["from"]].add(r["from"])

    if sim_csv.exists():
        df = pd.read_csv(sim_csv)
        for _, r in df.iterrows():
            groups[r["from"]].add(r["to"])
            groups[r["from"]].add(r["from"])

    print(f"Duplicate groups found: {len(groups)}")

    keep, remove, report = set(), set(), []

    for g in groups.values():
        paths = [Path(p).resolve() for p in g if Path(p).resolve() in images]
        if len(paths) <= 1:
            continue

        paths.sort(
            key=lambda p: label_count(p, images_root, labels_root),
            reverse=True
        )

        keep.add(paths[0])
        for p in paths[1:]:
            remove.add(p)

        report.append([str(p) for p in paths])

    # -------- Save report --------
    with open(Path(LOG_DIR) / "dedup_report.json", "w") as f:
        json.dump({
            "total_images": len(images),
            "duplicates_found": len(report),
            "to_delete": len(remove),
            "dry_run": DRY_RUN,
            "groups": report
        }, f, indent=2)

    print(f"Images to delete: {len(remove)}")

    # -------- Previews --------
    for i, g in enumerate(report):
        show_group([Path(p) for p in g], Path(PREVIEW_DIR) / f"group_{i+1}.png")

    fd.vis.duplicates_gallery()
    fd.vis.component_gallery()

    if DRY_RUN:
        print("DRY-RUN active — nothing deleted")
        return

    # -------- Delete --------
    for img in remove:
        lbl = labels_root / img.relative_to(images_root)
        lbl = lbl.with_suffix(".txt")
        if img.exists():
            img.unlink()
        if lbl.exists():
            lbl.unlink()

    print("Deduplication complete")


if __name__ == "__main__":
    main()
