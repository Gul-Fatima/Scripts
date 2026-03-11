import datumaro as dm
import yaml
import os


# -------------------------------------------------
# FIX DATASET: ensure every image has a label file
# -------------------------------------------------

def ensure_empty_labels(dataset_path):

    for subset in ["train", "val", "test"]:

        images_dir = os.path.join(dataset_path, "images", subset)
        labels_dir = os.path.join(dataset_path, "labels", subset)

        if not os.path.exists(images_dir):
            continue

        os.makedirs(labels_dir, exist_ok=True)

        for img in os.listdir(images_dir):

            name = os.path.splitext(img)[0]
            label_path = os.path.join(labels_dir, name + ".txt")

            if not os.path.exists(label_path):
                open(label_path, "w").close()


def merge_annotations(
    src_dataset,
    dst_dataset,
    output_dataset,
    src_yaml_path,
    dst_yaml_path,
    mode,
    source_class_ids=None,
    explicit_mapping=None
):

    # -------------------------------------------------
    # FIX DATASET STRUCTURE BEFORE LOADING
    # -------------------------------------------------

    ensure_empty_labels(src_dataset)
    ensure_empty_labels(dst_dataset)

    dataset_src = dm.Dataset.import_from(src_dataset, "yolo")
    dataset_dst = dm.Dataset.import_from(dst_dataset, "yolo")

    # -----------------------------
    # LOAD YAML FILES
    # -----------------------------
    with open(src_yaml_path, "r") as f:
        src_yaml = yaml.safe_load(f)

    with open(dst_yaml_path, "r") as f:
        dst_yaml = yaml.safe_load(f)

    src_names = src_yaml["names"]
    dst_names = dst_yaml["names"]

    if isinstance(src_names, list):
        src_names = {i: n for i, n in enumerate(src_names)}

    if isinstance(dst_names, list):
        dst_names = {i: n for i, n in enumerate(dst_names)}

    # -----------------------------
    # CREATE CLASS MAPPING
    # -----------------------------
    class_mapping = {}

    if mode == "explicit":

        class_mapping = explicit_mapping or {}

        for src_id, dst_id in class_mapping.items():
            if dst_id not in dst_names:
                dst_names[dst_id] = src_names[src_id]

    elif mode == "auto":

        last_id = max(dst_names.keys()) if dst_names else -1

        for src_id in source_class_ids or []:
            last_id += 1
            class_mapping[src_id] = last_id
            dst_names[last_id] = src_names[src_id]

    else:
        raise ValueError("Mode must be 'auto' or 'explicit'")

    # -----------------------------
    # UPDATE YAML
    # -----------------------------
    dst_yaml["names"] = dict(sorted(dst_names.items()))
    dst_yaml["nc"] = len(dst_yaml["names"])

    # -----------------------------
    # TRANSFER ANNOTATIONS
    # -----------------------------
    missing_images = []

    for item_a in dataset_src:

        item_b = dataset_dst.get(item_a.id, subset=item_a.subset)

        if item_b is None:
            print(f"Warning: {item_a.id} not found in destination dataset, skipping")
            missing_images.append(item_a.id)
            continue

        new_annotations = []

        for ann in item_a.annotations:

            if ann.label in class_mapping:
                mapped_label = class_mapping[ann.label]
                new_ann = ann.wrap(label=mapped_label)
                new_annotations.append(new_ann)

        # if new_annotations:
        #     item_b.annotations.extend(new_annotations)

        if new_annotations:
            item_b.annotations.extend(new_annotations)
            dataset_dst.put(item_b)

    # -----------------------------
    # EXPORT DATASET
    # -----------------------------
    dataset_dst.export(output_dataset, "yolo_ultralytics", save_media=True)

    # -----------------------------
    # SAVE UPDATED YAML
    # -----------------------------
    yaml_output = os.path.join(output_dataset, "data.yaml")

    with open(yaml_output, "w") as f:
        yaml.safe_dump(dst_yaml, f, sort_keys=False)

    # -----------------------------
    # SUMMARY
    # -----------------------------
    print("\nMerge completed")
    print("Class mapping used:")
    print(class_mapping)

    if missing_images:
        print("\nSkipped images:")
        for img in missing_images:
            print(img)


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

SRC_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/src_data"
DST_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/dst_data"
OUTPUT_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/mgd_dataset"

SRC_YAML = os.path.join(SRC_DATASET, "data.yaml")
DST_YAML = os.path.join(DST_DATASET, "data.yaml")

MODE = "explicit"   #auto or explicit

SOURCE_CLASS_IDS = [31, 32]

EXPLICIT_MAPPING = {
    5: 1,
    6: 2
}

# -------------------------------------------------
# RUN SCRIPT
# -------------------------------------------------

merge_annotations(
    SRC_DATASET,
    DST_DATASET,
    OUTPUT_DATASET,
    SRC_YAML,
    DST_YAML,
    MODE,
    SOURCE_CLASS_IDS,
    EXPLICIT_MAPPING
)
