import datumaro as dm
import yaml
import os
from tqdm import tqdm

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
    explicit_mapping=None,
    inplace=False
):

    
    # FIX DATASET STRUCTURE BEFORE LOADING
    ensure_empty_labels(src_dataset)
    ensure_empty_labels(dst_dataset)

    dataset_src = dm.Dataset.import_from(src_dataset, "yolo")
    dataset_dst = dm.Dataset.import_from(dst_dataset, "yolo")

    # LOAD YAML FILES
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
            # safety check
            if src_id not in src_names:
                raise ValueError(f"\n\nInvalid source class id {src_id} in EXPLICIT_MAPPING")

            if dst_id not in dst_names:
                # dst_names[dst_id] = src_names[src_id]
                 raise ValueError(
                        f"\nInvalid destination class id {dst_id} in EXPLICIT_MAPPING.\n"
                        f"Destination dataset only contains ids: {sorted(dst_names.keys())}"
                                )   

    elif mode == "auto":

        last_id = max(dst_names.keys()) if dst_names else -1

        # reverse lookup: class_name -> id
        dst_name_to_id = {v: k for k, v in dst_names.items()}

        for src_id in source_class_ids or []:

            # safety check
            if src_id not in src_names:
                print(f"Warning: source class id {src_id} not found in src dataset")
                continue

            src_name = src_names[src_id]

            # check if class name already exists in destination
            if src_name in dst_name_to_id:

                existing_id = dst_name_to_id[src_name]

                print(
                    f"Class '{src_name}' already exists in destination "
                    f"(id={existing_id}). Skipping remap."
                )

                continue

            # create new class
            last_id += 1

            class_mapping[src_id] = last_id
            dst_names[last_id] = src_name

            print(
                f"Adding new class '{src_name}' "
                f"(src_id={src_id} -> dst_id={last_id})"
            )

        # if nothing mapped, stop early
        if not class_mapping:
            print("\nNo new classes to map. Skipping annotation merge.")
            return

    # -----------------------------
    # UPDATE YAML
    # -----------------------------
    dst_yaml["names"] = dict(sorted(dst_names.items()))
    dst_yaml["nc"] = len(dst_yaml["names"])

    # -----------------------------
    # TRANSFER ANNOTATIONS
    # -----------------------------
    missing_images = []

    for item_a in tqdm(dataset_src, total=len(dataset_src), desc="Merging annotations"):

        item_b = dataset_dst.get(item_a.id, subset=item_a.subset)

        if item_b is None:
            print(f"Warning: {item_a.id} not found in destination dataset, skipping")
            missing_images.append(item_a.id)
            continue

        new_annotations = []

        # existing annotations in destination
        existing = {
            (ann.label, tuple(ann.points))
            for ann in item_b.annotations
        }

        for ann in item_a.annotations:

            if ann.label in class_mapping:

                mapped_label = class_mapping[ann.label]
                new_ann = ann.wrap(label=mapped_label)

                key = (mapped_label, tuple(new_ann.points))

                if key not in existing:
                    new_annotations.append(new_ann)
                    existing.add(key)

        if new_annotations:
            item_b.annotations.extend(new_annotations)
            dataset_dst.put(item_b)

    # -----------------------------
    # EXPORT DATASET
    # -----------------------------
    # dataset_dst.export(output_dataset, "yolo_ultralytics", save_media=True)
    export_path = dst_dataset if inplace else output_dataset
    dataset_dst.export(export_path, "yolo_ultralytics", save_media=True)

    # -----------------------------
    # SAVE UPDATED YAML
    # -----------------------------
    # Save YAML to the same dataset location used for export
    yaml_output = os.path.join(export_path, "data.yaml")

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
        print(f"Total skipped: {len(missing_images)}")


'''-------------------------------------------------
USER CONFIGURATION

Modify the following parameters before running:

    SRC_DATASET   → source dataset directory
    DST_DATASET   → destination dataset directory
    OUTPUT_DATASET → where merged dataset will be saved

    INPLACE:
        If True → destination dataset will be modified directly
        If False → merged dataset will be saved in OUTPUT_DATASET

    MODE:
        explicit mode:
            user provides exact mapping

        auto mode:
            new classes are appended after the last existing class in destination dataset

    SOURCE_CLASS_IDS:
        used only in auto mode

    EXPLICIT_MAPPING:
        used only in explicit mode
-------------------------------------------------'''
if __name__ == "__main__":
    SRC_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/src_data"
    DST_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/dst_data"
    OUTPUT_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/mgd_dataset"

    INPLACE = True

    SRC_YAML = os.path.join(SRC_DATASET, "data.yaml")
    DST_YAML = os.path.join(DST_DATASET, "data.yaml")

    MODE = "explicit"   #auto or explicit

    SOURCE_CLASS_IDS = [3,4,5,14] #list of source classes, only used in auto mode

    EXPLICIT_MAPPING = {
        1:7
        
    } # {source_classid: destination_classid} only used in explicit mode


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
        EXPLICIT_MAPPING,
        INPLACE
    )import datumaro as dm
import yaml
import os
from tqdm import tqdm

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
    explicit_mapping=None,
    inplace=False
):

    
    # FIX DATASET STRUCTURE BEFORE LOADING
    ensure_empty_labels(src_dataset)
    ensure_empty_labels(dst_dataset)

    dataset_src = dm.Dataset.import_from(src_dataset, "yolo")
    dataset_dst = dm.Dataset.import_from(dst_dataset, "yolo")

    # LOAD YAML FILES
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
            # safety check
            if src_id not in src_names:
                raise ValueError(f"\n\nInvalid source class id {src_id} in EXPLICIT_MAPPING")

            if dst_id not in dst_names:
                # dst_names[dst_id] = src_names[src_id]
                 raise ValueError(
                        f"\nInvalid destination class id {dst_id} in EXPLICIT_MAPPING.\n"
                        f"Destination dataset only contains ids: {sorted(dst_names.keys())}"
                                )   

    elif mode == "auto":

        last_id = max(dst_names.keys()) if dst_names else -1

        # reverse lookup: class_name -> id
        dst_name_to_id = {v: k for k, v in dst_names.items()}

        for src_id in source_class_ids or []:

            # safety check
            if src_id not in src_names:
                print(f"Warning: source class id {src_id} not found in src dataset")
                continue

            src_name = src_names[src_id]

            # check if class name already exists in destination
            if src_name in dst_name_to_id:

                existing_id = dst_name_to_id[src_name]

                print(
                    f"Class '{src_name}' already exists in destination "
                    f"(id={existing_id}). Skipping remap."
                )

                continue

            # create new class
            last_id += 1

            class_mapping[src_id] = last_id
            dst_names[last_id] = src_name

            print(
                f"Adding new class '{src_name}' "
                f"(src_id={src_id} -> dst_id={last_id})"
            )

        # if nothing mapped, stop early
        if not class_mapping:
            print("\nNo new classes to map. Skipping annotation merge.")
            return

    # -----------------------------
    # UPDATE YAML
    # -----------------------------
    dst_yaml["names"] = dict(sorted(dst_names.items()))
    dst_yaml["nc"] = len(dst_yaml["names"])

    # -----------------------------
    # TRANSFER ANNOTATIONS
    # -----------------------------
    missing_images = []

    for item_a in tqdm(dataset_src, total=len(dataset_src), desc="Merging annotations"):

        item_b = dataset_dst.get(item_a.id, subset=item_a.subset)

        if item_b is None:
            print(f"Warning: {item_a.id} not found in destination dataset, skipping")
            missing_images.append(item_a.id)
            continue

        new_annotations = []

        # existing annotations in destination
        existing = {
            (ann.label, tuple(ann.points))
            for ann in item_b.annotations
        }

        for ann in item_a.annotations:

            if ann.label in class_mapping:

                mapped_label = class_mapping[ann.label]
                new_ann = ann.wrap(label=mapped_label)

                key = (mapped_label, tuple(new_ann.points))

                if key not in existing:
                    new_annotations.append(new_ann)
                    existing.add(key)

        if new_annotations:
            item_b.annotations.extend(new_annotations)
            dataset_dst.put(item_b)

    # -----------------------------
    # EXPORT DATASET
    # -----------------------------
    # dataset_dst.export(output_dataset, "yolo_ultralytics", save_media=True)
    export_path = dst_dataset if inplace else output_dataset
    dataset_dst.export(export_path, "yolo_ultralytics", save_media=True)

    # -----------------------------
    # SAVE UPDATED YAML
    # -----------------------------
    # Save YAML to the same dataset location used for export
    yaml_output = os.path.join(export_path, "data.yaml")

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
        print(f"Total skipped: {len(missing_images)}")


'''-------------------------------------------------
USER CONFIGURATION

Modify the following parameters before running:

    SRC_DATASET   → source dataset directory
    DST_DATASET   → destination dataset directory
    OUTPUT_DATASET → where merged dataset will be saved

    INPLACE:
        If True → destination dataset will be modified directly
        If False → merged dataset will be saved in OUTPUT_DATASET

    MODE:
        explicit mode:
            user provides exact mapping

        auto mode:
            new classes are appended after the last existing class in destination dataset

    SOURCE_CLASS_IDS:
        used only in auto mode

    EXPLICIT_MAPPING:
        used only in explicit mode
-------------------------------------------------'''
if __name__ == "__main__":
    SRC_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/src_data"
    DST_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/dst_data"
    OUTPUT_DATASET = "/mnt/Training/MLTraining/Projects/Script_testing/Annotation_map/mgd_dataset"

    INPLACE = True

    SRC_YAML = os.path.join(SRC_DATASET, "data.yaml")
    DST_YAML = os.path.join(DST_DATASET, "data.yaml")

    MODE = "explicit"   #auto or explicit

    SOURCE_CLASS_IDS = [3,4,5,14] #list of source classes, only used in auto mode

    EXPLICIT_MAPPING = {
        1:7
        
    } # {source_classid: destination_classid} only used in explicit mode


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
        EXPLICIT_MAPPING,
        INPLACE
    )
