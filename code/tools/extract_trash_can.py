import json
import os
import shutil
from collections import Counter, defaultdict


def clamp01(value):
    return max(0.0, min(1.0, value))


def extract_images_and_labels():
    base_dir = r"D:\Pythonfiles\dip\Project\Big_hw\dataset_original\instance_version"
    output_dir = r"D:\Pythonfiles\dip\Project\Big_hw\Datasets\extracted_trash_4classes"

    target_categories = [
        "trash_bottle",
        "trash_bag",
        "trash_can",
        "trash_cup",
    ]
    name_to_yolo_id = {name: index for index, name in enumerate(target_categories)}

    splits = {
        "train": {
            "json": os.path.join(base_dir, "instances_train_trashcan.json"),
            "img_dir": os.path.join(base_dir, "train"),
        },
        "val": {
            "json": os.path.join(base_dir, "instances_val_trashcan.json"),
            "img_dir": os.path.join(base_dir, "val"),
        },
    }

    total_images = 0
    total_labels = Counter()

    for split_name, paths in splits.items():
        json_path = paths["json"]
        img_dir = paths["img_dir"]
        img_out_dir = os.path.join(output_dir, "images", split_name)
        lbl_out_dir = os.path.join(output_dir, "labels", split_name)

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        if not os.path.exists(json_path):
            print(f"Missing json file: {json_path}")
            continue
        if not os.path.exists(img_dir):
            print(f"Missing image directory: {img_dir}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        id_to_name = {category["id"]: category["name"] for category in data["categories"]}
        target_coco_ids = {
            category_id
            for category_id, name in id_to_name.items()
            if name in name_to_yolo_id
        }

        missing_categories = [name for name in target_categories if name not in id_to_name.values()]
        if missing_categories:
            print(f"Missing categories in {json_path}: {missing_categories}")

        image_id_to_annotations = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] in target_coco_ids:
                image_id_to_annotations[ann["image_id"]].append(ann)

        split_images = 0
        split_labels = Counter()

        for image_info in data["images"]:
            image_id = image_info["id"]
            annotations = image_id_to_annotations.get(image_id)
            if not annotations:
                continue

            filename = image_info["file_name"]
            img_width = image_info["width"]
            img_height = image_info["height"]
            src_img_path = os.path.join(img_dir, filename)

            if not os.path.exists(src_img_path):
                print(f"Missing image: {src_img_path}")
                continue

            shutil.copy2(src_img_path, os.path.join(img_out_dir, filename))

            label_name = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(lbl_out_dir, label_name)

            with open(label_path, "w", encoding="utf-8") as label_file:
                for ann in annotations:
                    category_name = id_to_name[ann["category_id"]]
                    yolo_class_id = name_to_yolo_id[category_name]

                    x_min, y_min, width, height = ann["bbox"]
                    x_center = clamp01((x_min + width / 2.0) / img_width)
                    y_center = clamp01((y_min + height / 2.0) / img_height)
                    width_norm = clamp01(width / img_width)
                    height_norm = clamp01(height / img_height)

                    label_file.write(
                        f"{yolo_class_id} {x_center:.6f} {y_center:.6f} "
                        f"{width_norm:.6f} {height_norm:.6f}\n"
                    )
                    split_labels[category_name] += 1

            split_images += 1

        total_images += split_images
        total_labels.update(split_labels)

    print(f"Done: {output_dir}")


if __name__ == "__main__":
    extract_images_and_labels()
