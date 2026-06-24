import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from code.tools.darken_test_hsi import rgb_to_hsi


ROOT = Path(r"D:\Pythonfiles\dip\Project\Big_hw")
SOURCE_DATASET = ROOT / "Datasets" / "new_data"
OUTPUT_DATASET = ROOT / "Datasets" / "new_data_hsi_i_darkest_25"
OUTPUT_YAML = ROOT / "Datasets" / "new_hsi_i_darkest_25.yaml"
REPORT_CSV = OUTPUT_DATASET / "hsi_i_darkest_25_report.csv"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PERCENTILE = 25


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def iter_images(dataset_root):
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image split: {image_dir}")
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield split, image_path


def hsi_intensity_mean(image_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    _, _, intensity = rgb_to_hsi(rgb)
    return float(intensity.mean())


def label_path_for_image(source_root, split, image_path):
    relative_image = image_path.relative_to(source_root / "images" / split)
    return source_root / "labels" / split / relative_image.with_suffix(".txt")


def collect_ranked_images(source_root):
    rows = []
    for split, image_path in iter_images(source_root):
        label_path = label_path_for_image(source_root, split, image_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image: {image_path} -> {label_path}")
        rows.append(
            {
                "split": split,
                "image_path": image_path,
                "label_path": label_path,
                "i_mean": hsi_intensity_mean(image_path),
            }
        )
    rows.sort(key=lambda row: (row["i_mean"], row["split"], row["image_path"].name))
    return rows


def copy_subset(rows, source_root, output_root):
    split_counts = {split: 0 for split in SPLITS}
    for row in rows:
        split = row["split"]
        image_relative = row["image_path"].relative_to(source_root / "images" / split)
        label_relative = row["label_path"].relative_to(source_root / "labels" / split)

        target_image = output_root / "images" / split / image_relative
        target_label = output_root / "labels" / split / label_relative
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(row["image_path"], target_image)
        shutil.copy2(row["label_path"], target_label)
        split_counts[split] += 1
    return split_counts


def write_report(rows, source_root):
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "split", "i_mean", "image", "label"])
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["split"],
                    f"{row['i_mean']:.8f}",
                    row["image_path"].relative_to(source_root),
                    row["label_path"].relative_to(source_root),
                ]
            )


def write_yaml(output_yaml, output_dataset):
    content = f"""path: {output_dataset}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: litter
"""
    output_yaml.write_text(content, encoding="utf-8")


def validate_subset(output_root, expected_counts):
    for split, expected_count in expected_counts.items():
        image_count = sum(
            1
            for path in (output_root / "images" / split).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_count = sum(1 for path in (output_root / "labels" / split).rglob("*.txt") if path.is_file())
        if image_count != expected_count:
            raise ValueError(f"{split} image count mismatch: {image_count} != {expected_count}")
        if label_count != expected_count:
            raise ValueError(f"{split} label count mismatch: {label_count} != {expected_count}")


def main():
    ranked = collect_ranked_images(SOURCE_DATASET)
    if not ranked:
        raise FileNotFoundError(f"No images found in: {SOURCE_DATASET}")

    subset_count = int(np.ceil(len(ranked) * PERCENTILE / 100.0))
    selected = ranked[:subset_count]
    threshold = selected[-1]["i_mean"]

    reset_dir(OUTPUT_DATASET)
    split_counts = copy_subset(selected, SOURCE_DATASET, OUTPUT_DATASET)
    validate_subset(OUTPUT_DATASET, split_counts)
    write_report(selected, SOURCE_DATASET)
    write_yaml(OUTPUT_YAML, OUTPUT_DATASET)

    print(f"Total images: {len(ranked)}")
    print(f"Selected darkest {PERCENTILE}%: {len(selected)}")
    print(f"HSI-I mean threshold: {threshold:.8f}")
    for split in SPLITS:
        print(f"{split}: {split_counts[split]}")
    print(f"Output dataset: {OUTPUT_DATASET}")
    print(f"Output yaml: {OUTPUT_YAML}")
    print(f"Report: {REPORT_CSV}")


if __name__ == "__main__":
    main()
