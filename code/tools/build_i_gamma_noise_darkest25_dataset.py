import csv
import shutil
import stat
from pathlib import Path

import numpy as np
from PIL import Image

from code.tools.darken_test_hsi import hsi_to_rgb, rgb_to_hsi


ROOT = Path(r"D:\Pythonfiles\dip\Project\Big_hw")
SOURCE_DATASET = ROOT / "Datasets" / "new_data_gap20_valtest_clean"
FULL_OUTPUT = ROOT / "Datasets" / "new_data_gap20_valtest_clean_i_gamma1p5_sigma0p03"
SUBSET_OUTPUT = ROOT / "Datasets" / "new_data_gap20_valtest_clean_i_gamma1p5_sigma0p03_darkest25"
FULL_YAML = ROOT / "Datasets" / "new_gap20_valtest_clean_i_gamma1p5_sigma0p03.yaml"
SUBSET_YAML = ROOT / "Datasets" / "new_gap20_valtest_clean_i_gamma1p5_sigma0p03_darkest25.yaml"
REPORT_CSV = SUBSET_OUTPUT / "i_gamma1p5_sigma0p03_darkest25_report.csv"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
GAMMA = 1.5
SIGMA = 0.03
SEED = 20260614
PERCENTILE = 25


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path, onerror=handle_remove_readonly)
    path.mkdir(parents=True)


def handle_remove_readonly(func, path, exc_info):
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def iter_images(dataset_root):
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image split: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label split: {label_dir}")

        for image_path in sorted(image_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(image_dir)
            label_path = label_dir / relative.with_suffix(".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            yield split, image_path, label_path, relative


def transform_image(image_path, rng):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))

    hue, saturation, intensity = rgb_to_hsi(rgb)
    noise = rng.normal(loc=0.0, scale=SIGMA, size=intensity.shape).astype(np.float32)
    transformed_intensity = np.clip(np.power(intensity, GAMMA) + noise, 0.0, 1.0)
    transformed_rgb = hsi_to_rgb(hue, saturation, transformed_intensity)
    return transformed_rgb, float(transformed_intensity.mean())


def original_i_mean(image_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    _, _, intensity = rgb_to_hsi(rgb)
    return float(intensity.mean())


def save_rgb(rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="JPEG", quality=95)


def copy_label(label_path, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, target_path)


def write_yaml(path, dataset_root):
    content = f"""path: {dataset_root}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: litter
"""
    path.write_text(content, encoding="utf-8")


def collect_source_rows():
    rows = []

    for split, image_path, label_path, relative in iter_images(SOURCE_DATASET):
        rows.append(
            {
                "split": split,
                "source_image": image_path,
                "source_label": label_path,
                "relative": relative.with_suffix(".jpg"),
                "source_i_mean": original_i_mean(image_path),
            }
        )

    return rows


def build_darkest_subset(rows):
    reset_dir(SUBSET_OUTPUT)
    rng = np.random.default_rng(SEED)
    selected_count = int(np.ceil(len(rows) * PERCENTILE / 100.0))
    selected = sorted(rows, key=lambda row: (row["source_i_mean"], row["split"], str(row["relative"])))[:selected_count]
    selected_keys = {(row["split"], row["relative"]) for row in selected}
    transformed_means = {}

    for row in selected:
        split = row["split"]
        relative = row["relative"]
        target_image = SUBSET_OUTPUT / "images" / split / relative
        target_label = SUBSET_OUTPUT / "labels" / split / relative.with_suffix(".txt")
        rgb, i_mean = transform_image(row["source_image"], rng)
        transformed_means[(split, relative)] = i_mean
        target_image.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(rgb, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        copy_label(row["source_label"], target_label)

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "split",
                "source_i_mean",
                "i_mean_after_transform",
                "image",
                "label",
                "selected",
            ]
        )
        ranked = sorted(rows, key=lambda row: (row["source_i_mean"], row["split"], str(row["relative"])))
        for rank, row in enumerate(ranked, start=1):
            split = row["split"]
            relative = row["relative"]
            selected_flag = (split, relative) in selected_keys
            transformed_mean = transformed_means.get((split, relative), "")
            writer.writerow(
                [
                    rank,
                    split,
                    f"{row['source_i_mean']:.8f}",
                    f"{transformed_mean:.8f}" if transformed_mean != "" else "",
                    Path("images") / split / relative,
                    Path("labels") / split / relative.with_suffix(".txt"),
                    int(selected_flag),
                ]
            )

    write_yaml(SUBSET_YAML, SUBSET_OUTPUT)
    return selected


def count_dataset(dataset_root):
    counts = {}
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_count = (
            sum(1 for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if image_dir.exists()
            else 0
        )
        label_count = sum(1 for path in label_dir.rglob("*.txt") if path.is_file()) if label_dir.exists() else 0
        counts[split] = (image_count, label_count)
        if image_count != label_count:
            raise ValueError(f"{dataset_root.name} {split} images/labels mismatch: {image_count} != {label_count}")
    return counts


def main():
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {SOURCE_DATASET}")

    rows = collect_source_rows()
    selected = build_darkest_subset(rows)
    subset_counts = count_dataset(SUBSET_OUTPUT)
    threshold = max(row["source_i_mean"] for row in selected)

    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Formula: I' = I^{GAMMA} + N(0, {SIGMA}^2), clipped to [0, 1]")
    print(f"Random seed: {SEED}")
    print(f"Darkest {PERCENTILE}% selected: {len(selected)}")
    print(f"Selection threshold on source clean data: I_mean <= {threshold:.8f}")
    for split, (image_count, label_count) in subset_counts.items():
        print(f"subset {split}: images={image_count}, labels={label_count}")
    print(f"Subset output: {SUBSET_OUTPUT}")
    print(f"Subset yaml: {SUBSET_YAML}")
    print(f"Report: {REPORT_CSV}")


if __name__ == "__main__":
    main()
