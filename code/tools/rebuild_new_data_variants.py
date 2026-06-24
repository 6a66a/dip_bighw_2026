import shutil
import stat
from pathlib import Path

from code.tools.darken_test_hsi import IMAGE_EXTENSIONS as DARKEN_IMAGE_EXTENSIONS
from code.tools.darken_test_hsi import darken_image_hsi
from code.tools.image_deal_tools import apply_gamma_to_dataset


ROOT = Path(r"D:\Pythonfiles\dip\Project\Big_hw")
SOURCE_DATASET = ROOT / "Datasets" / "new_data"
DARK_DATASET = ROOT / "Datasets" / "new_data_0.3"
GAMMA_DATASET = ROOT / "Datasets" / "new_data_0.3_gamma_0.8"
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 419, "val": 87, "test": 132}


def reset_dir(path):
    if path.exists():
        remove_tree(path)
    path.mkdir(parents=True)


def handle_remove_readonly(func, path, exc_info):
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def remove_tree(path):
    shutil.rmtree(path, onerror=handle_remove_readonly)


def copy_labels(source_root, output_root):
    for split in SPLITS:
        source_dir = source_root / "labels" / split
        output_dir = output_root / "labels" / split
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing label split: {source_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for label_path in source_dir.rglob("*.txt"):
            target = output_dir / label_path.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, target)


def build_dark_dataset(source_root, output_root, delta):
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source dataset: {source_root}")

    reset_dir(output_root)
    copy_labels(source_root, output_root)

    total = 0
    for split in SPLITS:
        source_dir = source_root / "images" / split
        output_dir = output_root / "images" / split
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing image split: {source_dir}")

        image_paths = [
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in DARKEN_IMAGE_EXTENSIONS
        ]
        if len(image_paths) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split} image count mismatch: {len(image_paths)} != {EXPECTED_COUNTS[split]}")

        for index, image_path in enumerate(image_paths, start=1):
            target = output_dir / image_path.relative_to(source_dir)
            darken_image_hsi(image_path, target, delta)
        total += len(image_paths)

    return total


def count_images(dataset_root, split):
    image_dir = dataset_root / "images" / split
    return sum(
        1
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DARKEN_IMAGE_EXTENSIONS
    )


def count_labels(dataset_root, split):
    return sum(1 for path in (dataset_root / "labels" / split).rglob("*.txt") if path.is_file())


def validate_dataset(dataset_root):
    for split, expected in EXPECTED_COUNTS.items():
        image_count = count_images(dataset_root, split)
        label_count = count_labels(dataset_root, split)
        if image_count != expected:
            raise ValueError(f"{dataset_root.name} {split} images: {image_count} != {expected}")
        if label_count != expected:
            raise ValueError(f"{dataset_root.name} {split} labels: {label_count} != {expected}")


def replace_dataset(tmp_root, final_root):
    if final_root.exists():
        remove_tree(final_root)
    tmp_root.rename(final_root)


def main():
    dark_tmp = DARK_DATASET.with_name(f"{DARK_DATASET.name}_tmp")
    gamma_tmp = GAMMA_DATASET.with_name(f"{GAMMA_DATASET.name}_tmp")

    total_dark = build_dark_dataset(SOURCE_DATASET, dark_tmp, delta=0.3)
    validate_dataset(dark_tmp)

    apply_gamma_to_dataset(dark_tmp, gamma_tmp, gamma_value=0.8)
    validate_dataset(gamma_tmp)

    replace_dataset(dark_tmp, DARK_DATASET)
    replace_dataset(gamma_tmp, GAMMA_DATASET)

    print(f"Done: {GAMMA_DATASET}")


if __name__ == "__main__":
    main()
