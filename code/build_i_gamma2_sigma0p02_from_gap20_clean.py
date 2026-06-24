from __future__ import annotations

import csv
import hashlib
import math
import shutil
import stat
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"D:\Pythonfiles\dip\Project\Big_hw")
SOURCE_DATASET = ROOT / "Datasets" / "new_data_gap20_valtest_clean"

FULL_DATASET = ROOT / "Datasets" / "new_data_gap20_valtest_clean_i_gamma2_sigma0p02"
SUBSET_DATASET = ROOT / "Datasets" / "new_data_gap20_valtest_clean_i_gamma2_sigma0p02_source_darkest25"

FULL_YAML = ROOT / "Datasets" / "new_gap20_valtest_clean_i_gamma2_sigma0p02.yaml"
SUBSET_YAML = ROOT / "Datasets" / "new_gap20_valtest_clean_i_gamma2_sigma0p02_source_darkest25.yaml"

REPORT_CSV = SUBSET_DATASET / "i_gamma2_sigma0p02_source_darkest25_report.csv"
REPORT_TXT = SUBSET_DATASET / "i_gamma2_sigma0p02_source_darkest25_summary.txt"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

GAMMA = 2.0
SIGMA = 0.02
SEED = 20260614
SUBSET_FRACTION = 0.25


def rgb_to_hsi(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_norm = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    intensity = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1.0 - 3.0 * min_rgb / (r + g + b + 1e-6)
    saturation[intensity == 0] = 0

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))

    hue = np.where(b <= g, theta, 2 * np.pi - theta)
    hue = hue / (2 * np.pi)
    hue[saturation == 0] = 0

    return hue, saturation, intensity


def hsi_to_rgb(hue: np.ndarray, saturation: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    h = hue * 2 * np.pi
    s = saturation
    i = intensity

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    sector_0 = (h >= 0) & (h < 2 * np.pi / 3)
    sector_1 = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    sector_2 = (h >= 4 * np.pi / 3) & (h < 2 * np.pi)

    h0 = h[sector_0]
    b[sector_0] = i[sector_0] * (1 - s[sector_0])
    r[sector_0] = i[sector_0] * (
        1 + s[sector_0] * np.cos(h0) / (np.cos(np.pi / 3 - h0) + 1e-6)
    )
    g[sector_0] = 3 * i[sector_0] - (r[sector_0] + b[sector_0])

    h1 = h[sector_1] - 2 * np.pi / 3
    r[sector_1] = i[sector_1] * (1 - s[sector_1])
    g[sector_1] = i[sector_1] * (
        1 + s[sector_1] * np.cos(h1) / (np.cos(np.pi / 3 - h1) + 1e-6)
    )
    b[sector_1] = 3 * i[sector_1] - (r[sector_1] + g[sector_1])

    h2 = h[sector_2] - 4 * np.pi / 3
    g[sector_2] = i[sector_2] * (1 - s[sector_2])
    b[sector_2] = i[sector_2] * (
        1 + s[sector_2] * np.cos(h2) / (np.cos(np.pi / 3 - h2) + 1e-6)
    )
    r[sector_2] = 3 * i[sector_2] - (g[sector_2] + b[sector_2])

    rgb = np.stack([r, g, b], axis=2)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def on_remove_error(func, path, _exc_info):
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=on_remove_error)
    path.mkdir(parents=True, exist_ok=True)


def iter_images(split: str) -> list[Path]:
    image_dir = SOURCE_DATASET / "images" / split
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, quality=95)


def deterministic_rng(split: str, relative_image: str) -> np.random.Generator:
    key = f"{SEED}/{split}/{relative_image}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def source_intensity_mean(path: Path) -> float:
    _, _, intensity = rgb_to_hsi(read_rgb(path))
    return float(np.mean(intensity))


def transform_image(path: Path, split: str, relative_image: str) -> tuple[np.ndarray, float]:
    hue, saturation, intensity = rgb_to_hsi(read_rgb(path))
    rng = deterministic_rng(split, relative_image)
    noise = rng.normal(loc=0.0, scale=SIGMA, size=intensity.shape).astype(np.float32)
    transformed_i = np.clip(np.power(intensity, GAMMA) + noise, 0.0, 1.0)
    rgb = hsi_to_rgb(hue, saturation, transformed_i)
    return rgb, float(np.mean(transformed_i))


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        for image_path in iter_images(split):
            label_path = SOURCE_DATASET / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            relative_image = image_path.name
            rows.append(
                {
                    "split": split,
                    "image_name": image_path.name,
                    "label_name": label_path.name,
                    "source_image": image_path,
                    "source_label": label_path,
                    "source_i_mean": source_intensity_mean(image_path),
                    "transformed_i_mean": None,
                    "selected_darkest25_from_source": False,
                }
            )
    return rows


def copy_label(source_label: Path, target_label: Path) -> None:
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_label, target_label)


def build_full_dataset(rows: list[dict[str, object]]) -> None:
    reset_dir(FULL_DATASET)
    for split in SPLITS:
        (FULL_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (FULL_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        split = str(row["split"])
        image_name = str(row["image_name"])
        label_name = str(row["label_name"])
        rgb, transformed_mean = transform_image(Path(row["source_image"]), split, image_name)
        row["transformed_i_mean"] = transformed_mean
        write_rgb(FULL_DATASET / "images" / split / image_name, rgb)
        copy_label(Path(row["source_label"]), FULL_DATASET / "labels" / split / label_name)


def build_subset_dataset(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_count = math.ceil(len(rows) * SUBSET_FRACTION)
    selected_rows = sorted(rows, key=lambda r: (float(r["source_i_mean"]), str(r["split"]), str(r["image_name"])))[:selected_count]
    selected_keys = {(str(r["split"]), str(r["image_name"])) for r in selected_rows}

    reset_dir(SUBSET_DATASET)
    for split in SPLITS:
        (SUBSET_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (SUBSET_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        key = (str(row["split"]), str(row["image_name"]))
        row["selected_darkest25_from_source"] = key in selected_keys

    for row in selected_rows:
        split = str(row["split"])
        image_name = str(row["image_name"])
        label_name = str(row["label_name"])
        rgb, transformed_mean = transform_image(Path(row["source_image"]), split, image_name)
        row["subset_transformed_i_mean"] = transformed_mean
        write_rgb(SUBSET_DATASET / "images" / split / image_name, rgb)
        copy_label(Path(row["source_label"]), SUBSET_DATASET / "labels" / split / label_name)

    return selected_rows


def write_yaml(path: Path, dataset_path: Path) -> None:
    content = (
        f"path: {dataset_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: litter\n"
    )
    path.write_text(content, encoding="utf-8")


def count_split(dataset: Path, split: str) -> tuple[int, int]:
    images = [p for p in (dataset / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = list((dataset / "labels" / split).glob("*.txt"))
    return len(images), len(labels)


def write_reports(rows: list[dict[str, object]], selected_rows: list[dict[str, object]]) -> None:
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (float(r["source_i_mean"]), str(r["split"]), str(r["image_name"])))
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank_by_source_i_mean",
                "split",
                "image_name",
                "label_name",
                "source_i_mean",
                "full_transformed_i_mean",
                "selected_darkest25_from_source",
                "subset_transformed_i_mean",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow(
                {
                    "rank_by_source_i_mean": rank,
                    "split": row["split"],
                    "image_name": row["image_name"],
                    "label_name": row["label_name"],
                    "source_i_mean": f"{float(row['source_i_mean']):.8f}",
                    "full_transformed_i_mean": f"{float(row['transformed_i_mean']):.8f}",
                    "selected_darkest25_from_source": row["selected_darkest25_from_source"],
                    "subset_transformed_i_mean": (
                        f"{float(row['subset_transformed_i_mean']):.8f}"
                        if "subset_transformed_i_mean" in row
                        else ""
                    ),
                }
            )

    full_counts = {split: count_split(FULL_DATASET, split) for split in SPLITS}
    subset_counts = {split: count_split(SUBSET_DATASET, split) for split in SPLITS}
    threshold = max(float(r["source_i_mean"]) for r in selected_rows)
    lines = [
        "I-gamma plus Gaussian-noise dataset build summary",
        "",
        f"Source dataset: {SOURCE_DATASET}",
        f"Full transformed dataset: {FULL_DATASET}",
        f"Source-darkest 25 percent transformed subset: {SUBSET_DATASET}",
        f"Gamma: {GAMMA}",
        f"Gaussian noise sigma on normalized I: {SIGMA}",
        f"Noise seed: {SEED}",
        "",
        "Selection rule:",
        "  Sort all source clean images by original HSI I mean in ascending order.",
        f"  Select ceil({len(rows)} * {SUBSET_FRACTION}) = {len(selected_rows)} images.",
        f"  Source I mean threshold: {threshold:.8f}",
        "",
        "Full transformed counts (images, labels):",
        *[f"  {split}: {full_counts[split][0]}, {full_counts[split][1]}" for split in SPLITS],
        "",
        "Subset counts (images, labels):",
        *[f"  {split}: {subset_counts[split][0]}, {subset_counts[split][1]}" for split in SPLITS],
        "",
        f"CSV report: {REPORT_CSV}",
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_dataset(dataset: Path) -> None:
    for split in SPLITS:
        image_count, label_count = count_split(dataset, split)
        if image_count != label_count:
            raise RuntimeError(f"{dataset} {split}: image_count={image_count}, label_count={label_count}")


def main() -> None:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {SOURCE_DATASET}")

    rows = collect_rows()
    build_full_dataset(rows)
    selected_rows = build_subset_dataset(rows)

    write_yaml(FULL_YAML, FULL_DATASET)
    write_yaml(SUBSET_YAML, SUBSET_DATASET)

    validate_dataset(FULL_DATASET)
    validate_dataset(SUBSET_DATASET)
    write_reports(rows, selected_rows)


if __name__ == "__main__":
    main()
