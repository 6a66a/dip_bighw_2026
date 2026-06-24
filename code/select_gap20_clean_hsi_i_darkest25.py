from __future__ import annotations

import csv
import math
import shutil
import stat
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"D:\Pythonfiles\dip\Project\Big_hw")
SOURCE_DATASET = ROOT / "Datasets" / "new_data_gap20_valtest_clean"
OUTPUT_DATASET = ROOT / "Datasets" / "new_data_gap20_valtest_clean_hsi_i_darkest25"
OUTPUT_YAML = ROOT / "Datasets" / "new_gap20_valtest_clean_hsi_i_darkest25.yaml"

REPORT_CSV = OUTPUT_DATASET / "gap20_clean_hsi_i_darkest25_report.csv"
SUMMARY_TXT = OUTPUT_DATASET / "gap20_clean_hsi_i_darkest25_summary.txt"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SUBSET_FRACTION = 0.25


def on_remove_error(func, path, _exc_info):
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=on_remove_error)
    path.mkdir(parents=True, exist_ok=True)


def iter_images(split: str) -> list[Path]:
    image_dir = SOURCE_DATASET / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image split directory: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def hsi_i_mean(image_path: Path) -> float:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return float(rgb.mean())


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        for image_path in iter_images(split):
            label_path = SOURCE_DATASET / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            rows.append(
                {
                    "split": split,
                    "image_name": image_path.name,
                    "label_name": label_path.name,
                    "source_image": image_path,
                    "source_label": label_path,
                    "hsi_i_mean": hsi_i_mean(image_path),
                    "selected": False,
                }
            )
    return rows


def copy_selected(selected_rows: list[dict[str, object]]) -> None:
    reset_dir(OUTPUT_DATASET)
    for split in SPLITS:
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in selected_rows:
        split = str(row["split"])
        shutil.copy2(
            Path(row["source_image"]),
            OUTPUT_DATASET / "images" / split / str(row["image_name"]),
        )
        shutil.copy2(
            Path(row["source_label"]),
            OUTPUT_DATASET / "labels" / split / str(row["label_name"]),
        )


def count_split(dataset: Path, split: str) -> tuple[int, int]:
    images = [p for p in (dataset / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = list((dataset / "labels" / split).glob("*.txt"))
    return len(images), len(labels)


def validate_dataset(dataset: Path) -> None:
    for split in SPLITS:
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        image_stems = {p.stem for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        label_stems = {p.stem for p in label_dir.glob("*.txt")}
        missing_labels = image_stems - label_stems
        extra_labels = label_stems - image_stems
        if missing_labels or extra_labels:
            raise RuntimeError(
                f"{dataset} {split}: missing_labels={len(missing_labels)}, extra_labels={len(extra_labels)}"
            )


def write_yaml() -> None:
    content = (
        f"path: {OUTPUT_DATASET}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: litter\n"
    )
    OUTPUT_YAML.write_text(content, encoding="utf-8")


def write_reports(rows: list[dict[str, object]], selected_rows: list[dict[str, object]]) -> None:
    sorted_rows = sorted(rows, key=lambda row: (float(row["hsi_i_mean"]), str(row["split"]), str(row["image_name"])))
    selected_keys = {(str(row["split"]), str(row["image_name"])) for row in selected_rows}
    for row in rows:
        row["selected"] = (str(row["split"]), str(row["image_name"])) in selected_keys

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank_by_hsi_i_mean",
                "split",
                "image_name",
                "label_name",
                "hsi_i_mean",
                "selected_darkest25",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow(
                {
                    "rank_by_hsi_i_mean": rank,
                    "split": row["split"],
                    "image_name": row["image_name"],
                    "label_name": row["label_name"],
                    "hsi_i_mean": f"{float(row['hsi_i_mean']):.8f}",
                    "selected_darkest25": row["selected"],
                }
            )

    threshold = max(float(row["hsi_i_mean"]) for row in selected_rows)
    counts = {split: count_split(OUTPUT_DATASET, split) for split in SPLITS}
    lines = [
        "Gap20 clean HSI-I darkest 25 percent subset summary",
        "",
        f"Source dataset: {SOURCE_DATASET}",
        f"Output dataset: {OUTPUT_DATASET}",
        f"Output YAML: {OUTPUT_YAML}",
        "",
        "Selection rule:",
        "  Compute HSI-I mean on each clean source image.",
        "  Sort all train/val/test images by HSI-I mean ascending.",
        f"  Select ceil({len(rows)} * {SUBSET_FRACTION}) = {len(selected_rows)} images.",
        f"  HSI-I mean threshold: {threshold:.8f}",
        "",
        "Output counts (images, labels):",
        *[f"  {split}: {counts[split][0]}, {counts[split][1]}" for split in SPLITS],
        "",
        f"CSV report: {REPORT_CSV}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {SOURCE_DATASET}")

    rows = collect_rows()
    selected_count = math.ceil(len(rows) * SUBSET_FRACTION)
    selected_rows = sorted(
        rows,
        key=lambda row: (float(row["hsi_i_mean"]), str(row["split"]), str(row["image_name"])),
    )[:selected_count]

    copy_selected(selected_rows)
    validate_dataset(OUTPUT_DATASET)
    write_yaml()
    write_reports(rows, selected_rows)


if __name__ == "__main__":
    main()
