from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def label_for_image(dataset: Path, split: str, image: Path) -> Path:
    return dataset / "labels" / split / f"{image.stem}.txt"


def copy_pair(image: Path, label: Path, out_root: Path, category: str, split: str) -> None:
    image_dir = out_root / category / "images" / split
    label_dir = out_root / category / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dir / image.name)
    if label.exists():
        shutil.copy2(label, label_dir / label.name)


def measure_image(image_path: Path) -> dict[str, float] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    std = float(gray.std())
    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dark_ratio = float((gray < 45).mean())
    bright_ratio = float((gray > 235).mean())
    low_sat_bright_ratio = float(((gray > 220) & (saturation < 45)).mean())

    return {
        "mean": mean,
        "std": std,
        "p05": p05,
        "p95": p95,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "low_sat_bright_ratio": low_sat_bright_ratio,
        "lap_var": lap_var,
    }


def classify(metrics: dict[str, float]) -> list[str]:
    categories = []

    # Dim or strongly underexposed images.
    if metrics["mean"] <= 55 or metrics["dark_ratio"] >= 0.40:
        categories.append("dark_or_underexposed")

    # Bright clipped regions, useful for exposure correction examples.
    if metrics["bright_ratio"] >= 0.12 or metrics["low_sat_bright_ratio"] >= 0.08:
        categories.append("overexposed")

    # Low Laplacian variance catches heavy blur and many ghosting/motion-smear cases.
    if metrics["lap_var"] <= 700:
        categories.append("blur_or_ghosting")

    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract images that suit digital image processing experiments.")
    parser.add_argument("--dataset", type=Path, default=Path("Datasets"))
    parser.add_argument("--output", type=Path, default=Path("Datasets") / "image_processing_issues")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    clear_dir(output)

    manifest_rows = []
    counts: dict[str, int] = {
        "dark_or_underexposed": 0,
        "overexposed": 0,
        "blur_or_ghosting": 0,
        "all_issues": 0,
    }
    seen_all: set[str] = set()

    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            metrics = measure_image(image)
            if metrics is None:
                continue

            categories = classify(metrics)
            if not categories:
                continue

            label = label_for_image(dataset, split, image)
            for category in categories:
                copy_pair(image, label, output, category, split)
                counts[category] += 1

            unique_key = f"{split}/{image.name}"
            if unique_key not in seen_all:
                copy_pair(image, label, output, "all_issues", split)
                counts["all_issues"] += 1
                seen_all.add(unique_key)

            row = {
                "split": split,
                "image": image.name,
                "label": label.name if label.exists() else "",
                "categories": ";".join(categories),
            }
            row.update({key: round(value, 6) for key, value in metrics.items()})
            manifest_rows.append(row)

    manifest = output / "manifest.csv"
    fieldnames = [
        "split",
        "image",
        "label",
        "categories",
        "mean",
        "std",
        "p05",
        "p95",
        "dark_ratio",
        "bright_ratio",
        "low_sat_bright_ratio",
        "lap_var",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Output: {output}")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
