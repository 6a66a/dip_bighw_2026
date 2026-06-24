from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path


VAL_GROUPS = {
    "vid_000105",
    "vid_000108",
    "vid_000118",
    "vid_000139",
    "vid_000142",
    "vid_000144",
    "vid_000148",
    "vid_000160",
    "vid_000161",
    "vid_000172",
    "vid_000273",
    "vid_000277",
    "vid_000316",
    "vid_000319",
    "vid_000386",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def video_group(stem: str) -> str:
    if stem.startswith("vid_") and "_frame" in stem:
        return stem.split("_frame", 1)[0]
    return stem


def read_label_classes(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(float(line.split()[0]))] += 1
    return counts


def collect_images(dataset: Path) -> list[Path]:
    images = []
    for split in ("train", "val"):
        split_dir = dataset / "images" / split
        images.extend(p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(images)


def move_pair(image_path: Path, dataset: Path, target_split: str, apply: bool) -> None:
    current_split = image_path.parent.name
    stem = image_path.stem
    label_path = dataset / "labels" / current_split / f"{stem}.txt"

    target_image = dataset / "images" / target_split / image_path.name
    target_label = dataset / "labels" / target_split / f"{stem}.txt"

    if current_split == target_split:
        return
    if target_image.exists() or target_label.exists():
        raise FileExistsError(f"Target already exists for {stem}: {target_image} or {target_label}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

    if apply:
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(target_image))
        shutil.move(str(label_path), str(target_label))


def main() -> None:
    parser = argparse.ArgumentParser(description="Resplit YOLO train/val by whole video groups.")
    parser.add_argument("--dataset", type=Path, default=Path("Datasets"))
    parser.add_argument("--apply", action="store_true", help="Actually move files. Omit for a dry run.")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    images = collect_images(dataset)

    plan = {}
    stats = defaultdict(lambda: {"images": 0, "groups": set(), "classes": Counter()})
    movements = []

    for image_path in images:
        group = video_group(image_path.stem)
        target_split = "val" if group in VAL_GROUPS else "train"
        plan[image_path] = target_split

        label_path = dataset / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        stats[target_split]["images"] += 1
        stats[target_split]["groups"].add(group)
        stats[target_split]["classes"].update(read_label_classes(label_path))

        if image_path.parent.name != target_split:
            movements.append((image_path, target_split))

    for image_path, target_split in movements:
        move_pair(image_path, dataset, target_split, apply=args.apply)

    report_lines = [
        "Group-based train/val split report",
        f"Dataset: {dataset}",
        f"Applied: {args.apply}",
        "",
    ]
    for split in ("train", "val"):
        report_lines.append(
            f"{split}: {stats[split]['images']} images, "
            f"{len(stats[split]['groups'])} groups, classes={dict(sorted(stats[split]['classes'].items()))}"
        )
    report_lines.extend(
        [
            "",
            f"Moved pairs: {len(movements)}",
            "",
            "Validation groups:",
            *sorted(VAL_GROUPS),
            "",
        ]
    )

    report_path = dataset / "split_by_video_group_report.txt"
    if args.apply:
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Done: {report_path if args.apply else dataset}")


if __name__ == "__main__":
    main()
