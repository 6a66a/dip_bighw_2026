from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_counts(label_path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not label_path.exists():
        return counts

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            counts[int(parts[0])] += 1
        except ValueError:
            continue
    return counts


def filename_group(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def choose_subset(items: list[tuple[Path, Path, Counter[int]]], ratio: float, seed: int) -> list[tuple[Path, Path, Counter[int]]]:
    rng = random.Random(seed)
    groups: dict[str, list[tuple[Path, Path, Counter[int]]]] = defaultdict(list)
    for item in items:
        groups[filename_group(item[0])].append(item)

    selected: list[tuple[Path, Path, Counter[int]]] = []
    for group_name in sorted(groups):
        group_items = groups[group_name][:]
        rng.shuffle(group_items)
        target_size = max(1, int(round(len(group_items) * ratio))) if group_items else 0
        selected.extend(group_items[:target_size])

    return sorted(selected, key=lambda item: item[0].name)


def copy_split(src_split: Path, dst_split: Path, ratio: float, seed: int) -> dict[str, object]:
    dst_split.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path
        for path in src_split.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and (src_split / f"{path.stem}.txt").exists()
    )
    items = [(image, src_split / f"{image.stem}.txt", read_yolo_counts(src_split / f"{image.stem}.txt")) for image in images]
    selected = choose_subset(items, ratio, seed)

    list_lines: list[str] = []
    selected_counts: Counter[int] = Counter()
    for image, label, counts in selected:
        shutil.copy2(image, dst_split / image.name)
        shutil.copy2(label, dst_split / label.name)
        selected_counts.update(counts)
        list_lines.append(f"{dst_split.name}/{image.name}")

    return {
        "images": len(images),
        "selected": len(selected),
        "groups": dict(Counter(filename_group(image) for image, _, _ in items)),
        "selected_groups": dict(Counter(filename_group(image) for image, _, _ in selected)),
        "objects": dict(sum((counts for _, _, counts in items), Counter())),
        "selected_objects": dict(selected_counts),
        "list_lines": list_lines,
    }


def write_data_files(src_root: Path, dst_root: Path, split_lists: dict[str, list[str]]) -> None:
    names_src = src_root.parent / "cfgs_and_weights" / "yolo" / "cfg" / "yolo.names"
    if names_src.exists():
        shutil.copy2(names_src, dst_root / "yolo.names")

    for split, lines in split_lists.items():
        (dst_root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    names_count = 0
    names_file = dst_root / "yolo.names"
    if names_file.exists():
        names_count = len([line for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()])

    data_text = "\n".join(
        [
            f"classes = {names_count}",
            "train = train.txt",
            "valid = val.txt",
            "test = test.txt",
            "names = yolo.names",
            "",
        ]
    )
    (dst_root / "yolo.data").write_text(data_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified YOLO-only subset of trash_ICRA19.")
    parser.add_argument("--src", type=Path, required=True, help="Path to original dataset directory containing train/val/test.")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory for the YOLO-only subset.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Subset ratio.")
    parser.add_argument("--seed", type=int, default=20260527, help="Random seed.")
    args = parser.parse_args()

    if args.dst.exists():
        raise SystemExit(f"Output directory already exists: {args.dst}")
    if not 0 < args.ratio <= 1:
        raise SystemExit("--ratio must be in (0, 1].")

    args.dst.mkdir(parents=True)
    split_lists: dict[str, list[str]] = {}
    report_lines = [
        f"source: {args.src}",
        f"output: {args.dst}",
        f"ratio: {args.ratio}",
        f"seed: {args.seed}",
        "",
    ]

    for split_index, split in enumerate(("train", "val", "test")):
        stats = copy_split(args.src / split, args.dst / split, args.ratio, args.seed + split_index)
        split_lists[split] = stats["list_lines"]
        report_lines.extend(
            [
                f"[{split}] images: {stats['images']} -> {stats['selected']}",
                f"[{split}] groups: {stats['groups']}",
                f"[{split}] selected_groups: {stats['selected_groups']}",
                f"[{split}] objects: {stats['objects']}",
                f"[{split}] selected_objects: {stats['selected_objects']}",
                "",
            ]
        )

    write_data_files(args.src, args.dst, split_lists)
    (args.dst / "subset_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
