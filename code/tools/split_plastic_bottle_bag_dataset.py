from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import os


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def group_name(path: Path) -> str:
    match = re.match(r"([A-Za-z_]+)", path.stem)
    if not match:
        return path.stem.lower()
    return match.group(1).lower().rstrip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("Datasets") / "Plastic_Bottl_and_Bag")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    image_train = dataset / "images" / "train"
    image_val = dataset / "images" / "val"
    label_train = dataset / "labels" / "train"
    label_val = dataset / "labels" / "val"

    groups: dict[str, list[Path]] = defaultdict(list)
    for image in image_train.iterdir():
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            groups[group_name(image)].append(image)

    rng = random.Random(args.seed)
    selected: list[tuple[str, Path]] = []
    before = {name: len(images) for name, images in groups.items()}

    for name, images in sorted(groups.items()):
        images = sorted(images, key=lambda p: p.name.lower())
        take = round(len(images) * args.val_ratio)
        take = max(1, take) if images else 0
        sampled = rng.sample(images, take)
        selected.extend((name, image) for image in sampled)

    if args.apply:
        image_val.mkdir(parents=True, exist_ok=True)
        label_val.mkdir(parents=True, exist_ok=True)
        for _name, image in selected:
            label = label_train / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label for {image}: {label}")

            target_image = image_val / image.name
            target_label = label_val / label.name
            if target_label.exists():
                raise FileExistsError(f"Target label already exists: {target_label}")
            if target_image.exists():
                if target_image.stat().st_size != image.stat().st_size:
                    raise FileExistsError(f"Different target image already exists: {target_image}")
                os.remove(image)
            else:
                shutil.move(str(image), str(target_image))

            shutil.move(str(label), str(target_label))

    moved = Counter(name for name, _image in selected)
    print(f"Done: {dataset}")


if __name__ == "__main__":
    main()
