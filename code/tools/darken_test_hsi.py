import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def rgb_to_hsi(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    intensity = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    nonzero_intensity = intensity > 1e-8
    saturation[nonzero_intensity] = 1.0 - min_rgb[nonzero_intensity] / intensity[nonzero_intensity]

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    cos_theta = np.zeros_like(intensity)
    valid = denominator > 1e-8
    cos_theta[valid] = numerator[valid] / denominator[valid]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)
    hue[~valid] = 0.0

    return hue, saturation, intensity


def hsi_to_rgb(hue, saturation, intensity):
    hue = np.mod(hue, 2.0 * np.pi)
    saturation = np.clip(saturation, 0.0, 1.0)
    intensity = np.clip(intensity, 0.0, 1.0)

    r = np.zeros_like(intensity)
    g = np.zeros_like(intensity)
    b = np.zeros_like(intensity)

    sector1 = hue < 2.0 * np.pi / 3.0
    sector2 = (hue >= 2.0 * np.pi / 3.0) & (hue < 4.0 * np.pi / 3.0)
    sector3 = ~(sector1 | sector2)

    h = hue[sector1]
    s = saturation[sector1]
    i = intensity[sector1]
    b[sector1] = i * (1.0 - s)
    r[sector1] = i * (1.0 + s * np.cos(h) / np.maximum(np.cos(np.pi / 3.0 - h), 1e-8))
    g[sector1] = 3.0 * i - (r[sector1] + b[sector1])

    h = hue[sector2] - 2.0 * np.pi / 3.0
    s = saturation[sector2]
    i = intensity[sector2]
    r[sector2] = i * (1.0 - s)
    g[sector2] = i * (1.0 + s * np.cos(h) / np.maximum(np.cos(np.pi / 3.0 - h), 1e-8))
    b[sector2] = 3.0 * i - (r[sector2] + g[sector2])

    h = hue[sector3] - 4.0 * np.pi / 3.0
    s = saturation[sector3]
    i = intensity[sector3]
    g[sector3] = i * (1.0 - s)
    b[sector3] = i * (1.0 + s * np.cos(h) / np.maximum(np.cos(np.pi / 3.0 - h), 1e-8))
    r[sector3] = 3.0 * i - (g[sector3] + b[sector3])

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def darken_image_hsi(input_path, output_path, delta):
    with Image.open(input_path) as image:
        rgb_image = image.convert("RGB")
        rgb = np.asarray(rgb_image)

    hue, saturation, intensity = rgb_to_hsi(rgb)
    dark_intensity = np.clip(intensity - delta, 0.0, 1.0)
    dark_rgb = hsi_to_rgb(hue, saturation, dark_intensity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(dark_rgb, mode="RGB").save(output_path, format="JPEG", quality=95)


def copy_tree(source, destination):
    if not source.exists():
        return
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def build_dataset(input_root, output_root, delta):
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset was not found: {input_root}")

    if output_root.exists():
        raise FileExistsError(f"Output dataset already exists: {output_root}")

    images_root = input_root / "images"
    labels_root = input_root / "labels"
    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError("Expected input dataset to contain images/ and labels/ folders.")

    copy_tree(labels_root, output_root / "labels")

    for split in ("train", "val"):
        copy_tree(images_root / split, output_root / "images" / split)

    test_input = images_root / "test"
    test_output = output_root / "images" / "test"
    image_paths = [path for path in test_input.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise FileNotFoundError(f"No test images were found under: {test_input}")

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(test_input)
        output_path = (test_output / relative_path).with_suffix(".jpg")
        darken_image_hsi(image_path, output_path, delta)
        if index % 25 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} test images")

    return len(image_paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Darken YOLO test images by subtracting from HSI intensity.")
    parser.add_argument(
        "--input",
        default=r"D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data",
        help="Input YOLO dataset root.",
    )
    parser.add_argument(
        "--output",
        default=r"D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_0.3",
        help="Output YOLO dataset root.",
    )
    parser.add_argument("--delta", type=float, default=0.3, help="Value subtracted from normalized HSI intensity.")
    return parser.parse_args()


def main():
    args = parse_args()
    count = build_dataset(Path(args.input), Path(args.output), args.delta)
    print(f"Done. Darkened {count} test images.")
    print(f"Output dataset: {Path(args.output)}")


if __name__ == "__main__":
    main()
