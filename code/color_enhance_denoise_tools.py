from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np


ColorOrder = Literal["RGB", "BGR"]
DenoiseMethod = Literal["gaussian", "median", "bilateral", "nl_means"]


def _as_uint8(image: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(image)
    was_float01 = np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0

    if was_float01:
        arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint8:
        arr_uint8 = arr.copy()
    else:
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)

    return arr_uint8, was_float01


def _restore_dtype(image_uint8: np.ndarray, as_float01: bool) -> np.ndarray:
    if as_float01:
        return image_uint8.astype(np.float32) / 255.0
    return image_uint8


def read_image(path: str | Path, color_order: ColorOrder = "RGB") -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if color_order == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(path: str | Path, image: np.ndarray, color_order: ColorOrder = "RGB") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8, _ = _as_uint8(image)
    if color_order == "RGB":
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix, image_uint8)
    if not ok:
        raise ValueError(f"无法按 {path.suffix} 格式编码图像: {path}")
    encoded.tofile(str(path))


def gamma_transform_color(image: np.ndarray, gamma: float, gain: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma 必须为正数")

    image_uint8, was_float01 = _as_uint8(image)
    normalized = image_uint8.astype(np.float32) / 255.0
    transformed = np.clip(gain * np.power(normalized, gamma), 0.0, 1.0)
    transformed_uint8 = (transformed * 255.0).round().astype(np.uint8)
    return _restore_dtype(transformed_uint8, was_float01)


def rgb_to_hsi(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_uint8, _ = _as_uint8(rgb)
    rgb_norm = rgb_uint8.astype(np.float32) / 255.0
    r = rgb_norm[:, :, 0]
    g = rgb_norm[:, :, 1]
    b = rgb_norm[:, :, 2]

    intensity = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1.0 - 3.0 * min_rgb / (r + g + b + 1e-6)
    saturation[intensity == 0] = 0

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))

    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)
    hue = hue / (2.0 * np.pi)
    hue[saturation == 0] = 0
    return hue, saturation, intensity


def hsi_to_rgb(hue: np.ndarray, saturation: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    h = np.clip(hue, 0.0, 1.0) * 2.0 * np.pi
    s = np.clip(saturation, 0.0, 1.0)
    i = np.clip(intensity, 0.0, 1.0)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    sector_0 = (h >= 0) & (h < 2.0 * np.pi / 3.0)
    sector_1 = (h >= 2.0 * np.pi / 3.0) & (h < 4.0 * np.pi / 3.0)
    sector_2 = (h >= 4.0 * np.pi / 3.0) & (h < 2.0 * np.pi)

    h0 = h[sector_0]
    b[sector_0] = i[sector_0] * (1.0 - s[sector_0])
    r[sector_0] = i[sector_0] * (
        1.0 + s[sector_0] * np.cos(h0) / (np.cos(np.pi / 3.0 - h0) + 1e-6)
    )
    g[sector_0] = 3.0 * i[sector_0] - (r[sector_0] + b[sector_0])

    h1 = h[sector_1] - 2.0 * np.pi / 3.0
    r[sector_1] = i[sector_1] * (1.0 - s[sector_1])
    g[sector_1] = i[sector_1] * (
        1.0 + s[sector_1] * np.cos(h1) / (np.cos(np.pi / 3.0 - h1) + 1e-6)
    )
    b[sector_1] = 3.0 * i[sector_1] - (r[sector_1] + g[sector_1])

    h2 = h[sector_2] - 4.0 * np.pi / 3.0
    g[sector_2] = i[sector_2] * (1.0 - s[sector_2])
    b[sector_2] = i[sector_2] * (
        1.0 + s[sector_2] * np.cos(h2) / (np.cos(np.pi / 3.0 - h2) + 1e-6)
    )
    r[sector_2] = 3.0 * i[sector_2] - (g[sector_2] + b[sector_2])

    rgb = np.stack([r, g, b], axis=2)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def equalize_hsi_intensity(
    image: np.ndarray,
    color_order: ColorOrder = "RGB",
    use_clahe: bool = False,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    image_uint8, was_float01 = _as_uint8(image)
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("equalize_hsi_intensity 需要输入三通道彩色图像")

    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB) if color_order == "BGR" else image_uint8
    hue, saturation, intensity = rgb_to_hsi(rgb)
    intensity_uint8 = (np.clip(intensity, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized_i = clahe.apply(intensity_uint8)
    else:
        equalized_i = cv2.equalizeHist(intensity_uint8)

    equalized_rgb = hsi_to_rgb(hue, saturation, equalized_i.astype(np.float32) / 255.0)
    output = cv2.cvtColor(equalized_rgb, cv2.COLOR_RGB2BGR) if color_order == "BGR" else equalized_rgb
    return _restore_dtype(output, was_float01)


def equalize_yuv_luminance(image: np.ndarray, color_order: ColorOrder = "RGB") -> np.ndarray:
    image_uint8, was_float01 = _as_uint8(image)
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("equalize_yuv_luminance 需要输入三通道彩色图像")

    bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR) if color_order == "RGB" else image_uint8
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    equalized_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    output = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB) if color_order == "RGB" else equalized_bgr
    return _restore_dtype(output, was_float01)


def clahe_yuv_luminance(
    image: np.ndarray,
    color_order: ColorOrder = "RGB",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    if clip_limit <= 0:
        raise ValueError("clip_limit 必须为正数")
    image_uint8, was_float01 = _as_uint8(image)
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("clahe_yuv_luminance 需要输入三通道彩色图像")

    bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR) if color_order == "RGB" else image_uint8
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    clahe_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    output = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB) if color_order == "RGB" else clahe_bgr
    return _restore_dtype(output, was_float01)


def gaussian_denoise(image: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须为正奇数")
    image_uint8, was_float01 = _as_uint8(image)
    output = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), sigmaX=sigma)
    return _restore_dtype(output, was_float01)


def median_denoise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须为正奇数")
    image_uint8, was_float01 = _as_uint8(image)
    output = cv2.medianBlur(image_uint8, kernel_size)
    return _restore_dtype(output, was_float01)


def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    if diameter <= 0:
        raise ValueError("diameter 必须为正数")
    image_uint8, was_float01 = _as_uint8(image)
    output = cv2.bilateralFilter(image_uint8, diameter, sigma_color, sigma_space)
    return _restore_dtype(output, was_float01)


def nl_means_denoise(
    image: np.ndarray,
    h: float = 10.0,
    h_color: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
    color_order: ColorOrder = "RGB",
) -> np.ndarray:
    image_uint8, was_float01 = _as_uint8(image)

    if image_uint8.ndim == 2:
        output = cv2.fastNlMeansDenoising(
            image_uint8,
            None,
            h,
            template_window_size,
            search_window_size,
        )
    else:
        bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR) if color_order == "RGB" else image_uint8
        output = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h,
            h_color,
            template_window_size,
            search_window_size,
        )
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) if color_order == "RGB" else output

    return _restore_dtype(output, was_float01)


def denoise(
    image: np.ndarray,
    method: DenoiseMethod = "gaussian",
    **kwargs,
) -> np.ndarray:
    functions = {
        "gaussian": gaussian_denoise,
        "median": median_denoise,
        "bilateral": bilateral_denoise,
        "nl_means": nl_means_denoise,
    }
    if method not in functions:
        raise ValueError(f"未知降噪方法: {method}")
    return functions[method](image, **kwargs)
