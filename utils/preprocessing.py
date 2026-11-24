from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_rgb_image(path_or_file) -> Image.Image:
    image = Image.open(path_or_file)
    image.load()
    return image.convert("RGB")


def resize_for_model(image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.Resampling.BILINEAR)


def extract_hog_features(image: Image.Image, size: Tuple[int, int] = (128, 128),
                         cell_size: int = 8, block_size: int = 2,
                         num_bins: int = 9) -> np.ndarray:
    gray = np.asarray(resize_for_model(image, size).convert("L"), dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    height, width = gray.shape
    cells_y, cells_x = height // cell_size, width // cell_size
    histogram = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)
    bin_width = 180.0 / num_bins

    for cy in range(cells_y):
        for cx in range(cells_x):
            y0, x0 = cy * cell_size, cx * cell_size
            for y in range(y0, y0 + cell_size):
                for x in range(x0, x0 + cell_size):
                    position = angle[y, x] / bin_width
                    left = int(np.floor(position)) % num_bins
                    right = (left + 1) % num_bins
                    weight = position - np.floor(position)
                    value = magnitude[y, x]
                    histogram[cy, cx, left] += value * (1.0 - weight)
                    histogram[cy, cx, right] += value * weight

    descriptors = []
    for cy in range(cells_y - block_size + 1):
        for cx in range(cells_x - block_size + 1):
            block = histogram[cy:cy + block_size, cx:cx + block_size].ravel()
            norm = np.sqrt(np.dot(block, block) + 1e-6)
            descriptors.append(block / norm)
    return np.ascontiguousarray(np.concatenate(descriptors), dtype=np.float32)
