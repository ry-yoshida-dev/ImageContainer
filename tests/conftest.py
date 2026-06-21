from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def bgr_array(rng: np.random.Generator) -> np.ndarray:
    """BGR uint8 image (32x64x3)."""
    return rng.integers(0, 256, size=(32, 64, 3), dtype=np.uint8)


@pytest.fixture
def gray_array(rng: np.random.Generator) -> np.ndarray:
    """Grayscale uint8 image (32x64)."""
    return rng.integers(0, 256, size=(32, 64), dtype=np.uint8)


@pytest.fixture
def rgb_pil_image(rng: np.random.Generator) -> Image.Image:
    """RGB PIL image (64x32)."""
    array = rng.integers(0, 256, size=(32, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def gray_pil_image(rng: np.random.Generator) -> Image.Image:
    """Grayscale PIL image (64x32)."""
    array = rng.integers(0, 256, size=(32, 64), dtype=np.uint8)
    return Image.fromarray(array, mode="L")


@pytest.fixture
def temp_image_dir(
    tmp_path: Path,
    bgr_array: np.ndarray,
    gray_array: np.ndarray,
) -> Path:
    """Temporary directory containing sample PNG images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    bgr_rgb = bgr_array[..., ::-1]
    Image.fromarray(bgr_rgb, mode="RGB").save(image_dir / "color.png")
    Image.fromarray(gray_array, mode="L").save(image_dir / "gray.png")

    nested = image_dir / "nested"
    nested.mkdir()
    Image.fromarray(bgr_rgb, mode="RGB").save(nested / "nested_color.jpg")

    (image_dir / "not_an_image.txt").write_text("skip me")

    return image_dir
