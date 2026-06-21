from __future__ import annotations

import warnings

import numpy as np
import pytest

from image_container.ch_order import ChannelOrder
from image_container.dct_image import BlockedDctImage, NonBlockedDctImage
from image_container.dct_image.utils.block_transform import BlockDctTransform
from image_container.dct_image.utils.dct_matrix import (
    BlockDctMatrices,
    _cached_block_dct_matrices,
)


@pytest.fixture
def bgr_image_odd() -> np.ndarray:
    """BGR uint8 image with odd height and width."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(33, 65, 3), dtype=np.uint8)


@pytest.fixture
def bgr_image_even() -> np.ndarray:
    """BGR uint8 image divisible by 8."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(32, 64, 3), dtype=np.uint8)


class TestNonBlockedDctImage:
    def test_from_array_pads_odd_dimensions(self, bgr_image_odd: np.ndarray) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            container = NonBlockedDctImage.from_array(bgr_image_odd, ChannelOrder.BGR)

        assert container.shape == (34, 66)
        assert container.source_shape == (33, 65)
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)

    def test_idct_crops_to_source_shape(self, bgr_image_odd: np.ndarray) -> None:
        container = NonBlockedDctImage.from_array(bgr_image_odd, ChannelOrder.BGR)
        restored = container.idct(is_uint8_cast_enabled=False)
        assert restored.shape == (33, 65)

    def test_high_pass_clears_global_dc(self, bgr_image_even: np.ndarray) -> None:
        container = NonBlockedDctImage.from_array(bgr_image_even, ChannelOrder.BGR)
        filtered = container.high_pass_filter(threshold=0.0)
        assert filtered.value[0, 0] == 0.0

    def test_low_pass_rejects_negative_threshold(
        self,
        bgr_image_even: np.ndarray,
    ) -> None:
        container = NonBlockedDctImage.from_array(bgr_image_even, ChannelOrder.BGR)
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            container.low_pass_filter(-1.0)


class TestBlockedDctImage:
    def test_from_array_pads_to_block_size(self, bgr_image_odd: np.ndarray) -> None:
        block_size = 8
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            container = BlockedDctImage.from_array(
                bgr_image_odd,
                ChannelOrder.BGR,
                block_size=block_size,
            )

        assert container.shape == (40, 72)
        assert container.source_shape == (33, 65)
        assert container.block_size == block_size
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)

    def test_idct_crops_to_source_shape(self, bgr_image_odd: np.ndarray) -> None:
        container = BlockedDctImage.from_array(
            bgr_image_odd,
            ChannelOrder.BGR,
            block_size=8,
        )
        restored = container.idct(is_uint8_cast_enabled=False)
        assert restored.shape == (33, 65)

    def test_high_pass_clears_per_block_dc(self, bgr_image_even: np.ndarray) -> None:
        block_size = 8
        container = BlockedDctImage.from_array(
            bgr_image_even,
            ChannelOrder.BGR,
            block_size=block_size,
        )
        filtered = container.high_pass_filter(threshold=0.0)
        dc_positions = filtered.value[0::block_size, 0::block_size]
        assert np.all(dc_positions == 0.0)

    def test_roundtrip_matches_grayscale(
        self,
        bgr_image_even: np.ndarray,
    ) -> None:
        container = BlockedDctImage.from_array(
            bgr_image_even,
            ChannelOrder.BGR,
            block_size=8,
        )
        restored = container.idct(is_uint8_cast_enabled=False)
        gray = ChannelOrder.GRAY.cv2_array_converter(ChannelOrder.BGR)(
            bgr_image_even
        ).astype(np.float32)
        max_error = float(np.max(np.abs(restored - gray)))
        assert max_error < 1e-3


class TestBlockDctTransform:
    def test_forward_inverse_roundtrip(self) -> None:
        rng = np.random.default_rng(2)
        block_size = 8
        gray = rng.random((32, 64), dtype=np.float32)
        transform = BlockDctTransform(block_size=block_size)
        coefficients = transform.forward(gray)
        restored = transform.inverse(coefficients)
        max_error = float(np.max(np.abs(restored - gray)))
        assert max_error < 1e-5

    def test_from_array_matches_transform(self, bgr_image_even: np.ndarray) -> None:
        block_size = 8
        container = BlockedDctImage.from_array(
            bgr_image_even,
            ChannelOrder.BGR,
            block_size=block_size,
        )
        gray = ChannelOrder.GRAY.cv2_array_converter(ChannelOrder.BGR)(
            bgr_image_even
        ).astype(np.float32)
        transform = BlockDctTransform(block_size=block_size)
        expected = transform.forward(gray)
        max_error = float(np.max(np.abs(container.value - expected)))
        assert max_error < 1e-5


class TestBlockDctMatricesCache:
    def test_from_block_size_returns_cached_instance(self) -> None:
        _cached_block_dct_matrices.cache_clear()
        first = BlockDctMatrices.from_block_size(8)
        second = BlockDctMatrices.from_block_size(8)
        assert first is second
