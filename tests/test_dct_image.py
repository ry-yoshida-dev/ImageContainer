from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest

from image_container.ch_order import ChannelOrder
from image_container.dct_image import (
    BlockedDctImage,
    NonBlockedDctImage,
    ZigzagDctCoefficients,
)
from image_container.dct_image.utils.block_transform import BlockDctTransform
from image_container.dct_image.utils.zigzag_scan import ZigzagScan
from image_container.types import ImageArray

JPEG_ZIGZAG_PREFIX_8X8 = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5]


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
            _ = container.low_pass_filter(-1.0)

    def test_dc_and_ac_components_partition_coefficients(
        self,
        bgr_image_even: np.ndarray,
    ) -> None:
        container = NonBlockedDctImage.from_array(bgr_image_even, ChannelOrder.BGR)
        dc = container.dc_component
        ac = container.ac_components

        assert dc.shape == container.shape
        assert ac.shape == container.shape
        assert dc.dtype == np.float32
        assert ac.dtype == np.float32
        assert dc[0, 0] == container.value[0, 0]
        assert ac[0, 0] == 0.0
        assert np.all(np.equal(dc[1:, :], 0.0))
        assert np.all(np.equal(dc[0, 1:], 0.0))
        np.testing.assert_allclose(dc + ac, container.value)


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
        assert np.all(np.equal(dc_positions, 0.0))

    def test_dc_and_ac_components_partition_per_block_coefficients(
        self,
        bgr_image_even: np.ndarray,
    ) -> None:
        block_size = 8
        container = BlockedDctImage.from_array(
            bgr_image_even,
            ChannelOrder.BGR,
            block_size=block_size,
        )
        dc = container.dc_component
        ac = container.ac_components
        dc_positions = dc[0::block_size, 0::block_size]

        assert dc.shape == container.shape
        assert ac.shape == container.shape
        assert np.array_equal(
            dc_positions,
            container.value[0::block_size, 0::block_size],
        )
        assert np.all(np.equal(ac[0::block_size, 0::block_size], 0.0))
        np.testing.assert_allclose(dc + ac, container.value)

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
        error: ImageArray = np.abs(restored - gray)
        max_error = float(np.max(error))
        assert max_error < 1e-3


class TestBlockDctTransform:
    def test_forward_inverse_roundtrip(self) -> None:
        rng = np.random.default_rng(2)
        block_size = 8
        gray = rng.random((32, 64), dtype=np.float32)
        transform = BlockDctTransform(block_size=block_size)
        coefficients = transform.forward(gray)
        restored = transform.inverse(coefficients)
        error: ImageArray = np.abs(restored - gray)
        max_error = float(np.max(error))
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
        error: ImageArray = np.abs(container.value - expected)
        max_error = float(np.max(error))
        assert max_error < 1e-5

    def test_matches_per_block_opencv(self) -> None:
        rng = np.random.default_rng(3)
        block_size = 8
        height, width = 24, 40
        gray = rng.uniform(0.0, 255.0, (height, width)).astype(np.float32)
        transform = BlockDctTransform(block_size=block_size)
        coefficients = transform.forward(gray)
        expected = np.zeros_like(gray)
        for top in range(0, height, block_size):
            for left in range(0, width, block_size):
                block = gray[top : top + block_size, left : left + block_size]
                expected[top : top + block_size, left : left + block_size] = cv2.dct(
                    block
                )
        np.testing.assert_allclose(coefficients, expected, atol=1e-3)

    @pytest.mark.parametrize("block_size", [0, -2, 3, 7])
    def test_rejects_non_positive_or_odd_block_size(self, block_size: int) -> None:
        with pytest.raises(ValueError, match="positive even number"):
            _ = BlockDctTransform(block_size=block_size)


class TestZigzagScan:
    def test_raster_indices_follow_jpeg_order(self) -> None:
        indices = ZigzagScan(block_size=8).raster_indices
        assert indices[: len(JPEG_ZIGZAG_PREFIX_8X8)].tolist() == JPEG_ZIGZAG_PREFIX_8X8
        assert indices[-1] == 63

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
    def test_raster_indices_are_permutation(self, block_size: int) -> None:
        indices = ZigzagScan(block_size=block_size).raster_indices
        assert np.array_equal(np.sort(indices), np.arange(block_size * block_size))

    def test_scatter_zero_fills_missing_slots(self) -> None:
        scan = ZigzagScan(block_size=4)
        coefficients = np.arange(1, 4, dtype=np.float32).reshape(1, 1, 3)
        layout = scan.scatter(coefficients)
        expected = np.zeros((4, 4), dtype=np.float32)
        expected[0, 0] = 1.0
        expected[0, 1] = 2.0
        expected[1, 0] = 3.0
        np.testing.assert_array_equal(layout, expected)

    def test_gather_inverts_scatter(self) -> None:
        rng = np.random.default_rng(4)
        scan = ZigzagScan(block_size=8)
        coefficients = rng.random((3, 5, 15), dtype=np.float32)
        restored = scan.gather(scan.scatter(coefficients), 15)
        np.testing.assert_array_equal(restored, coefficients)

    @pytest.mark.parametrize("coefficient_count", [0, 65])
    def test_rejects_out_of_range_count(self, coefficient_count: int) -> None:
        scan = ZigzagScan(block_size=8)
        coefficients = np.zeros((1, 1, coefficient_count), dtype=np.float32)
        with pytest.raises(ValueError, match="coefficient_count must be in"):
            _ = scan.scatter(coefficients)

    def test_validate_truncated_layout_rejects_nan_in_dropped_slot(self) -> None:
        layout = np.zeros((8, 8), dtype=np.float32)
        layout[7, 7] = np.nan
        with pytest.raises(ValueError, match="must be zero"):
            ZigzagScan(block_size=8).validate_truncated_layout(layout, 15)

    def test_validate_truncated_layout_accepts_values_in_kept_slots(self) -> None:
        scan = ZigzagScan(block_size=8)
        coefficients = np.ones((2, 3, 15), dtype=np.float32)
        scan.validate_truncated_layout(scan.scatter(coefficients), 15)


class TestBlockedDctImageCoefficientCount:
    def test_from_array_marks_every_coefficient_available(
        self,
        bgr_image_even: np.ndarray,
    ) -> None:
        container = BlockedDctImage.from_array(
            bgr_image_even,
            ChannelOrder.BGR,
            block_size=8,
        )
        assert container.coefficient_count == 64
        assert container.is_complete

    def test_rejects_non_zero_beyond_coefficient_count(self) -> None:
        value = np.zeros((8, 8), dtype=np.float32)
        value[7, 7] = 1.0
        with pytest.raises(ValueError, match="must be zero"):
            _ = BlockedDctImage(value=value, block_size=8, coefficient_count=15)

    @pytest.mark.parametrize("coefficient_count", [0, 65])
    def test_rejects_out_of_range_count(self, coefficient_count: int) -> None:
        value = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="coefficient_count must be in"):
            _ = BlockedDctImage(
                value=value,
                block_size=8,
                coefficient_count=coefficient_count,
            )

    def test_filters_preserve_coefficient_count(self) -> None:
        rng = np.random.default_rng(5)
        truncated = ZigzagDctCoefficients(
            value=rng.random((2, 2, 15), dtype=np.float32),
            block_size=8,
        ).to_blocked()
        assert truncated.low_pass_filter(4.0).coefficient_count == 15
        assert truncated.high_pass_filter(1.0).coefficient_count == 15
        assert not truncated.is_complete


class TestZigzagDctCoefficients:
    def test_full_roundtrip_restores_image(
        self,
        bgr_image_even: np.ndarray,
    ) -> None:
        container = BlockedDctImage.from_array(
            bgr_image_even,
            ChannelOrder.BGR,
            block_size=8,
        )
        zigzag = ZigzagDctCoefficients.from_blocked(container)
        rebuilt = zigzag.to_blocked()
        assert zigzag.coefficient_count == 64
        np.testing.assert_array_equal(rebuilt.value, container.value)
        np.testing.assert_allclose(
            rebuilt.idct(is_uint8_cast_enabled=False),
            container.idct(is_uint8_cast_enabled=False),
            atol=1e-3,
        )

    def test_truncated_keeps_only_low_frequencies(
        self,
        bgr_image_even: np.ndarray,
    ) -> None:
        block_size = 8
        container = BlockedDctImage.from_array(
            bgr_image_even,
            ChannelOrder.BGR,
            block_size=block_size,
        )
        truncated = ZigzagDctCoefficients.from_blocked(container, 15).to_blocked()
        first_block = truncated.value[:block_size, :block_size]
        rows, columns = np.nonzero(first_block)
        assert truncated.coefficient_count == 15
        assert int(np.max(rows + columns)) <= 4
        np.testing.assert_array_equal(
            first_block[rows, columns],
            container.value[:block_size, :block_size][rows, columns],
        )

    def test_from_blocked_rejects_more_than_available(self) -> None:
        truncated = ZigzagDctCoefficients(
            value=np.zeros((1, 1, 15), dtype=np.float32),
            block_size=8,
        ).to_blocked()
        with pytest.raises(ValueError, match=r"must be in \[1, 15\]"):
            _ = ZigzagDctCoefficients.from_blocked(truncated, 16)

    def test_rejects_non_3d_value(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            _ = ZigzagDctCoefficients(
                value=np.zeros((8, 8), dtype=np.float32),
                block_size=8,
            )

    def test_rejects_non_float32_value(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            _ = ZigzagDctCoefficients(
                value=np.zeros((1, 1, 15), dtype=np.float64),
                block_size=8,
            )

    @pytest.mark.parametrize("coefficient_count", [0, 65])
    def test_rejects_out_of_range_count(self, coefficient_count: int) -> None:
        with pytest.raises(ValueError, match="coefficient_count must be in"):
            _ = ZigzagDctCoefficients(
                value=np.zeros((1, 1, coefficient_count), dtype=np.float32),
                block_size=8,
            )

    def test_roundtrip_preserves_source_shape(
        self,
        bgr_image_odd: np.ndarray,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            container = BlockedDctImage.from_array(
                bgr_image_odd,
                ChannelOrder.BGR,
                block_size=8,
            )
        zigzag = ZigzagDctCoefficients.from_blocked(container, 15)
        rebuilt = zigzag.to_blocked()
        assert zigzag.source_shape == (33, 65)
        assert rebuilt.source_shape == (33, 65)
        assert rebuilt.idct().shape == (33, 65)

    @pytest.mark.parametrize("source_shape", [(0, 8), (8, 25), (17, 8)])
    def test_rejects_source_shape_outside_grid(
        self,
        source_shape: tuple[int, int],
    ) -> None:
        with pytest.raises(ValueError, match="must be positive and fit"):
            _ = ZigzagDctCoefficients(
                value=np.zeros((2, 3, 15), dtype=np.float32),
                block_size=8,
                source_shape=source_shape,
            )
