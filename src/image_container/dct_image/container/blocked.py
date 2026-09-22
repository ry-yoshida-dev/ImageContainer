from __future__ import annotations

import warnings
from functools import cached_property

import numpy as np
from dataclasses import dataclass

from ...ch_order import ChannelOrder
from ...types import BinaryArray, ImageArray, UInt8Image
from ...utils.padding import padding_to_block_size
from ..base import DctImageBase
from ..utils.block_transform import BlockDctTransform
from ..utils.zigzag_scan import ZigzagScan


@dataclass(kw_only=True)
class BlockedDctImage(DctImageBase):
    """
    A container for block-wise 2D DCT coefficient images in spatial layout.

    Every block_size x block_size tile of value holds the coefficients of the
    matching image block in row-major order, so the coefficient for vertical
    frequency v and horizontal frequency u of a block sits at offset (v, u)
    inside the tile. Coefficients in JPEG zigzag order must be converted with
    ZigzagDctCoefficients.to_blocked instead of being reshaped into value.

    Parameters
    ----------
    value : ImageArray
        DCT coefficients as a 2D float32 array of shape (height, width).
    block_size : int
        Side length of each square DCT block in pixels; must be positive and
        even.
    coefficient_count : int
        Number of leading zigzag coefficients per block that carry data; must
        be in [1, block_size ** 2]. Coefficients beyond it were not available
        and must be zero. block_size ** 2 means every coefficient is present.
    source_shape : tuple[int, int] | None, default None
        Original (height, width) before block-size padding in from_array.
        When set, idct crops the restored image back to this shape.
    """

    block_size: int
    coefficient_count: int

    def _validate_transform_specific(self) -> None:
        """
        Validate block_size, spatial divisibility, and coefficient_count.

        Raises
        ------
        ValueError
            If block_size is not a positive even number or does not divide
            spatial dimensions, coefficient_count is out of range, or a
            coefficient beyond coefficient_count is non-zero.
        """
        self._block_transform.validate_spatial_array(self.value)
        ZigzagScan(block_size=self.block_size).validate_truncated_layout(
            self.value,
            self.coefficient_count,
        )

    @property
    def is_complete(self) -> bool:
        """
        Check whether every coefficient of each block is available.

        Returns
        -------
        bool
            True when coefficient_count equals block_size ** 2.
        """
        return self.coefficient_count == self.block_size * self.block_size

    @cached_property
    def _block_transform(self) -> BlockDctTransform:
        """
        Block transform helper configured with this container's block_size.

        Returns
        -------
        BlockDctTransform
            Cached transform for this container's block_size.
        """
        return BlockDctTransform(block_size=self.block_size)

    def _frequency_radius_squared(self) -> ImageArray:
        """
        Build the tiled per-block squared radial frequency index (u^2 + v^2).

        Returns
        -------
        ImageArray
            Mask array of shape (height, width) with float32 dtype.
        """
        height, width = self.value.shape
        return self._block_transform.frequency_radius_squared(height, width)

    def _apply_inverse_dct(self, coefficients: ImageArray) -> ImageArray:
        """
        Apply block-wise inverse DCT.

        Parameters
        ----------
        coefficients : ImageArray
            DCT coefficients of shape (height, width).

        Returns
        -------
        ImageArray
            Restored grayscale float32 array with the same shape as coefficients.
        """
        return self._block_transform.inverse(coefficients)

    def _clear_dc_components(self, mask: BinaryArray) -> None:
        """
        Clear the DC coefficient at the origin of each block.

        Parameters
        ----------
        mask : BinaryArray
            Boolean mask array modified in place.
        """
        mask[0::self.block_size, 0::self.block_size] = False

    def _with_filtered_coefficients(self, filtered: ImageArray) -> BlockedDctImage:
        """
        Build a BlockedDctImage from filtered coefficients.

        Parameters
        ----------
        filtered : ImageArray
            Filtered coefficient array of shape (height, width).

        Returns
        -------
        BlockedDctImage
            New container with the same block_size, coefficient_count, and
            source_shape.
        """
        return BlockedDctImage(
            value=filtered,
            block_size=self.block_size,
            coefficient_count=self.coefficient_count,
            source_shape=self.source_shape,
        )

    @classmethod
    def from_array(
        cls,
        array: UInt8Image,
        channel_order: ChannelOrder,
        block_size: int,
    ) -> BlockedDctImage:
        """
        Create a BlockedDctImage from a block-wise forward DCT.

        The input is converted to grayscale, padded to multiples of block_size,
        then each non-overlapping block is transformed independently.

        Parameters
        ----------
        array : UInt8Image
            Input image array in the given channel order.
        channel_order : ChannelOrder
            Channel layout of the input array.
        block_size : int
            Side length of each square DCT block in pixels.

        Returns
        -------
        BlockedDctImage
            Block DCT coefficients with shape (H, W), float32 dtype, and every
            coefficient available.

        Warns
        -----
        UserWarning
            When height or width is not divisible by block_size, the grayscale
            image is padded on the bottom and/or right before DCT.

        Raises
        ------
        ValueError
            If block_size is not a positive even number.
        """
        block_transform = BlockDctTransform(block_size=block_size)
        to_gray = ChannelOrder.GRAY.cv2_array_converter(channel_order)
        gray = to_gray(array)
        gray, source_shape = padding_to_block_size(gray, block_size)
        if source_shape is not None:
            warnings.warn(
                (
                    f"Grayscale shape {source_shape} is not divisible by "
                    f"block_size {block_size}; padded to {gray.shape} for DCT."
                ),
                UserWarning,
                stacklevel=2,
            )
        coefficients = block_transform.forward(gray.astype(np.float32))
        return cls(
            value=coefficients,
            block_size=block_size,
            coefficient_count=block_size * block_size,
            source_shape=source_shape,
        )
