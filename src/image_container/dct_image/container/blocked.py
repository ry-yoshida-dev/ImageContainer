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


@dataclass(kw_only=True)
class BlockedDctImage(DctImageBase):
    """
    A container for block-wise 2D DCT coefficient images.

    Parameters
    ----------
    value : ImageArray
        DCT coefficients as a 2D float32 array of shape (height, width).
    block_size : int
        Side length of each square DCT block in pixels.
    source_shape : tuple[int, int] | None, default None
        Original (height, width) before block-size padding in from_array.
        When set, idct crops the restored image back to this shape.
    """

    block_size: int

    def _validate_transform_specific(self) -> None:
        """
        Validate block_size and spatial divisibility.

        Raises
        ------
        ValueError
            If block_size is not positive or does not divide spatial dimensions.
        """
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        height, width = self.value.shape
        if height % self.block_size != 0 or width % self.block_size != 0:
            raise ValueError(
                "DCT image shape "
                f"{self.value.shape} must be divisible by block_size "
                f"{self.block_size}"
            )

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
            New container with the same block_size and source_shape.
        """
        return BlockedDctImage(
            value=filtered,
            block_size=self.block_size,
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
            Block DCT coefficients with shape (H, W) and float32 dtype.

        Warns
        -----
        UserWarning
            When height or width is not divisible by block_size, the grayscale
            image is padded on the bottom and/or right before DCT.

        Raises
        ------
        ValueError
            If block_size is not positive.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
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
        block_transform = BlockDctTransform(block_size=block_size)
        coefficients = block_transform.forward(gray.astype(np.float32))
        return cls(
            value=coefficients,
            block_size=block_size,
            source_shape=source_shape,
        )
