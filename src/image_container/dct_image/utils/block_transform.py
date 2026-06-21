from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ...types import ImageArray
from .dct_matrix import BlockDctMatrices

_MIN_BLOCK_SIZE = 1


@dataclass(frozen=True)
class BlockDctTransform:
    """
    Forward and inverse 2D DCT applied independently to non-overlapping blocks.

    Block transforms are vectorized with batched matrix multiplication against
    precomputed OpenCV-compatible 2D DCT matrices.

    Parameters
    ----------
    block_size : int
        Side length of each square block in pixels.
    """

    block_size: int

    def __post_init__(self) -> None:
        """
        Validate block size.

        Raises
        ------
        ValueError
            If block_size is less than 1.
        """
        if self.block_size < _MIN_BLOCK_SIZE:
            raise ValueError(
                f"block_size must be at least {_MIN_BLOCK_SIZE}, got {self.block_size}"
            )

    def forward(self, gray: ImageArray) -> ImageArray:
        """
        Apply block-wise forward DCT.

        Parameters
        ----------
        gray : ImageArray
            Grayscale float32 array of shape (height, width). Height and width
            must be divisible by block_size.

        Returns
        -------
        ImageArray
            Block DCT coefficients with the same shape as gray.

        Raises
        ------
        ValueError
            If gray is not 2D or spatial dimensions are not divisible by
            block_size.
        """
        self._validate_spatial_array(gray)
        matrices = BlockDctMatrices.from_block_size(self.block_size)
        return self._apply_block_matrix(gray, matrices.forward)

    def inverse(self, coefficients: ImageArray) -> ImageArray:
        """
        Apply block-wise inverse DCT.

        Parameters
        ----------
        coefficients : ImageArray
            Block DCT coefficients of shape (height, width). Height and width
            must be divisible by block_size.

        Returns
        -------
        ImageArray
            Restored grayscale float32 array with the same shape as coefficients.

        Raises
        ------
        ValueError
            If coefficients is not 2D or spatial dimensions are not divisible
            by block_size.
        """
        self._validate_spatial_array(coefficients)
        matrices = BlockDctMatrices.from_block_size(self.block_size)
        return self._apply_block_matrix(coefficients, matrices.inverse)

    def frequency_radius_squared(self, height: int, width: int) -> ImageArray:
        """
        Build a tiled per-block squared radial frequency index mask.

        Parameters
        ----------
        height : int
            Output mask height; must be divisible by block_size.
        width : int
            Output mask width; must be divisible by block_size.

        Returns
        -------
        ImageArray
            Mask array of shape (height, width) with float32 dtype.

        Raises
        ------
        ValueError
            If height or width is not divisible by block_size.
        """
        if height % self.block_size != 0:
            raise ValueError(
                f"height {height} must be divisible by block_size {self.block_size}"
            )
        if width % self.block_size != 0:
            raise ValueError(
                f"width {width} must be divisible by block_size {self.block_size}"
            )
        local_v_indices = np.arange(self.block_size, dtype=np.float32)[:, np.newaxis]
        local_u_indices = np.arange(self.block_size, dtype=np.float32)[np.newaxis, :]
        local_mask = (local_u_indices**2) + (local_v_indices**2)
        row_tiles = height // self.block_size
        col_tiles = width // self.block_size
        return np.tile(local_mask, (row_tiles, col_tiles))

    def _apply_block_matrix(
        self,
        array: ImageArray,
        matrix: ImageArray,
    ) -> ImageArray:
        """
        Reshape spatial data into blocks, apply a linear map, and flatten back.

        Parameters
        ----------
        array : ImageArray
            Input array of shape (height, width).
        matrix : ImageArray
            Square matrix of shape (pixel_count, pixel_count) applied to each
            flattened block via right multiplication by matrix.T.

        Returns
        -------
        ImageArray
            Transformed array with the same shape as array.
        """
        height, width = array.shape
        pixel_count = self.block_size * self.block_size
        block_rows = height // self.block_size
        block_cols = width // self.block_size
        blocks = array.reshape(
            block_rows,
            self.block_size,
            block_cols,
            self.block_size,
        ).transpose(0, 2, 1, 3)
        transformed = blocks.reshape(-1, pixel_count) @ matrix.T
        return transformed.reshape(blocks.shape).transpose(0, 2, 1, 3).reshape(
            height,
            width,
        )

    def _validate_spatial_array(self, array: ImageArray) -> None:
        """
        Validate a 2D array whose dimensions are divisible by block_size.

        Parameters
        ----------
        array : ImageArray
            Input array to validate.

        Raises
        ------
        ValueError
            If array is not 2D or spatial dimensions are not divisible by
            block_size.
        """
        if array.ndim != 2:
            raise ValueError(
                f"Block DCT expects a 2D array, got {array.ndim} dimensions"
            )
        height, width = array.shape
        if height % self.block_size != 0:
            raise ValueError(
                f"height {height} must be divisible by block_size {self.block_size}"
            )
        if width % self.block_size != 0:
            raise ValueError(
                f"width {width} must be divisible by block_size {self.block_size}"
            )
