from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from ...types import ImageArray


@dataclass(frozen=True)
class BlockDctTransform:
    """
    Forward and inverse 2D DCT applied independently to non-overlapping blocks.

    The 2D DCT is separable, so every block row of the image is transformed in
    one cv2.dct / cv2.idct call with cv2.DCT_ROWS, then every block column in a
    second call. The result is identical to calling cv2.dct / cv2.idct on each
    block separately.

    Parameters
    ----------
    block_size : int
        Side length of each square block in pixels. Must be positive and even,
        because OpenCV only implements the DCT for even-length rows.
    """

    block_size: int

    def __post_init__(self) -> None:
        """
        Validate block size.

        Raises
        ------
        ValueError
            If block_size is not positive or not even.
        """
        if self.block_size <= 0 or self.block_size % 2 != 0:
            raise ValueError(
                f"block_size must be a positive even number, got {self.block_size}"
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
        self.validate_spatial_array(gray)
        return self._apply_separable(gray, self._forward_rows)

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
        self.validate_spatial_array(coefficients)
        return self._apply_separable(coefficients, self._inverse_rows)

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

    @staticmethod
    def _forward_rows(rows: ImageArray) -> ImageArray:
        """
        Apply the 1D forward DCT to every row.

        Parameters
        ----------
        rows : ImageArray
            Contiguous float32 array of shape (row_count, block_size).

        Returns
        -------
        ImageArray
            Row-wise DCT coefficients with the same shape as rows.
        """
        return np.asarray(cv2.dct(rows, flags=cv2.DCT_ROWS), dtype=np.float32)

    @staticmethod
    def _inverse_rows(rows: ImageArray) -> ImageArray:
        """
        Apply the 1D inverse DCT to every row.

        Parameters
        ----------
        rows : ImageArray
            Contiguous float32 array of shape (row_count, block_size).

        Returns
        -------
        ImageArray
            Row-wise restored samples with the same shape as rows.
        """
        return np.asarray(cv2.idct(rows, flags=cv2.DCT_ROWS), dtype=np.float32)

    def _apply_separable(
        self,
        array: ImageArray,
        row_transform: Callable[[ImageArray], ImageArray],
    ) -> ImageArray:
        """
        Apply a 1D transform along block rows, then along block columns.

        Parameters
        ----------
        array : ImageArray
            Input array of shape (height, width) divisible by block_size.
        row_transform : Callable[[ImageArray], ImageArray]
            1D transform applied to each row of a (row_count, block_size) array.

        Returns
        -------
        ImageArray
            Transformed float32 array with the same shape as array.
        """
        height, width = array.shape
        blocks = array.reshape(
            height // self.block_size,
            self.block_size,
            width // self.block_size,
            self.block_size,
        ).transpose(0, 2, 1, 3)
        block_shape = blocks.shape
        row_transformed = row_transform(
            np.ascontiguousarray(blocks, dtype=np.float32).reshape(-1, self.block_size)
        ).reshape(block_shape)
        columns = np.ascontiguousarray(row_transformed.transpose(0, 1, 3, 2))
        fully_transformed = row_transform(columns.reshape(-1, self.block_size)).reshape(
            block_shape
        )
        restored: ImageArray = fully_transformed.transpose(0, 3, 1, 2).reshape(
            height,
            width,
        )
        return restored

    def validate_spatial_array(self, array: ImageArray) -> None:
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
