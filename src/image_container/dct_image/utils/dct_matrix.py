from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from dataclasses import dataclass

from ...types import ImageArray

_DCT_ROWS_FLAG = cv2.DCT_ROWS
_MAX_CACHED_BLOCK_SIZES = 16


def _build_transform_matrix(block_size: int, *, is_inverse: bool) -> ImageArray:
    """
    Build a flattened 2D block DCT or IDCT matrix via row-wise OpenCV transforms.

    The 2D DCT is separable: apply 1D DCT along rows, then along columns.
    Each identity-basis block is a single spatial impulse; transforming it yields
    one column of the flattened block operator. The final transpose maps
    ``flattened_spatial @ matrix.T`` to the coefficient layout used in
    ``BlockDctTransform._apply_block_matrix``.

    Parameters
    ----------
    block_size : int
        Side length of each square block in pixels.
    is_inverse : bool
        If True, build the inverse (IDCT) matrix; otherwise the forward matrix.

    Returns
    -------
    ImageArray
        Square matrix of shape (block_size ** 2, block_size ** 2).
    """
    pixel_count = block_size * block_size
    transform = cv2.idct if is_inverse else cv2.dct
    identity_basis = np.eye(pixel_count, dtype=np.float32).reshape(
        pixel_count,
        block_size,
        block_size,
    )
    flat_rows = identity_basis.reshape(pixel_count * block_size, block_size)
    row_transformed = transform(flat_rows, flags=_DCT_ROWS_FLAG).reshape(
        pixel_count,
        block_size,
        block_size,
    )
    flat_columns = row_transformed.transpose(0, 2, 1).reshape(
        pixel_count * block_size,
        block_size,
    )
    block_results = transform(flat_columns, flags=_DCT_ROWS_FLAG).reshape(
        pixel_count,
        block_size,
        block_size,
    ).transpose(0, 2, 1)
    return block_results.reshape(pixel_count, pixel_count).T


@lru_cache(maxsize=_MAX_CACHED_BLOCK_SIZES)
def _cached_block_dct_matrices(block_size: int) -> BlockDctMatrices:
    """
    Build or retrieve LRU-cached DCT matrices for a block size.

    Parameters
    ----------
    block_size : int
        Side length of each square block in pixels.

    Returns
    -------
    BlockDctMatrices
        Forward and inverse matrices for the given block size.
    """
    return BlockDctMatrices(
        forward=_build_transform_matrix(block_size, is_inverse=False),
        inverse=_build_transform_matrix(block_size, is_inverse=True),
        block_size=block_size,
    )


@dataclass(frozen=True)
class BlockDctMatrices:
    """
    Cached forward and inverse 2D DCT linear maps for a fixed block size.

    Parameters
    ----------
    forward : ImageArray
        Matrix of shape (block_size ** 2, block_size ** 2) mapping flattened
        spatial blocks to flattened DCT coefficients via right multiplication
        by forward.T.
    inverse : ImageArray
        Matrix of shape (block_size ** 2, block_size ** 2) mapping flattened
        DCT coefficients back to flattened spatial blocks via right
        multiplication by inverse.T.
    block_size : int
        Side length of each square block in pixels.
    """

    forward: ImageArray
    inverse: ImageArray
    block_size: int

    @classmethod
    def from_block_size(cls, block_size: int) -> BlockDctMatrices:
        """
        Build or retrieve cached DCT matrices for a block size.

        Parameters
        ----------
        block_size : int
            Side length of each square block in pixels.

        Returns
        -------
        BlockDctMatrices
            Forward and inverse matrices for the given block size.

        Raises
        ------
        ValueError
            If block_size is not positive.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        return _cached_block_dct_matrices(block_size)
