from __future__ import annotations

import cv2

from ..types import UInt8Image

_EVEN_DIMENSION_REMAINDER = 0


def padding_to_even(
    array: UInt8Image,
) -> tuple[UInt8Image, tuple[int, int] | None]:
    """
    Pad a 2D array so height and width are even.

    When either spatial dimension is odd, replicates one pixel on the bottom
    and/or right edge via cv2.BORDER_REPLICATE.

    Parameters
    ----------
    array : UInt8Image
        Input array of shape (height, width).

    Returns
    -------
    array : UInt8Image
        Input or padded array with even height and width.
    source_shape : tuple[int, int] | None
        Original (height, width) when padding was applied; otherwise None.

    Raises
    ------
    ValueError
        If the input is not a 2D array.
    """
    if array.ndim != 2:
        raise ValueError(f"Array must be 2D, got {array.ndim} dimensions")

    height, width = array.shape
    pad_bottom = height % 2
    pad_right = width % 2
    if (
        pad_bottom == _EVEN_DIMENSION_REMAINDER
        and pad_right == _EVEN_DIMENSION_REMAINDER
    ):
        return array, None

    padded = cv2.copyMakeBorder(
        array,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    return padded, (height, width)


def padding_to_block_size(
    array: UInt8Image,
    block_size: int,
) -> tuple[UInt8Image, tuple[int, int] | None]:
    """
    Pad a 2D array so height and width are multiples of block_size.

    When either spatial dimension is not divisible by block_size, replicates
    one or more pixels on the bottom and/or right edge via cv2.BORDER_REPLICATE.

    Parameters
    ----------
    array : UInt8Image
        Input array of shape (height, width).
    block_size : int
        Block side length; must be positive.

    Returns
    -------
    array : UInt8Image
        Input or padded array with height and width divisible by block_size.
    source_shape : tuple[int, int] | None
        Original (height, width) when padding was applied; otherwise None.

    Raises
    ------
    ValueError
        If the input is not a 2D array or block_size is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if array.ndim != 2:
        raise ValueError(f"Array must be 2D, got {array.ndim} dimensions")

    height, width = array.shape
    pad_bottom = (block_size - (height % block_size)) % block_size
    pad_right = (block_size - (width % block_size)) % block_size
    if pad_bottom == 0 and pad_right == 0:
        return array, None

    padded = cv2.copyMakeBorder(
        array,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    return padded, (height, width)
