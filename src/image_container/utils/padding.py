from __future__ import annotations

import cv2
import numpy as np

_EVEN_DIMENSION_REMAINDER = 0


def padding_to_even(
    array: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """
    Pad a 2D array so height and width are even.

    When either spatial dimension is odd, replicates one pixel on the bottom
    and/or right edge via cv2.BORDER_REPLICATE.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (height, width).

    Returns
    -------
    array : np.ndarray
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
