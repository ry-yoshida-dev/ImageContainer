"""
Mixin for OpenCV grayscale derivative filters on array-backed image containers.
"""

from __future__ import annotations

import cv2
import numpy as np

from ....ch_order import ChannelOrder
from ....types import ImageArray
from .convert import ArrayConvertMixin


class ArrayFilterMixin(ArrayConvertMixin):
    """
    Mixin for Laplacian and Sobel on a grayscale view of the container image.

    Subclasses ArrayConvertMixin for ``to_array`` and channel-order handling.
    """

    def laplacian(
        self,
        ksize: int = 3,
        *,
        is_uint8_output: bool = True,
    ) -> ImageArray:
        """
        Laplacian filter on a grayscale view of the image.

        Uses cv2.Laplacian with cv2.CV_64F. When is_uint8_output is True (default),
        cv2.convertScaleAbs is applied so values fit uint8.

        Parameters
        ----------
        ksize : int
            Aperture size (1, 3, 5, ...).
        is_uint8_output : bool
            If True, convert to uint8 via convertScaleAbs for typical visualization.

        Returns
        -------
        ImageArray
            Single-channel array of shape (H, W); uint8 if is_uint8_output is True,
            otherwise float64.
        """
        gray: ImageArray = self.to_array(ChannelOrder.GRAY)
        lap: ImageArray = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        if is_uint8_output:
            return cv2.convertScaleAbs(lap)
        return lap.astype(np.float64)

    def sobel(
        self,
        ksize: int = 3,
        *,
        is_uint8_output: bool = True,
    ) -> ImageArray:
        """
        Sobel derivatives on a grayscale view of the image.

        Computes horizontal and vertical Sobel with cv2.CV_64F and stacks them along
        the last axis as (G_x, G_y). Magnitude is cv2.magnitude(gx, gy) if needed.
        When is_uint8_output is True (default), convertScaleAbs is applied per
        component for typical visualization.

        Parameters
        ----------
        ksize : int
            Aperture size (1, 3, 5, ...).
        is_uint8_output : bool
            If True, convert each derivative to uint8 via convertScaleAbs.

        Returns
        -------
        ImageArray
            Array of shape (H, W, 2): channel 0 is ∂I/∂x, channel 1 is ∂I/∂y.
            dtype uint8 if is_uint8_output is True, otherwise float64.
        """
        gray: ImageArray = self.to_array(ChannelOrder.GRAY)
        gx: ImageArray = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy: ImageArray = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        if is_uint8_output:
            return np.stack(
                [cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy)],
                axis=-1,
            )
        return np.stack([gx.astype(np.float64), gy.astype(np.float64)], axis=-1)
