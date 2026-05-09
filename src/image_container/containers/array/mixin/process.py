from __future__ import annotations

import cv2
import numpy as np

from collections.abc import Callable

from ....ch_order import ChannelOrder
from .convert import ArrayConvertMixin


class ArrayProcessMixin(ArrayConvertMixin):
    """
    Mixin for OpenCV-based processing helpers on array-backed image containers.

    Subclasses ArrayConvertMixin for to_array and shared annotations.
    """

    def equalize_histogram(self) -> np.ndarray:
        """
        Histogram equalization.

        Grayscale images use cv2.equalizeHist. Color images are equalized on the
        L channel in LAB space (via ChannelOrder.LAB and to_array).

        Returns
        -------
        np.ndarray
            Equalized image array with the same channel layout as to_array would
            produce for this container's channel_order.

        Notes
        -----
        OpenCV equalizeHist expects 8-bit single-channel images for grayscale.
        """
        if self.channel_order == ChannelOrder.GRAY:
            return cv2.equalizeHist(self.value)

        return self._lab_with_modified_l(cv2.equalizeHist)

    def clahe(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Grayscale: applied directly to the intensity channel.
        Color: applied to the L channel in LAB space (via ChannelOrder.LAB).

        Parameters
        ----------
        clip_limit : float
            Threshold for contrast limiting.
        tile_grid_size : tuple[int, int]
            Grid block size for adaptive histogram equalization.

        Returns
        -------
        np.ndarray
            Result array: grayscale (H, W) or color with the same channel layout as
            to_array would produce for this container's channel_order.
        """
        if self.channel_order == ChannelOrder.GRAY:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(self.value)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return self._lab_with_modified_l(clahe.apply)

    def laplacian(
        self,
        ksize: int = 3,
        *,
        is_uint8_output: bool = True,
    ) -> np.ndarray:
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
        np.ndarray
            Single-channel array of shape (H, W); uint8 if is_uint8_output is True,
            otherwise float64.
        """
        gray: np.ndarray = self.to_array(ChannelOrder.GRAY)
        lap: np.ndarray = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        if is_uint8_output:
            return cv2.convertScaleAbs(lap)
        return lap.astype(np.float64)

    def sobel(
        self,
        ksize: int = 3,
        *,
        is_uint8_output: bool = True,
    ) -> np.ndarray:
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
        np.ndarray
            Array of shape (H, W, 2): channel 0 is ∂I/∂x, channel 1 is ∂I/∂y.
            dtype uint8 if is_uint8_output is True, otherwise float64.
        """
        gray: np.ndarray = self.to_array(ChannelOrder.GRAY)
        gx: np.ndarray = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy: np.ndarray = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        if is_uint8_output:
            return np.stack(
                [cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy)],
                axis=-1,
            )
        return np.stack([gx.astype(np.float64), gy.astype(np.float64)], axis=-1)

    def _lab_with_modified_l(self, modify_l: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Apply a single-channel image function to only the L plane in LAB.

        Converts self to LAB, applies modify_l to the L channel only (A and B
        are kept as-is to preserve color), then converts back to the container's
        original channel_order.

        Parameters
        ----------
        modify_l : Callable[[np.ndarray], np.ndarray]
            Function applied to the L channel. It receives a single-channel
            uint8 array of shape (H, W) and must return a modified array of the
            same shape and dtype. Typical examples are cv2.equalizeHist and
            cv2.CLAHE.apply.

        Returns
        -------
        np.ndarray
            Image array with L modified, in the same channel layout as
            to_array would produce for this container's channel_order.
        """
        from ..container import ArrayImageContainer

        lab = self.to_array(ChannelOrder.LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        merged = cv2.merge([modify_l(l_ch), a_ch, b_ch])
        return ArrayImageContainer(value=merged, channel_order=ChannelOrder.LAB).to_array(
            self.channel_order
        )
