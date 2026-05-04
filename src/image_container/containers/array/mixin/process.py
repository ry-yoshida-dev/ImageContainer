from __future__ import annotations

import cv2
import numpy as np

from ....ch_order import ChannelOrder
from .convert import ArrayConvertMixin


class ArrayProcessMixin(ArrayConvertMixin):
    """
    Mixin for OpenCV-based processing helpers on array-backed image containers.

    Subclasses ArrayConvertMixin for to_array and shared annotations.
    """

    @staticmethod
    def _bgr_equalized_lab_luminance(bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = cv2.equalizeHist(l_ch)
        merged = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _bgr_clahe_lab_luminance(
        bgr: np.ndarray,
        clip_limit: float,
        tile_grid_size: tuple[int, int],
    ) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l2 = clahe.apply(l_ch)
        merged = cv2.merge([l2, a_ch, b_ch])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def equalize_histogram(self) -> np.ndarray:
        """
        Histogram equalization.

        Grayscale images use cv2.equalizeHist. Color images are equalized on the
        L channel in LAB space (after converting to BGR for processing).

        Returns
        -------
        np.ndarray
            Equalized image array with the same channel layout as to_array would
            produce for this container's channel_order.

        Notes
        -----
        OpenCV equalizeHist expects 8-bit single-channel images for grayscale.
        """
        from ..container import ArrayImageContainer

        if self.channel_order == ChannelOrder.GRAY:
            return cv2.equalizeHist(self.value)

        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        out_bgr = self._bgr_equalized_lab_luminance(bgr)
        return ArrayImageContainer(value=out_bgr, channel_order=ChannelOrder.BGR).to_array(
            self.channel_order
        )

    def clahe(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Grayscale: applied directly to the intensity channel.
        Color: applied to the L channel in LAB space (BGR pipeline).

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
        from ..container import ArrayImageContainer

        if self.channel_order == ChannelOrder.GRAY:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(self.value)

        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        out_bgr = self._bgr_clahe_lab_luminance(bgr, clip_limit, tile_grid_size)
        return ArrayImageContainer(value=out_bgr, channel_order=ChannelOrder.BGR).to_array(
            self.channel_order
        )

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
