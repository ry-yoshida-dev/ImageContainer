from __future__ import annotations

import cv2

from collections.abc import Callable

from ....ch_order import ChannelOrder
from ....types import UInt8Image
from .convert import ArrayConvertMixin


class ArrayProcessMixin(ArrayConvertMixin):
    """
    Mixin for OpenCV-based processing helpers on array-backed image containers.

    Subclasses ArrayConvertMixin for to_array and shared annotations.
    """

    def equalize_histogram(self) -> UInt8Image:
        """
        Histogram equalization.

        Grayscale images use cv2.equalizeHist. Color images are equalized on the
        L channel in LAB space (via ChannelOrder.LAB and to_array).

        Returns
        -------
        UInt8Image
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
    ) -> UInt8Image:
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
        UInt8Image
            Result array: grayscale (H, W) or color with the same channel layout as
            to_array would produce for this container's channel_order.
        """
        if self.channel_order == ChannelOrder.GRAY:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(self.value)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return self._lab_with_modified_l(clahe.apply)

    def _lab_with_modified_l(self, modify_l: Callable[[UInt8Image], UInt8Image]) -> UInt8Image:
        """
        Apply a single-channel image function to only the L plane in LAB.

        Converts self to LAB, applies modify_l to the L channel only (A and B
        are kept as-is to preserve color), then converts back to the container's
        original channel_order.

        Parameters
        ----------
        modify_l : Callable[[UInt8Image], UInt8Image]
            Function applied to the L channel. It receives a single-channel
            uint8 array of shape (H, W) and must return a modified array of the
            same shape and dtype. Typical examples are cv2.equalizeHist and
            cv2.CLAHE.apply.

        Returns
        -------
        UInt8Image
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
