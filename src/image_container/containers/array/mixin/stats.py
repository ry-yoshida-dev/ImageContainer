from __future__ import annotations

import cv2
import numpy as np

from ....ch_order import ChannelOrder
from ....types import ImageArray, UInt8Image
from .convert import ArrayConvertMixin


class ArrayStatsMixin(ArrayConvertMixin):
    """
    Mixin for aggregate statistics and histograms on array-backed containers.

    Gray statistics use to_array(ChannelOrder.GRAY). BGR statistics use
    to_array(ChannelOrder.BGR); per-channel arrays are indexed B, G, R at 0, 1, 2.
    """

    @property
    def mean_gray(self) -> float:
        """Mean pixel value over the grayscale plane."""
        gray = self.to_array(ChannelOrder.GRAY)
        return float(np.mean(gray))

    @property
    def max_gray(self) -> float:
        """Maximum pixel value over the grayscale plane."""
        gray = self.to_array(ChannelOrder.GRAY)
        return float(np.max(gray))

    @property
    def min_gray(self) -> float:
        """Minimum pixel value over the grayscale plane."""
        gray = self.to_array(ChannelOrder.GRAY)
        return float(np.min(gray))

    @property
    def mean_bgr(self) -> ImageArray:
        """
        Per-channel means in BGR order after conversion to BGR.

        Returns
        -------
        ImageArray
            Shape (3,), dtype float64; indices 0, 1, 2 are B, G, R.
        """
        bgr = self.to_array(ChannelOrder.BGR)
        return np.mean(bgr, axis=(0, 1))

    @property
    def max_bgr(self) -> ImageArray:
        """
        Per-channel maxima in BGR order after conversion to BGR.

        Returns
        -------
        ImageArray
            Shape (3,); indices 0, 1, 2 are B, G, R. Dtype follows the BGR array.
        """
        bgr = self.to_array(ChannelOrder.BGR)
        return np.max(bgr, axis=(0, 1))

    @property
    def min_bgr(self) -> ImageArray:
        """
        Per-channel minima in BGR order after conversion to BGR.

        Returns
        -------
        ImageArray
            Shape (3,); indices 0, 1, 2 are B, G, R. Dtype follows the BGR array.
        """
        bgr = self.to_array(ChannelOrder.BGR)
        return np.min(bgr, axis=(0, 1))

    def compute_histogram(
        self,
        bins: int = 256,
        hist_range: tuple[float, float] = (0.0, 256.0),
        mask: ImageArray | None = None,
        *,
        histogram_order: ChannelOrder = ChannelOrder.GRAY,
    ) -> ImageArray:
        """
        Histogram using OpenCV cv2.calcHist.

        Default (histogram_order is GRAY): multi-channel images are converted to
        grayscale first; result shape (bins, 1), dtype float32.

        With histogram_order=ChannelOrder.BGR: histograms are computed separately
        for each BGR channel after converting to BGR; result shape (bins, 3)
        (columns are B, G, R in that order).

        Parameters
        ----------
        bins : int
            Number of histogram bins per channel.
        hist_range : tuple[float, float]
            Pixel value range included in the histogram.
        mask : ImageArray or None
            Optional (H, W) mask; non-zero pixels are counted (common case: 0/1
            uint8). None means use the full image.
        histogram_order : ChannelOrder
            ChannelOrder.GRAY (default) or ChannelOrder.BGR only.

        Returns
        -------
        ImageArray
            (bins, 1) for grayscale mode, (bins, 3) for BGR mode; dtype float32.

        Raises
        ------
        ValueError
            If histogram_order is not GRAY or BGR.
        """
        rng = list(hist_range)
        match histogram_order:
            case ChannelOrder.GRAY:
                gray: UInt8Image = self.to_array(ChannelOrder.GRAY)
                return cv2.calcHist([gray], [0], mask, [bins], rng)
            case ChannelOrder.BGR:
                bgr: UInt8Image = self.to_array(ChannelOrder.BGR)
                h0 = cv2.calcHist([bgr], [0], mask, [bins], rng)
                h1 = cv2.calcHist([bgr], [1], mask, [bins], rng)
                h2 = cv2.calcHist([bgr], [2], mask, [bins], rng)
                return np.concatenate([h0, h1, h2], axis=1)
            case _:
                raise ValueError(
                    "histogram_order must be ChannelOrder.GRAY or ChannelOrder.BGR, "
                    f"got {histogram_order!r}"
                )
