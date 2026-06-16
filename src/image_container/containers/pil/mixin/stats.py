from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageStat

from ....ch_order import ChannelOrder
from ....types import ImageArray
from .convert import PILConvertMixin


class PILStatsMixin(PILConvertMixin):
    """
    Mixin for aggregate statistics and histograms on PIL-backed containers.

    Gray statistics use to_gray(); BGR-order aggregates use RGB triplet storage
    via to_rgb_image(), reported as B, G, R at indices 0, 1, 2.
    """

    def _gray_stat(self) -> ImageStat.Stat:
        return ImageStat.Stat(self.to_gray())

    def _rgb_stat(self) -> ImageStat.Stat:
        return ImageStat.Stat(self.to_rgb_image())

    @property
    def mean_gray(self) -> float:
        """Mean pixel value over the grayscale plane."""
        return float(self._gray_stat().mean[0])

    @property
    def max_gray(self) -> float:
        """Maximum pixel value over the grayscale plane."""
        return float(self._gray_stat().extrema[0][1])

    @property
    def min_gray(self) -> float:
        """Minimum pixel value over the grayscale plane."""
        return float(self._gray_stat().extrema[0][0])

    @property
    def mean_bgr(self) -> ImageArray:
        """
        Per-channel means in BGR order (RGB storage interpreted as BGR indices).

        Returns
        -------
        ImageArray
            Shape (3,), dtype float64; indices 0, 1, 2 are B, G, R.
        """
        m = self._rgb_stat().mean
        return np.array([m[2], m[1], m[0]], dtype=np.float64)

    @property
    def max_bgr(self) -> ImageArray:
        """
        Per-channel maxima in BGR order (RGB storage interpreted as BGR indices).

        Returns
        -------
        ImageArray
            Shape (3,); indices 0, 1, 2 are B, G, R.
        """
        e = self._rgb_stat().extrema
        return np.array([e[2][1], e[1][1], e[0][1]], dtype=np.int64)

    @property
    def min_bgr(self) -> ImageArray:
        """
        Per-channel minima in BGR order (RGB storage interpreted as BGR indices).

        Returns
        -------
        ImageArray
            Shape (3,); indices 0, 1, 2 are B, G, R.
        """
        e = self._rgb_stat().extrema
        return np.array([e[2][0], e[1][0], e[0][0]], dtype=np.int64)

    def _histogram_via_opencv(
        self,
        bins: int,
        hist_range: tuple[float, float],
        mask: ImageArray | None,
        histogram_order: ChannelOrder,
    ) -> ImageArray:
        rng = list(hist_range)
        match histogram_order:
            case ChannelOrder.GRAY:
                gray = self.to_array(ChannelOrder.GRAY)
                return cv2.calcHist([gray], [0], mask, [bins], rng)
            case ChannelOrder.BGR:
                bgr = self.to_array(ChannelOrder.BGR)
                h0 = cv2.calcHist([bgr], [0], mask, [bins], rng)
                h1 = cv2.calcHist([bgr], [1], mask, [bins], rng)
                h2 = cv2.calcHist([bgr], [2], mask, [bins], rng)
                return np.concatenate([h0, h1, h2], axis=1)
            case _:
                raise ValueError(
                    "histogram_order must be ChannelOrder.GRAY or ChannelOrder.BGR, "
                    f"got {histogram_order!r}"
                )

    def compute_histogram(
        self,
        bins: int = 256,
        hist_range: tuple[float, float] = (0.0, 256.0),
        mask: ImageArray | None = None,
        *,
        histogram_order: ChannelOrder = ChannelOrder.GRAY,
    ) -> ImageArray:
        """
        Histogram: Pillow for 8-bit, 256 bins, full range, compatible mask; else OpenCV.

        Default (histogram_order is GRAY): grayscale; result shape (bins, 1), float32.

        With histogram_order=ChannelOrder.BGR: result shape (bins, 3); columns B, G, R.

        Parameters
        ----------
        bins : int
            Number of histogram bins per channel.
        hist_range : tuple[float, float]
            Pixel value range included in the histogram.
        mask : ImageArray or None
            Optional (H, W) mask; non-zero pixels are counted. For the Pillow path the
            mask is converted to mode L.
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
        use_pil = bins == 256 and hist_range == (0.0, 256.0)
        pil_mask: Image.Image | None = None
        if mask is not None:
            pil_mask = Image.fromarray(np.asarray(mask).astype(np.uint8), mode="L")

        if use_pil:
            match histogram_order:
                case ChannelOrder.GRAY:
                    h = self.to_gray().histogram(pil_mask)
                    if len(h) != bins:
                        return self._histogram_via_opencv(
                            bins, hist_range, mask, histogram_order
                        )
                    return np.asarray(h, dtype=np.float32).reshape(bins, 1)
                case ChannelOrder.BGR:
                    flat = self.to_rgb_image().histogram(pil_mask)
                    if len(flat) != bins * 3:
                        return self._histogram_via_opencv(
                            bins, hist_range, mask, histogram_order
                        )
                    r = flat[0:bins]
                    g = flat[bins : bins * 2]
                    b = flat[bins * 2 : bins * 3]
                    return np.column_stack(
                        [
                            np.asarray(b, dtype=np.float32),
                            np.asarray(g, dtype=np.float32),
                            np.asarray(r, dtype=np.float32),
                        ]
                    )
                case _:
                    raise ValueError(
                        "histogram_order must be ChannelOrder.GRAY or ChannelOrder.BGR, "
                        f"got {histogram_order!r}"
                    )

        return self._histogram_via_opencv(bins, hist_range, mask, histogram_order)
