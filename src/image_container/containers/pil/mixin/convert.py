from __future__ import annotations

import numpy as np
from PIL import Image

from ....binary_image import BinaryImage
from ....ch_order import ChannelOrder
from ....types import UInt8Image


class PILConvertMixin:
    """Convert the held PIL image.

    - NumPy
    - BinaryImage
    - PIL with reordered channels
    """

    value: Image.Image
    channel_order: ChannelOrder

    def to_gray(self) -> Image.Image:
        """
        Pillow image in mode L (grayscale / luminance).

        Same visual gray as to_array(ChannelOrder.GRAY) after ndarray conversion.
        """
        return self.value.convert(ChannelOrder.GRAY.pil_mode)

    def to_3ch(self) -> Image.Image:
        """
        Pillow image in the 3-channel mode for this container's
        channel_order (RGB triplet storage, HSV, or LAB).

        A gray container is promoted to RGB (triplets replicate luminance).
        """
        match self.channel_order:
            case ChannelOrder.GRAY | ChannelOrder.RGB | ChannelOrder.BGR:
                return self.value.convert(ChannelOrder.RGB.pil_mode)
            case ChannelOrder.HSV:
                return self.value.convert(ChannelOrder.HSV.pil_mode)
            case ChannelOrder.LAB:
                return self.value.convert(ChannelOrder.LAB.pil_mode)

    def to_rgb_image(self) -> Image.Image:
        """
        Pillow RGB image (triplet storage).

        Use when the operation needs RGB pixels regardless of whether the
        container's semantic order is RGB or BGR (both use RGB storage).
        Converts from the current mode as needed (including HSV / LAB to RGB).
        """
        return self.value.convert(ChannelOrder.RGB.pil_mode)

    def to_hsv_image(self) -> Image.Image:
        """Pillow HSV image."""
        return self.value.convert(ChannelOrder.HSV.pil_mode)

    def to_lab_image(self) -> Image.Image:
        """Pillow LAB image."""
        return self.value.convert(ChannelOrder.LAB.pil_mode)

    def to_PIL(self) -> Image.Image:
        """
        Get the PIL image.

        Returns
        -------
        Image.Image: The PIL image.
        """
        return self.value

    def to_array(
        self,
        ch_order: ChannelOrder = ChannelOrder.BGR
        ) -> UInt8Image:
        """
        Get the array image.

        Returns
        -------
        UInt8Image: The array image.

        Raises
        ------
        ValueError:
            If the channel order is not supported.
        """
        match ch_order:
            case ChannelOrder.RGB:
                return np.asarray(self.to_rgb_image())
            case ChannelOrder.BGR:
                return np.asarray(self.to_rgb_image())[..., ::-1]
            case ChannelOrder.GRAY:
                return np.asarray(self.to_gray())
            case ChannelOrder.HSV:
                return np.asarray(self.to_hsv_image())
            case ChannelOrder.LAB:
                return np.asarray(self.to_lab_image())

    def to_binary(self, threshold: int | float) -> BinaryImage:
        """
        Convert the image into a binary image using a threshold.

        Values greater than or equal to the threshold become 1, otherwise 0.
        If the input has 3 channels, it will be converted to gray first.
        """
        gray = np.asarray(self.to_gray())
        return BinaryImage(value=(gray >= threshold))

    def to_ch_swapped_image(
        self,
        output_order: ChannelOrder
        ) -> Image.Image:
        """
        Get the channel swapped image.

        Parameters:
        ----------
        output_order: ChannelOrder
            The output channel order to get the image.
        """
        if self.channel_order == output_order:
            return self.value.copy()

        arr = self.to_array(output_order)
        match output_order:
            case ChannelOrder.RGB:
                return Image.fromarray(arr, mode=ChannelOrder.RGB.pil_mode)
            case ChannelOrder.BGR:
                return Image.fromarray(
                    arr[..., ::-1],
                    mode=ChannelOrder.RGB.pil_mode,
                )
            case ChannelOrder.GRAY:
                return Image.fromarray(arr, mode=ChannelOrder.GRAY.pil_mode)
            case ChannelOrder.HSV:
                return Image.fromarray(arr, mode=ChannelOrder.HSV.pil_mode)
            case ChannelOrder.LAB:
                return Image.fromarray(arr, mode=ChannelOrder.LAB.pil_mode)
