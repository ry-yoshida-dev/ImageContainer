from __future__ import annotations

import numpy as np
from PIL import Image

from ....binary_image import BinaryImage
from ....ch_order import ChannelOrder


class PILConvertMixin:
    """Convert the held PIL image.

    - NumPy
    - BinaryImage
    - PIL with reordered channels
    """

    value: Image.Image
    channel_order: ChannelOrder

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
        ) -> np.ndarray:
        """
        Get the array image.

        Returns
        -------
        np.ndarray: The array image.

        Raises
        ------
        ValueError:
            If the channel order is not supported.
        """
        match ch_order:
            case ChannelOrder.RGB:
                return np.array(self.value.convert(ChannelOrder.RGB.pil_mode))
            case ChannelOrder.BGR:
                return np.array(self.value.convert(ChannelOrder.RGB.pil_mode))[..., ::-1]
            case ChannelOrder.GRAY:
                return np.array(self.value.convert(ChannelOrder.GRAY.pil_mode))
            case ChannelOrder.HSV:
                return np.array(self.value.convert(ChannelOrder.HSV.pil_mode))

    def to_binary(self, threshold: int | float) -> BinaryImage:
        """
        Convert the image into a binary image using a threshold.

        Values greater than or equal to the threshold become 1, otherwise 0.
        If the input has 3 channels, it will be converted to gray first.
        """
        gray = self.to_array(ChannelOrder.GRAY)  # (H, W)
        binary01 = (gray >= threshold).astype(np.uint8)  # 0/1
        return BinaryImage(value=binary01)

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
            return self.value

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
