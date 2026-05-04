from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ....binary_image import BinaryImage
from ....ch_order import ChannelOrder


class ArrayConvertMixin:
    """Convert the held array image.

    - PIL
    - NumPy with another channel order
    - BinaryImage
    """

    value: np.ndarray
    channel_order: ChannelOrder

    def to_PIL(self) -> Image.Image:
        """
        Get the PIL image.

        Returns
        -------
        Image.Image: The PIL image.
        """
        if self.channel_order == ChannelOrder.BGR:
            return Image.fromarray(self.value[..., [2, 1, 0]], mode='RGB')
        elif self.channel_order == ChannelOrder.GRAY:
            return Image.fromarray(self.value, mode='L')
        elif self.channel_order == ChannelOrder.HSV:
            rgb = cv2.cvtColor(self.value, cv2.COLOR_HSV2RGB)
            return Image.fromarray(rgb, mode='RGB')
        else:
            return Image.fromarray(self.value, mode='RGB')

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
        if ch_order == self.channel_order:
            return self.value.copy()

        if ch_order == ChannelOrder.GRAY:
            match self.channel_order:
                case ChannelOrder.BGR:
                    return cv2.cvtColor(self.value, cv2.COLOR_BGR2GRAY)
                case ChannelOrder.RGB:
                    return cv2.cvtColor(self.value, cv2.COLOR_RGB2GRAY)
                case ChannelOrder.HSV:
                    return cv2.cvtColor(
                        cv2.cvtColor(self.value, cv2.COLOR_HSV2BGR),
                        cv2.COLOR_BGR2GRAY,
                    )
                case ChannelOrder.GRAY:
                    return self.value.copy()

        if self.channel_order == ChannelOrder.GRAY:
            match ch_order:
                case ChannelOrder.BGR:
                    return cv2.cvtColor(self.value, cv2.COLOR_GRAY2BGR)
                case ChannelOrder.RGB:
                    return cv2.cvtColor(self.value, cv2.COLOR_GRAY2RGB)
                case ChannelOrder.HSV:
                    return cv2.cvtColor(
                        cv2.cvtColor(self.value, cv2.COLOR_GRAY2BGR),
                        cv2.COLOR_BGR2HSV,
                    )

        if ch_order in (ChannelOrder.BGR, ChannelOrder.RGB) and self.channel_order in (
            ChannelOrder.BGR,
            ChannelOrder.RGB,
        ):
            return self.value[..., ::-1].copy()

        match (self.channel_order, ch_order):
            case (ChannelOrder.RGB, ChannelOrder.HSV):
                return cv2.cvtColor(self.value, cv2.COLOR_RGB2HSV)
            case (ChannelOrder.HSV, ChannelOrder.RGB):
                return cv2.cvtColor(self.value, cv2.COLOR_HSV2RGB)
            case (ChannelOrder.BGR, ChannelOrder.HSV):
                return cv2.cvtColor(self.value, cv2.COLOR_BGR2HSV)
            case (ChannelOrder.HSV, ChannelOrder.BGR):
                return cv2.cvtColor(self.value, cv2.COLOR_HSV2BGR)
            case _:
                raise ValueError(f"Unsupported channel order: {ch_order}")

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
        output_order: ChannelOrder = ChannelOrder.BGR
        ) -> np.ndarray:
        """
        Get the channel swapped image.

        Parameters:
        ----------
        output_order: ChannelOrder
            The output channel order to get the image.

        Returns
        -------
        np.ndarray: The channel swapped image.
        """
        if self.channel_order == output_order:
            return self.value
        return self.to_array(output_order)
