from __future__ import annotations

from PIL import Image

from ....binary_image import BinaryImage
from ....ch_order import ChannelOrder
from ....dct_image import DctImage
from ....types import UInt8Image


class ArrayConvertMixin:
    """Convert the held array image.

    - PIL
    - NumPy with another channel order
    - BinaryImage
    - DctImage
    """

    value: UInt8Image
    channel_order: ChannelOrder

    def to_gray(self) -> UInt8Image:
        """
        Grayscale plane (H, W).

        Same pixel values as to_array(ChannelOrder.GRAY).
        """
        return self.to_array(ChannelOrder.GRAY)

    def to_3ch(self) -> UInt8Image:
        """
        Three-channel array (H, W, 3).

        If the container is already RGB/BGR/HSV/LAB, returns a copy of value.
        If gray, returns BGR layout (same as expanding via to_array(BGR)).
        """
        if self.channel_order == ChannelOrder.GRAY:
            return self.to_array(ChannelOrder.BGR)
        return self.value.copy()

    def to_PIL(self) -> Image.Image:
        """
        Get the PIL image.

        Returns
        -------
        Image.Image: The PIL image.
        """
        match self.channel_order:
            case ChannelOrder.BGR:
                return Image.fromarray(self.value[..., [2, 1, 0]], mode='RGB')
            case ChannelOrder.GRAY:
                return Image.fromarray(
                    self.to_gray(),
                    mode=ChannelOrder.GRAY.pil_mode,
                )
            case ChannelOrder.HSV:
                bgr: UInt8Image = self.to_array(ChannelOrder.BGR)
                return Image.fromarray(bgr[..., [2, 1, 0]], mode='RGB')
            case ChannelOrder.LAB:
                return Image.fromarray(self.value, mode=ChannelOrder.LAB.pil_mode)
            case ChannelOrder.RGB:
                return Image.fromarray(self.value, mode='RGB')

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
        if ch_order == self.channel_order:
            return self.value.copy()

        convert = ch_order.cv2_array_converter(self.channel_order)
        return convert(self.value)

    def to_binary(self, threshold: int | float) -> BinaryImage:
        """
        Convert the image into a binary image using a threshold.

        Values greater than or equal to the threshold become 1, otherwise 0.
        If the input has 3 channels, it will be converted to gray first.
        """
        gray = self.to_gray()
        return BinaryImage(value=(gray >= threshold))

    def dct(self) -> DctImage:
        """
        Discrete Cosine Transform (DCT).

        Returns
        -------
        DctImage
            DCT coefficients with shape (H, W) and float32 dtype.
        """
        return DctImage.from_array(self.value, self.channel_order)

    def to_ch_swapped_image(
        self,
        output_order: ChannelOrder = ChannelOrder.BGR
        ) -> UInt8Image:
        """
        Get the channel swapped image.

        Parameters:
        ----------
        output_order: ChannelOrder
            The output channel order to get the image.

        Returns
        -------
        UInt8Image: The channel swapped image.
        """
        if self.channel_order == output_order:
            return self.value.copy()
        return self.to_array(output_order)
