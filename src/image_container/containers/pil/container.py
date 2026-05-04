from __future__ import annotations

import os

from PIL import Image

from ...ch_order import ChannelOrder
from ...container import ImageContainer
from ...format import ImageFormat
from .mixin.geometry import PILGeometryMixin
from .mixin.stats import PILStatsMixin
from .protocol import SupportsPILGeometry


class PILImageContainer(
    PILGeometryMixin,
    PILStatsMixin,
    ImageContainer[Image.Image],
    SupportsPILGeometry,
):
    """
    Container class for PIL images.

    Attributes:
    ----------
    value: Image.Image
        The PIL image.
    channel_order: ChannelOrder
        The channel order of the image.
    """

    def _validate_image(self) -> None:
        expected = self.channel_order.pil_mode
        if self.value.mode != expected:
            raise ValueError(
                f"channel_order={self.channel_order} expects image.mode={expected!r}, "
                f"got {self.value.mode!r}"
            )

    @property
    def format(self) -> ImageFormat:
        """The image format (PIL)."""
        return ImageFormat.PIL

    def save(self, save_path: str) -> None:
        """
        Save the image to a path.

        Parameters
        ----------
        save_path: str
            The path to save the image.
        """
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.value.save(save_path)

    def crop(
        self,
        crop_slice: tuple[slice, slice]
        ) -> PILImageContainer:
        """
        Crop the image.

        Parameters:
        ----------
        crop_slice: tuple[slice, slice]
            The slice to crop the image(y_slice, x_slice).
            example: [100:200, 300:400]

        Returns:
        ----------
        PILImageContainer: The cropped image container.
        """
        y_slice, x_slice = crop_slice
        left = x_slice.start or 0
        upper = y_slice.start or 0
        right = x_slice.stop or self.value.width
        lower = y_slice.stop or self.value.height

        cropped_image = self.value.crop((left, upper, right, lower))
        return PILImageContainer(
            value=cropped_image,
            channel_order=self.channel_order
            )

    @classmethod
    def from_path(
        cls,
        image_path: str,
        ) -> PILImageContainer:
        """
        Create a PIL image container from an image path.

        Parameters:
        ----------
        image_path: str
            The path to the image.

        Returns:
        ----------
        PILImageContainer: The PIL image container.
        """
        image = Image.open(image_path)
        channel_order = ChannelOrder.RGB
        return cls(
            value=image,
            channel_order=channel_order
            )
