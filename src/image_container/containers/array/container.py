from __future__ import annotations

import os

import cv2
import numpy as np

from ...ch_order import ChannelOrder
from ...container import ImageContainer
from ...format import ImageFormat
from .mixin import ArrayGeometryMixin, ArrayProcessMixin, ArrayStatsMixin
from .mixin.hash import ArrayHashMixin
from .protocol import SupportsArrayHash


class ArrayImageContainer(
    ArrayGeometryMixin,
    ArrayHashMixin,
    ArrayStatsMixin,
    ArrayProcessMixin,
    ImageContainer[np.ndarray],
    SupportsArrayHash,
):
    """
    Container class for numpy arrays.

    Attributes:
    ----------
    value: np.ndarray
        The numpy array.
    channel_order: ChannelOrder
        The channel order of the image.
    """

    def __post_init__(self) -> None:
        """
        Post initialize the array image container.
        """
        super().__post_init__()
        self.value.setflags(write=False)

    @property
    def format(self) -> ImageFormat:
        """The image format (ARRAY)."""
        return ImageFormat.ARRAY

    def _validate_image(self) -> None:
        """
        Validate the image.

        Parameters:
        ----------
        image: np.ndarray
            The image to validate.
        channel_order: ChannelOrder
            The channel order of the image.

        Raises
        ------
        ValueError:
            If the image is not a 3D numpy array.
            If the image is not a 2D numpy array.
            If the image has the wrong number of channels.
        """
        if self.channel_order.is_3ch:
            if self.value.ndim != 3:
                raise ValueError(f"Image must have 3 dimensions. Got {self.value.ndim}")
            if self.value.shape[2] != 3:
                raise ValueError(f"Image must have 3 channels. Got {self.value.shape[2]}")
        if self.channel_order.is_1ch:
            if self.value.ndim != 2:
                raise ValueError(f"Image must have 2 dimensions. Got {self.value.ndim}")

    def save(self, save_path: str) -> None:
        """
        Save with cv2.imwrite.

        Parameters
        ----------
        save_path: str
            Output path.
        """
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if self.channel_order.is_1ch:
            to_write = self.value
        else:
            to_write = self.to_array(ChannelOrder.BGR)
        if not cv2.imwrite(save_path, to_write):
            raise OSError(f"Failed to write image to {save_path}")

    def crop(
        self,
        crop_slice: tuple[slice, slice]
        ) -> ArrayImageContainer:
        """
        Crop the image.

        Parameters:
        ----------
        crop_slice: tuple[slice, slice]
            The slice to crop the image(y_slice, x_slice).
            example: (slice(100, 200), slice(300, 400))

        Returns:
        ----------
        ArrayImageContainer: The cropped image container.
        """
        return ArrayImageContainer(
            value=self.value[crop_slice],
            channel_order=self.channel_order
            )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, width={self.width}, height={self.height}, channel_order={self.channel_order})"

    @classmethod
    def from_path(
        cls,
        image_path: str,
        ) -> ArrayImageContainer:
        """
        Create an array image container from an image path.

        Parameters:
        ----------
        image_path: str
            The path to the image.

        Returns:
        ----------
        ArrayImageContainer: The array image container.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        channel_order = ChannelOrder.BGR
        return cls(
            value=image,
            channel_order=channel_order
            )
