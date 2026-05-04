from __future__ import annotations
import numpy as np
import torch
from PIL import Image

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Self

from .ch_order import ChannelOrder
from .binary_image import BinaryImage
from .format import ImageFormats

T = TypeVar("T", np.ndarray, Image.Image, torch.Tensor)

@dataclass(frozen=True)
class ImageContainer(ABC, Generic[T]):
    """
    Base class for image containers.

    Attributes
    ----------
    value: T
        The image data (np.ndarray, Image.Image, etc.).
    channel_order: ChannelOrder
        The channel order of the image.
    """
    value: T
    channel_order: ChannelOrder

    def __post_init__(self) -> None:
        """
        Post initialize the image container.
        """
        self._validate_image()

    @abstractmethod
    def _validate_image(self) -> None:
        """
        Validate the image.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]:
        """
        Get the shape of the image.

        Returns
        -------
        tuple[int, int, int]: The shape(height, width, channels) of the image.
        """

    @property
    @abstractmethod
    def width(self) -> int:
        """
        Get the width of the image.

        Returns
        -------
        int: The width of the image.
        """

    @property
    @abstractmethod
    def height(self) -> int:
        """
        Get the height of the image.

        Returns
        -------
        int: The height of the image.
        """

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """
        Get the size(width, height) of the image.

        Returns
        -------
        tuple[int, int]: The size(width, height) of the image.
        """

    @property
    @abstractmethod
    def ch(self) -> int:
        """
        Get the number of channels of the image.

        Returns
        -------
        int: The number of channels of the image.
        """

    @abstractmethod
    def crop(
        self, 
        crop_slice: tuple[slice, slice]
        ) -> Self:
        """
        Crop the image.

        Parameters:
        ----------
        crop_slice: tuple[slice, slice]
            The slice to crop the image(y_slice, x_slice).
            example: [100:200, 300:400]

        Returns:
        ----------
        Self: The cropped image container.
        """

    @abstractmethod
    def to_ch_swapped_image(
        self,
        output_order: ChannelOrder
        ) -> ImageFormats:
        """
        Get the image with the channel order swapped.

        Returns
        -------
        ImageFormats: The image with the channel order swapped.
        """

    @abstractmethod
    def to_PIL(self) -> Image.Image:
        """
        Get the PIL image(RGB ordered).

        Returns
        -------
        Image.Image: The PIL image.
        """

    @abstractmethod
    def to_array(
        self,
        ch_order: ChannelOrder = ChannelOrder.BGR
        ) -> np.ndarray:
        """
        Get the image with the array format in the specified channel order.

        Returns
        -------
        np.ndarray: The array image.
        """

    @abstractmethod
    def to_binary(self, threshold: int | float) -> BinaryImage:
        """
        Convert the image into a binary image using a threshold.

        Values greater than or equal to the threshold become 1, otherwise 0.
        If the input has 3 channels, it will be converted to gray first.
        """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, width={self.width}, height={self.height})"

    @classmethod
    @abstractmethod
    def from_path(
        cls,
        image_path: str
        ) -> Self:
        """
        Create an image container from an image path.

        Parameters:
        ----------
        image_path: str
            The path to the image.

        Returns:
        ----------
        Self: The image container.
        """

    @classmethod
    def register(
        cls,
        image: ImageFormats,
        channel_order: ChannelOrder
        ) -> ImageContainer[np.ndarray] | ImageContainer[Image.Image]:
        """
        Register the image container.

        Parameters:
        ----------
        image: ImageFormats
            The image to register.
        channel_order: ChannelOrder
            The channel order of the image.

        Returns:
        ----------
        ImageContainer[np.ndarray] | ImageContainer[Image.Image]:
            The registered image container.
        """
        if isinstance(image, np.ndarray):
            from .containers.array import ArrayImageContainer
            return ArrayImageContainer(
                value=image,
                channel_order=channel_order
                )
        elif isinstance(image, Image.Image):
            from .containers.pil import PILImageContainer
            return PILImageContainer(
                value=image,
                channel_order=channel_order
                )
        raise ValueError(f"Unsupported image type: {type(image)}")

    @abstractmethod
    def save(
        self,
        save_path: str
        ) -> None:
        """
        Save the image to a path.
        
        Parameters:
        ----------
        save_path: str
            The path to save the image.
        """
