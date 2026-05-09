from __future__ import annotations

from PIL import Image

from ....ch_order import ChannelOrder


class PILGeometryMixin:
    """
    Mixin for geometry-related properties on PIL-backed image containers.

    Expects value (PIL.Image.Image) and channel_order (ChannelOrder) on the
    concrete class (typically from ImageContainer).

    The mixin does not subclass SupportsPILGeometry: combining that protocol
    with dataclass field declarations confuses static checkers. Declare
    SupportsPILGeometry on the concrete container class instead.
    """

    value: Image.Image
    channel_order: ChannelOrder

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Get the shape of the image.

        Returns
        -------
        tuple[int, int, int]: The shape(height, width, channels) of the image.
        """
        return (self.height, self.width, self.ch)

    @property
    def width(self) -> int:
        """
        Get the width of the image.

        Returns
        -------
        int: The width of the image.
        """
        return self.value.width

    @property
    def height(self) -> int:
        """
        Get the height of the image.

        Returns
        -------
        int: The height of the image.
        """
        return self.value.height

    @property
    def size(self) -> tuple[int, int]:
        """
        Get the size of the image.

        Returns
        -------
        tuple[int, int]: The size(width, height) of the image.
        """
        return self.value.size

    @property
    def ch(self) -> int:
        """
        Get the number of channels of the image.

        Returns
        -------
        int: The number of channels of the image.
        """
        mode = self.value.mode
        match mode:
            case 'L':
                return 1
            case 'RGB':
                return 3
            case 'RGBA':
                return 4
            case 'HSV':
                return 3
            case 'LAB':
                return 3
            case _:
                raise ValueError(f"Unsupported PIL mode: {self.value.mode}")
