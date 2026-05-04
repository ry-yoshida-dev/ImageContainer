from __future__ import annotations

import numpy as np

from ....ch_order import ChannelOrder


class ArrayGeometryMixin:
    """
    Mixin for geometry-related properties on array-backed image containers.

    Expects value (numpy.ndarray) and channel_order (ChannelOrder) on the
    concrete class (typically from ImageContainer).

    The mixin does not subclass SupportsArrayGeometry: combining that protocol
    with dataclass field declarations confuses static checkers. Declare
    SupportsArrayProcess (or SupportsArrayGeometry) on the concrete container
    class instead.
    """

    value: np.ndarray
    channel_order: ChannelOrder

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Get the shape of the image.

        Returns
        -------
        tuple[int, int, int]
            The shape (height, width, channels) of the image.

        Raises
        ------
        ValueError
            If the array rank is not 2 or 3.
        """
        shape = self.value.shape
        if len(shape) == 2:
            return (shape[0], shape[1], 1)
        if len(shape) == 3:
            return (int(shape[0]), int(shape[1]), int(shape[2]))
        raise ValueError(f"Unsupported shape: {shape}")

    @property
    def width(self) -> int:
        """
        Get the width of the image.

        Returns
        -------
        int
            The width of the image.
        """
        return self.shape[1]

    @property
    def height(self) -> int:
        """
        Get the height of the image.

        Returns
        -------
        int
            The height of the image.
        """
        return self.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        """
        Get the size of the image.

        Returns
        -------
        tuple[int, int]
            The size (width, height), matching PIL Image.size.
        """
        return (self.width, self.height)

    @property
    def ch(self) -> int:
        """
        Get the number of channels of the image.

        Returns
        -------
        int
            The number of channels of the image.
        """
        match self.channel_order:
            case ChannelOrder.GRAY:
                return 1
            case _:
                return 3
