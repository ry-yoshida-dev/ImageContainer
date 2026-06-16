from __future__ import annotations

import cv2

from collections.abc import Callable
from enum import Enum
from functools import reduce

from .types import UInt8Image


class ChannelOrder(Enum):
    """
    Channel order for images.

    Attributes
    ----------
    RGB: RGB channel order.
    BGR: BGR channel order.
    GRAY: Gray channel order.
    HSV: HSV color space.
    LAB: CIELAB in OpenCV layout (L, a, b) as uint8; typically produced via
        BGR2LAB / RGB2LAB.

    Notes
    -----
    pil_mode is the Pillow Image.mode used to hold pixels for this order.
    BGR has no distinct Pillow mode; triplets are stored as RGB.
    """

    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    LAB = "lab"

    @property
    def pil_mode(self) -> str:
        """Pillow Image.mode string for this channel order."""
        match self:
            case ChannelOrder.RGB | ChannelOrder.BGR:
                return "RGB"
            case ChannelOrder.GRAY:
                return "L"
            case ChannelOrder.HSV:
                return "HSV"
            case ChannelOrder.LAB:
                return "LAB"

    @property
    def is_3ch(self) -> bool:
        """True if the channel order is 3-channel."""
        return self in (
            ChannelOrder.RGB,
            ChannelOrder.BGR,
            ChannelOrder.HSV,
            ChannelOrder.LAB,
        )

    @property
    def is_1ch(self) -> bool:
        """True if the channel order is 1-channel."""
        return self == ChannelOrder.GRAY

    def cv2_conversion_flags(self, convert_from: ChannelOrder) -> tuple[int, ...]:
        """
        OpenCV cv2.COLOR_* values to turn convert_from layout into this layout.

        Applied in order with cv2.cvtColor for each code. Empty tuple means identity.
        """
        if convert_from == self:
            return ()
        match (convert_from, self):
            case (ChannelOrder.BGR, ChannelOrder.GRAY):
                return (cv2.COLOR_BGR2GRAY,)
            case (ChannelOrder.RGB, ChannelOrder.GRAY):
                return (cv2.COLOR_RGB2GRAY,)
            case (ChannelOrder.HSV, ChannelOrder.GRAY):
                return (cv2.COLOR_HSV2BGR, cv2.COLOR_BGR2GRAY)
            case (ChannelOrder.GRAY, ChannelOrder.BGR):
                return (cv2.COLOR_GRAY2BGR,)
            case (ChannelOrder.GRAY, ChannelOrder.RGB):
                return (cv2.COLOR_GRAY2RGB,)
            case (ChannelOrder.GRAY, ChannelOrder.HSV):
                return (cv2.COLOR_GRAY2BGR, cv2.COLOR_BGR2HSV)
            case (ChannelOrder.BGR, ChannelOrder.RGB):
                return (cv2.COLOR_BGR2RGB,)
            case (ChannelOrder.RGB, ChannelOrder.BGR):
                return (cv2.COLOR_RGB2BGR,)
            case (ChannelOrder.RGB, ChannelOrder.HSV):
                return (cv2.COLOR_RGB2HSV,)
            case (ChannelOrder.HSV, ChannelOrder.RGB):
                return (cv2.COLOR_HSV2RGB,)
            case (ChannelOrder.BGR, ChannelOrder.HSV):
                return (cv2.COLOR_BGR2HSV,)
            case (ChannelOrder.HSV, ChannelOrder.BGR):
                return (cv2.COLOR_HSV2BGR,)
            case (ChannelOrder.BGR, ChannelOrder.LAB):
                return (cv2.COLOR_BGR2LAB,)
            case (ChannelOrder.RGB, ChannelOrder.LAB):
                return (cv2.COLOR_RGB2LAB,)
            case (ChannelOrder.GRAY, ChannelOrder.LAB):
                return (cv2.COLOR_GRAY2BGR, cv2.COLOR_BGR2LAB)
            case (ChannelOrder.HSV, ChannelOrder.LAB):
                return (cv2.COLOR_HSV2BGR, cv2.COLOR_BGR2LAB)
            case (ChannelOrder.LAB, ChannelOrder.BGR):
                return (cv2.COLOR_LAB2BGR,)
            case (ChannelOrder.LAB, ChannelOrder.RGB):
                return (cv2.COLOR_LAB2RGB,)
            case (ChannelOrder.LAB, ChannelOrder.GRAY):
                return (cv2.COLOR_LAB2BGR, cv2.COLOR_BGR2GRAY)
            case (ChannelOrder.LAB, ChannelOrder.HSV):
                return (cv2.COLOR_LAB2BGR, cv2.COLOR_BGR2HSV)
            case _:
                raise ValueError(
                    f"No OpenCV conversion path from {convert_from!r} to {self!r}"
                )

    def cv2_array_converter(
        self,
        convert_from: ChannelOrder,
    ) -> Callable[[UInt8Image], UInt8Image]:
        """
        Callable that maps an array from convert_from layout to this layout.

        Uses cv2.cvtColor chained according to cv2_conversion_flags.
        """
        codes = self.cv2_conversion_flags(convert_from)
        if not codes:
            return lambda img: img.copy()
        return lambda img: reduce(
            lambda acc, code: cv2.cvtColor(acc, code),
            codes,
            img,
        )
