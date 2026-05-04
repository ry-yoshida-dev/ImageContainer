from enum import Enum


class ChannelOrder(Enum):
    """
    Channel order for images.

    Attributes
    ----------
    RGB: RGB channel order.
    BGR: BGR channel order.
    GRAY: Gray channel order.
    HSV: HSV color space.

    Notes
    -----
    pil_mode is the Pillow Image.mode used to hold pixels for this order.
    BGR has no distinct Pillow mode; triplets are stored as RGB.
    """
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"

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