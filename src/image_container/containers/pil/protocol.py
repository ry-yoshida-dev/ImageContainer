from __future__ import annotations

from typing import Protocol

from PIL import Image

from ...ch_order import ChannelOrder


class SupportsPILGeometry(Protocol):
    """
    Structural typing for PIL-backed containers used by geometry helpers.

    Attributes
    ----------
    value: Image.Image
        The PIL image payload.
    channel_order: ChannelOrder
        The channel order of the image.
    """

    value: Image.Image
    channel_order: ChannelOrder
