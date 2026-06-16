from __future__ import annotations

from typing import Protocol

from ....ch_order import ChannelOrder
from ....types import UInt8Image


class SupportsArrayGeometry(Protocol):
    """
    Structural typing for array-backed containers used by geometry helpers.

    Attributes
    ----------
    value : UInt8Image
        The numpy array payload.
    channel_order : ChannelOrder
        The channel order of the image.
    """

    value: UInt8Image
    channel_order: ChannelOrder
