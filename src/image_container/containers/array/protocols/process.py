from __future__ import annotations

from typing import Protocol

from ....ch_order import ChannelOrder
from ....types import UInt8Image

from .geometry import SupportsArrayGeometry


class SupportsArrayProcess(SupportsArrayGeometry, Protocol):
    """
    Structural typing for hosts of ArrayProcessMixin (requires conversion helpers).

    Methods
    -------
    to_array
        Converts the container payload to a numpy array in the requested channel order.
    """

    def to_array(self, ch_order: ChannelOrder = ChannelOrder.BGR) -> UInt8Image:
        ...
