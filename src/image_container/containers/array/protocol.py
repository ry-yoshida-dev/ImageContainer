from __future__ import annotations

from typing import Protocol

import numpy as np

from ...ch_order import ChannelOrder


class SupportsArrayGeometry(Protocol):
    """
    Structural typing for array-backed containers used by geometry helpers.

    Attributes
    ----------
    value: np.ndarray
        The numpy array payload.
    channel_order: ChannelOrder
        The channel order of the image.
    """

    value: np.ndarray
    channel_order: ChannelOrder


class SupportsArrayProcess(SupportsArrayGeometry, Protocol):
    """
    Structural typing for hosts of ArrayProcessMixin (requires conversion helpers).
    """

    def to_array(self, ch_order: ChannelOrder = ChannelOrder.BGR) -> np.ndarray:
        ...
