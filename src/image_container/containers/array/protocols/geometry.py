from __future__ import annotations

import numpy as np
from typing import Protocol

from ....ch_order import ChannelOrder


class SupportsArrayGeometry(Protocol):
    """
    Structural typing for array-backed containers used by geometry helpers.

    Attributes
    ----------
    value : np.ndarray
        The numpy array payload.
    channel_order : ChannelOrder
        The channel order of the image.
    """

    value: np.ndarray
    channel_order: ChannelOrder
