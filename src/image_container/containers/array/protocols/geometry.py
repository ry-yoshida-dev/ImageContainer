from __future__ import annotations

import numpy as np
from typing import Protocol

from ....ch_order import ChannelOrder
from ....types import ImageArray


class SupportsArrayGeometry(Protocol):
    """
    Structural typing for array-backed containers used by geometry helpers.

    Attributes
    ----------
    value : ImageArray
        The numpy array payload.
    channel_order : ChannelOrder
        The channel order of the image.
    """

    value: ImageArray
    channel_order: ChannelOrder
