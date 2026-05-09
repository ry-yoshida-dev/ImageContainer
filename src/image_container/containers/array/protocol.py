from __future__ import annotations

import numpy as np
from typing import Protocol

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


class SupportsArrayHash(SupportsArrayProcess, Protocol):
    """
    Structural typing for hosts of ArrayHashMixin (conversion + img_hash accessors).
    """

    @property
    def phash(self) -> np.ndarray:
        ...

    @property
    def average_hash(self) -> np.ndarray:
        ...

    @property
    def block_mean_hash(self) -> np.ndarray:
        ...

    @property
    def color_moment_hash(self) -> np.ndarray:
        ...

    @property
    def marr_hildreth_hash(self) -> np.ndarray:
        ...

    @property
    def radial_variance_hash(self) -> np.ndarray:
        ...
