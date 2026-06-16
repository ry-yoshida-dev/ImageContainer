from __future__ import annotations

from typing import Protocol

import numpy as np

from ....types import ImageArray
from .process import SupportsArrayProcess


class SupportsArrayHash(SupportsArrayProcess, Protocol):
    """
    Structural typing for hosts of ArrayHashMixin (conversion and img_hash accessors).

    Attributes
    ----------
    phash : ImageArray
        Perceptual hash values.
    average_hash : ImageArray
        Average hash values.
    block_mean_hash : ImageArray
        Block mean hash values.
    color_moment_hash : ImageArray
        Color moment hash values.
    marr_hildreth_hash : ImageArray
        Marr-Hildreth hash values.
    radial_variance_hash : ImageArray
        Radial variance hash values.
    """

    @property
    def phash(self) -> ImageArray:
        ...

    @property
    def average_hash(self) -> ImageArray:
        ...

    @property
    def block_mean_hash(self) -> ImageArray:
        ...

    @property
    def color_moment_hash(self) -> ImageArray:
        ...

    @property
    def marr_hildreth_hash(self) -> ImageArray:
        ...

    @property
    def radial_variance_hash(self) -> ImageArray:
        ...
