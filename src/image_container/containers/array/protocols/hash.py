from __future__ import annotations

from typing import Protocol

import numpy as np

from .process import SupportsArrayProcess


class SupportsArrayHash(SupportsArrayProcess, Protocol):
    """
    Structural typing for hosts of ArrayHashMixin (conversion and img_hash accessors).

    Attributes
    ----------
    phash : np.ndarray
        Perceptual hash values.
    average_hash : np.ndarray
        Average hash values.
    block_mean_hash : np.ndarray
        Block mean hash values.
    color_moment_hash : np.ndarray
        Color moment hash values.
    marr_hildreth_hash : np.ndarray
        Marr-Hildreth hash values.
    radial_variance_hash : np.ndarray
        Radial variance hash values.
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
