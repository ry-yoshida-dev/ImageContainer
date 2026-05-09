"""OpenCV img_hash accessors for array-backed image containers."""

from __future__ import annotations

import cv2
import numpy as np
from collections.abc import Callable
from typing import Protocol, cast

from image_container.ch_order import ChannelOrder
from image_container.containers.array.mixin.convert import ArrayConvertMixin


class ImgHashComputer(Protocol):
    """
    Structural type for OpenCV img_hash objects that expose compute(img).

    Methods
    -------
    compute
        Consumes a BGR image array and returns hash bytes as an ndarray.
    """

    def compute(self, img: np.ndarray) -> np.ndarray:
        ...


class ArrayHashMixin(ArrayConvertMixin):
    """
    OpenCV img_hash helpers on array-backed image containers.

    Hashing uses to_array(ChannelOrder.BGR) as input to each hasher.

    Notes
    -----
    Requires a build of OpenCV that includes the contrib img_hash module,
    for example opencv-contrib-python.

    Hash creation and compute calls are centralized in _compute_img_hash.
    In this module, default means calling each OpenCV *_create() with no args.
    """

    def _compute_img_hash(self, create_hasher: Callable[[], object]) -> np.ndarray:
        """
        Run an img_hash algorithm on the container as BGR.

        Parameters
        ----------
        create_hasher : Callable[[], object]
            Factory that returns one OpenCV img_hash object with a compute method.

        Returns
        -------
        np.ndarray
            Hash bytes; shape is typically (1, N), where N depends on the algorithm.

        Notes
        -----
        Input pixels are always obtained via to_array(ChannelOrder.BGR).
        """
        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        hasher = cast(ImgHashComputer, create_hasher())

        hash_bytes: np.ndarray = hasher.compute(bgr)
        return hash_bytes

    @property
    def phash(self) -> np.ndarray:
        """
        Perceptual hash (PHash).

        Returns
        -------
        np.ndarray
            Hash array with shape (1, 8) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.PHash_create)  # type: ignore[attr-defined]

    @property
    def average_hash(self) -> np.ndarray:
        """
        Average hash.

        Returns
        -------
        np.ndarray
            Hash array with shape (1, 8) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.AverageHash_create)  # type: ignore[attr-defined]

    @property
    def block_mean_hash(self) -> np.ndarray:
        """
        Block mean hash.

        Returns
        -------
        np.ndarray
            Hash array with shape (1, 32) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.BlockMeanHash_create)  # type: ignore[attr-defined]

    @property
    def color_moment_hash(self) -> np.ndarray:
        """
        Color moment hash.

        Returns
        -------
        np.ndarray
            Hash array with shape (1, 42) and dtype float64.
        """
        return self._compute_img_hash(cv2.img_hash.ColorMomentHash_create)  # type: ignore[attr-defined]

    @property
    def marr_hildreth_hash(self) -> np.ndarray:
        """
        Marr–Hildreth hash.

        Returns
        -------
        np.ndarray
            Hash array with shape (1, 72) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.MarrHildrethHash_create)  # type: ignore[attr-defined]

    @property
    def radial_variance_hash(self) -> np.ndarray:
        """
        Radial variance hash.

        Returns
        -------
        np.ndarray
            Hash array with shape (1, 40) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.RadialVarianceHash_create)  # type: ignore[attr-defined]
