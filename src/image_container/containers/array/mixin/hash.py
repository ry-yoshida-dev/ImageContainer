"""OpenCV img_hash accessors for array-backed image containers."""

from __future__ import annotations

import cv2
from collections.abc import Callable
from typing import Protocol, cast

from image_container.ch_order import ChannelOrder
from image_container.containers.array.mixin.convert import ArrayConvertMixin
from image_container.types import ImageArray, UInt8Image


class ImgHashComputer(Protocol):
    """
    Structural type for OpenCV img_hash objects that expose compute(img).

    Methods
    -------
    compute
        Consumes a BGR image array and returns hash bytes as an ndarray.
    """

    def compute(self, img: UInt8Image) -> ImageArray:
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

    def _compute_img_hash(self, create_hasher: Callable[[], object]) -> ImageArray:
        """
        Run an img_hash algorithm on the container as BGR.

        Parameters
        ----------
        create_hasher : Callable[[], object]
            Factory that returns one OpenCV img_hash object with a compute method.

        Returns
        -------
        ImageArray
            Hash bytes; shape is typically (1, N), where N depends on the algorithm.

        Notes
        -----
        Input pixels are always obtained via to_array(ChannelOrder.BGR).
        """
        bgr: UInt8Image = self.to_array(ChannelOrder.BGR)
        hasher = cast(ImgHashComputer, create_hasher())

        hash_bytes: ImageArray = hasher.compute(bgr)
        return hash_bytes

    @property
    def phash(self) -> ImageArray:
        """
        Perceptual hash (PHash).

        Returns
        -------
        ImageArray
            Hash array with shape (1, 8) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.PHash_create)  # type: ignore[attr-defined]

    @property
    def average_hash(self) -> ImageArray:
        """
        Average hash.

        Returns
        -------
        ImageArray
            Hash array with shape (1, 8) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.AverageHash_create)  # type: ignore[attr-defined]

    @property
    def block_mean_hash(self) -> ImageArray:
        """
        Block mean hash.

        Returns
        -------
        ImageArray
            Hash array with shape (1, 32) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.BlockMeanHash_create)  # type: ignore[attr-defined]

    @property
    def color_moment_hash(self) -> ImageArray:
        """
        Color moment hash.

        Returns
        -------
        ImageArray
            Hash array with shape (1, 42) and dtype float64.
        """
        return self._compute_img_hash(cv2.img_hash.ColorMomentHash_create)  # type: ignore[attr-defined]

    @property
    def marr_hildreth_hash(self) -> ImageArray:
        """
        Marr–Hildreth hash.

        Returns
        -------
        ImageArray
            Hash array with shape (1, 72) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.MarrHildrethHash_create)  # type: ignore[attr-defined]

    @property
    def radial_variance_hash(self) -> ImageArray:
        """
        Radial variance hash.

        Returns
        -------
        ImageArray
            Hash array with shape (1, 40) and dtype uint8.
        """
        return self._compute_img_hash(cv2.img_hash.RadialVarianceHash_create)  # type: ignore[attr-defined]
