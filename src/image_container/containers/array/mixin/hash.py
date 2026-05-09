from __future__ import annotations

import cv2
import numpy as np

from typing import Protocol, cast

from ....ch_order import ChannelOrder
from .convert import ArrayConvertMixin


class _ImgHashComputer(Protocol):
    """OpenCV img_hash hasher with ``compute``."""

    def compute(self, img: np.ndarray) -> np.ndarray:
        ...


class ArrayHashMixin(ArrayConvertMixin):
    """
    OpenCV img_hash helpers on array-backed image containers.

    Hashing uses ``to_array(ChannelOrder.BGR)`` input for ``compute``.
    Requires ``opencv-contrib-python`` (or equivalent) with the img_hash module.
    """

    @property
    def phash(self) -> np.ndarray:
        """PHash bytes via ``cv2.img_hash.PHash_create()``."""
        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        hasher: _ImgHashComputer = cast(
            _ImgHashComputer,
            cv2.img_hash.PHash_create(),  # type: ignore[attr-defined]
        )
        hash_bytes: np.ndarray = hasher.compute(bgr)
        return hash_bytes

    @property
    def average_hash(self) -> np.ndarray:
        """Average hash bytes via ``cv2.img_hash.AverageHash_create()``."""
        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        hasher: _ImgHashComputer = cast(
            _ImgHashComputer,
            cv2.img_hash.AverageHash_create(),  # type: ignore[attr-defined]
        )
        hash_bytes: np.ndarray = hasher.compute(bgr)
        return hash_bytes

    @property
    def block_mean_hash(self) -> np.ndarray:
        """Block mean hash bytes via ``cv2.img_hash.BlockMeanHash_create()``."""
        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        hasher: _ImgHashComputer = cast(
            _ImgHashComputer,
            cv2.img_hash.BlockMeanHash_create(),  # type: ignore[attr-defined]
        )
        hash_bytes: np.ndarray = hasher.compute(bgr)
        return hash_bytes

    @property
    def color_moment_hash(self) -> np.ndarray:
        """Color moment hash bytes via ``cv2.img_hash.ColorMomentHash_create()``."""
        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        hasher: _ImgHashComputer = cast(
            _ImgHashComputer,
            cv2.img_hash.ColorMomentHash_create(),  # type: ignore[attr-defined]
        )
        hash_bytes: np.ndarray = hasher.compute(bgr)
        return hash_bytes

    @property
    def marr_hildreth_hash(self) -> np.ndarray:
        """Marr–Hildreth hash bytes via ``cv2.img_hash.MarrHildrethHash_create()``."""
        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        hasher: _ImgHashComputer = cast(
            _ImgHashComputer,
            cv2.img_hash.MarrHildrethHash_create(),  # type: ignore[attr-defined]
        )
        hash_bytes: np.ndarray = hasher.compute(bgr)
        return hash_bytes

    @property
    def radial_variance_hash(self) -> np.ndarray:
        """Radial variance hash bytes via ``cv2.img_hash.RadialVarianceHash_create()``."""
        bgr: np.ndarray = self.to_array(ChannelOrder.BGR)
        hasher: _ImgHashComputer = cast(
            _ImgHashComputer,
            cv2.img_hash.RadialVarianceHash_create(),  # type: ignore[attr-defined]
        )
        hash_bytes: np.ndarray = hasher.compute(bgr)
        return hash_bytes
