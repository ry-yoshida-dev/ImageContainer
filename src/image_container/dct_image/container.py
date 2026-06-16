from __future__ import annotations

import warnings

import cv2
import numpy as np
from dataclasses import dataclass

from ..ch_order import ChannelOrder
from ..types import ImageArray
from ..utils.padding import padding_to_even

@dataclass
class DctImage:
    """
    A container class representing a 2D DCT coefficient image.

    Parameters
    ----------
    value : ImageArray
        DCT coefficients as a 2D float32 array of shape (height, width).
    source_shape : tuple[int, int] | None, default None
        Original (height, width) before odd-dimension padding in from_array.
        When set, idct crops the restored image back to this shape.
    """

    value: ImageArray
    source_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """
        Post-initialization validation.

        Raises
        ------
        ValueError
            If the array is not 2D or not float32.
        """
        if self.value.ndim != 2:
            raise ValueError(f"DCT image must have 2 dimensions, got {self.value.ndim}")
        if self.value.dtype != np.float32:
            raise ValueError(
                f"DCT image must have float32 dtype, got {self.value.dtype}"
            )

    @classmethod
    def from_array(
        cls,
        array: ImageArray,
        channel_order: ChannelOrder,
    ) -> DctImage:
        """
        Create a DctImage from a numpy array image.

        The input is converted to grayscale before applying the forward DCT.

        Parameters
        ----------
        array : ImageArray
            Input image array in the given channel order.
        channel_order : ChannelOrder
            Channel layout of the input array.

        Returns
        -------
        DctImage
            DCT coefficients with shape (H, W) and float32 dtype.

        Warns
        -----
        UserWarning
            When height or width is odd, the grayscale image is padded by one
            pixel on the bottom and/or right before DCT.
        """
        to_gray = ChannelOrder.GRAY.cv2_array_converter(channel_order)
        gray = to_gray(array)
        gray, source_shape = padding_to_even(gray)
        if source_shape is not None:
            warnings.warn(
                (
                    f"Grayscale shape {source_shape} has odd dimension(s); "
                    f"padded to {gray.shape} for DCT."
                ),
                UserWarning,
                stacklevel=2,
            )
        return cls(value=cv2.dct(gray.astype(np.float32)), source_shape=source_shape)

    def idct(self, is_uint8_cast_enabled: bool = True) -> ImageArray:
        """
        Inverse Discrete Cosine Transform (IDCT).

        Parameters
        ----------
        is_uint8_cast_enabled : bool, default True
            If True, clips the result to [0, 255] and converts to uint8.

        Returns
        -------
        ImageArray
            Restored grayscale image array with shape (H, W).
        """
        restored = cv2.idct(self.value)
        if self.source_shape is not None:
            height, width = self.source_shape
            restored = restored[:height, :width]
        if is_uint8_cast_enabled:
            return np.clip(restored, 0, 255).astype(np.uint8)
        return restored

    def laplacian(self, *, is_uint8_output: bool = True) -> ImageArray:
        """
        Frequency-domain Laplacian edge emphasis on DCT coefficients.

        Multiplies coefficients by the mask (u^2 + v^2), applies IDCT, crops to
        source_shape when odd-dimension padding was used, then optionally
        normalizes |response| to uint8 for visualization.

        Parameters
        ----------
        is_uint8_output : bool, default True
            If True, take absolute value and min-max normalize to [0, 255] as
            uint8. If False, return the float32 spatial response without scaling.

        Returns
        -------
        ImageArray
            Edge-emphasized grayscale image with shape (H, W); uint8 when
            is_uint8_output is True, otherwise float32. When uint8 output is
            requested and |response| is constant (including all zeros), returns
            a zero-filled uint8 image without min-max scaling.
        """
        height, width = self.value.shape
        v_indices = np.arange(height, dtype=np.float32)[:, np.newaxis]
        u_indices = np.arange(width, dtype=np.float32)[np.newaxis, :]
        laplacian_mask = (u_indices**2) + (v_indices**2)
        filtered = self.value * laplacian_mask
        restored = cv2.idct(filtered)
        if self.source_shape is not None:
            crop_height, crop_width = self.source_shape
            restored = restored[:crop_height, :crop_width]
        if is_uint8_output:
            magnitude = np.abs(restored)
            peak = float(np.max(magnitude))
            trough = float(np.min(magnitude))
            if peak == trough:
                return np.zeros(magnitude.shape, dtype=np.uint8)
            destination = np.empty_like(magnitude, dtype=np.float32)
            normalized = cv2.normalize(
                magnitude, destination, 0, 255, cv2.NORM_MINMAX
            )
            return normalized.astype(np.uint8)
        return restored.astype(np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the DCT image.

        Returns
        -------
        tuple[int, int]
            The shape (height, width) of the DCT image.
        """
        return self.value.shape

    @property
    def width(self) -> int:
        """
        Get the width of the DCT image.

        Returns
        -------
        int
            The width of the DCT image.
        """
        return self.shape[1]

    @property
    def height(self) -> int:
        """
        Get the height of the DCT image.

        Returns
        -------
        int
            The height of the DCT image.
        """
        return self.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        """
        Get the size of the DCT image.

        Returns
        -------
        tuple[int, int]
            The size (width, height) of the DCT image.
        """
        return (self.width, self.height)
