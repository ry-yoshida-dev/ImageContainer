from __future__ import annotations

import warnings

import cv2
import numpy as np
from dataclasses import dataclass

from ...ch_order import ChannelOrder
from ...types import BinaryArray, ImageArray, UInt8Image
from ...utils.padding import padding_to_even
from ..base import DctImageBase


@dataclass(kw_only=True)
class NonBlockedDctImage(DctImageBase):
    """
    A container for global (non-blocked) 2D DCT coefficient images.

    Parameters
    ----------
    value : ImageArray
        DCT coefficients as a 2D float32 array of shape (height, width).
    source_shape : tuple[int, int] | None, default None
        Original (height, width) before odd-dimension padding in from_array.
        When set, idct crops the restored image back to this shape.
    """

    def _validate_transform_specific(self) -> None:
        """Non-blocked DCT has no additional validation."""

    def _frequency_radius_squared(self) -> ImageArray:
        """
        Build the global squared radial frequency index (u^2 + v^2).

        Returns
        -------
        ImageArray
            Mask array of shape (height, width) with float32 dtype.
        """
        height, width = self.value.shape
        v_indices = np.arange(height, dtype=np.float32)[:, np.newaxis]
        u_indices = np.arange(width, dtype=np.float32)[np.newaxis, :]
        return (u_indices**2) + (v_indices**2)

    def _apply_inverse_dct(self, coefficients: ImageArray) -> ImageArray:
        """
        Apply global inverse DCT.

        Parameters
        ----------
        coefficients : ImageArray
            DCT coefficients of shape (height, width).

        Returns
        -------
        ImageArray
            Restored grayscale float32 array with the same shape as coefficients.
        """
        return cv2.idct(coefficients)

    def _clear_dc_components(self, mask: BinaryArray) -> None:
        """
        Clear the global DC coefficient at (0, 0).

        Parameters
        ----------
        mask : BinaryArray
            Boolean mask array modified in place.
        """
        mask[0, 0] = False

    def _with_filtered_coefficients(self, filtered: ImageArray) -> NonBlockedDctImage:
        """
        Build a NonBlockedDctImage from filtered coefficients.

        Parameters
        ----------
        filtered : ImageArray
            Filtered coefficient array of shape (height, width).

        Returns
        -------
        NonBlockedDctImage
            New container with the same source_shape.
        """
        return NonBlockedDctImage(value=filtered, source_shape=self.source_shape)

    @classmethod
    def from_array(
        cls,
        array: UInt8Image,
        channel_order: ChannelOrder,
    ) -> NonBlockedDctImage:
        """
        Create a NonBlockedDctImage from a global forward DCT.

        The input is converted to grayscale before applying the forward DCT.

        Parameters
        ----------
        array : UInt8Image
            Input image array in the given channel order.
        channel_order : ChannelOrder
            Channel layout of the input array.

        Returns
        -------
        NonBlockedDctImage
            Global DCT coefficients with shape (H, W) and float32 dtype.

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
        return cls(
            value=cv2.dct(gray.astype(np.float32)),
            source_shape=source_shape,
        )
