from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import cv2
import numpy as np
from dataclasses import dataclass

from ..types import BinaryArray, ImageArray


@dataclass(kw_only=True)
class DctImageBase(ABC):
    """
    Shared container fields and operations for DCT coefficient images.

    Parameters
    ----------
    value : ImageArray
        DCT coefficients as a 2D float32 array of shape (height, width).
    source_shape : tuple[int, int] | None, default None
        Original (height, width) before spatial padding in factory methods.
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
        self._validate_transform_specific()

    @abstractmethod
    def _validate_transform_specific(self) -> None:
        """Validate transform-specific constraints after common checks."""

    @abstractmethod
    def _frequency_radius_squared(self) -> ImageArray:
        """
        Build the per-coefficient squared radial frequency index (u^2 + v^2).

        Returns
        -------
        ImageArray
            Mask array of shape (height, width) with float32 dtype.
        """

    @abstractmethod
    def _apply_inverse_dct(self, coefficients: ImageArray) -> ImageArray:
        """
        Apply the inverse DCT matching this container's transform mode.

        Parameters
        ----------
        coefficients : ImageArray
            DCT coefficients of shape (height, width).

        Returns
        -------
        ImageArray
            Restored grayscale float32 array with the same shape as coefficients.
        """

    @abstractmethod
    def _clear_dc_components(self, mask: BinaryArray) -> None:
        """
        Zero out DC coefficient positions in a boolean frequency mask.

        Parameters
        ----------
        mask : BinaryArray
            Boolean mask array modified in place.
        """

    @abstractmethod
    def _with_filtered_coefficients(self, filtered: ImageArray) -> Self:
        """
        Build a new container holding filtered DCT coefficients.

        Parameters
        ----------
        filtered : ImageArray
            Filtered coefficient array of shape (height, width).

        Returns
        -------
        Self
            New container preserving transform-specific metadata.
        """

    def high_pass_filter(self, threshold: float) -> Self:
        """
        Zero out low-frequency DCT coefficients below a radial cutoff.

        Coefficients at index (v, u) are kept when u^2 + v^2 > threshold.
        DC components are always removed according to the transform mode.

        Parameters
        ----------
        threshold : float
            Squared radial frequency cutoff; must be non-negative.

        Returns
        -------
        Self
            Filtered DCT coefficients with the same shape and metadata.

        Raises
        ------
        ValueError
            If threshold is negative.
        """
        self._validate_filter_threshold(threshold)
        mask = self._build_high_pass_mask(threshold)
        filtered = self.value * mask.astype(np.float32)
        return self._with_filtered_coefficients(filtered)

    def low_pass_filter(self, threshold: float) -> Self:
        """
        Zero out high-frequency DCT coefficients above a radial cutoff.

        Coefficients at index (v, u) are kept when u^2 + v^2 <= threshold.

        Parameters
        ----------
        threshold : float
            Squared radial frequency cutoff; must be non-negative.

        Returns
        -------
        Self
            Filtered DCT coefficients with the same shape and metadata.

        Raises
        ------
        ValueError
            If threshold is negative.
        """
        self._validate_filter_threshold(threshold)
        radius_squared = self._frequency_radius_squared()
        mask = radius_squared <= threshold
        filtered = self.value * mask.astype(np.float32)
        return self._with_filtered_coefficients(filtered)

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
            Restored grayscale image array with shape (H, W); uint8 when
            is_uint8_cast_enabled is True, otherwise float32.
        """
        restored = self._apply_inverse_dct(self.value)
        restored = self._crop_to_source_shape(restored)
        if is_uint8_cast_enabled:
            return np.clip(restored, 0, 255).astype(np.uint8)
        return restored

    def laplacian(self, *, is_uint8_output: bool = True) -> ImageArray:
        """
        Frequency-domain Laplacian edge emphasis on DCT coefficients.

        Multiplies coefficients by the mask (u^2 + v^2), applies IDCT, crops to
        source_shape when spatial padding was used, then optionally normalizes
        |response| to uint8 for visualization.

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
        laplacian_mask = self._frequency_radius_squared()
        filtered = self.value * laplacian_mask
        restored = self._apply_inverse_dct(filtered)
        restored = self._crop_to_source_shape(restored)
        if is_uint8_output:
            return self._normalize_magnitude_to_uint8(restored)
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

    def _crop_to_source_shape(self, array: ImageArray) -> ImageArray:
        """
        Crop a spatial array back to source_shape when padding was applied.

        Parameters
        ----------
        array : ImageArray
            Spatial array of shape (height, width).

        Returns
        -------
        ImageArray
            Input array or its crop to source_shape.
        """
        if self.source_shape is None:
            return array
        height, width = self.source_shape
        return array[:height, :width]

    def _normalize_magnitude_to_uint8(self, restored: ImageArray) -> ImageArray:
        """
        Min-max normalize absolute spatial response to uint8.

        Parameters
        ----------
        restored : ImageArray
            Spatial response array.

        Returns
        -------
        ImageArray
            uint8 visualization image.
        """
        magnitude = np.abs(restored)
        peak = float(np.max(magnitude))
        trough = float(np.min(magnitude))
        if peak == trough:
            return np.zeros(magnitude.shape, dtype=np.uint8)
        destination = np.empty_like(magnitude, dtype=np.float32)
        normalized = cv2.normalize(magnitude, destination, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _validate_filter_threshold(self, threshold: float) -> None:
        """
        Validate a radial filter cutoff.

        Parameters
        ----------
        threshold : float
            Squared radial frequency cutoff.

        Raises
        ------
        ValueError
            If threshold is negative.
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

    def _build_high_pass_mask(self, threshold: float) -> BinaryArray:
        """
        Build a high-pass boolean mask with DC components cleared.

        Parameters
        ----------
        threshold : float
            Squared radial frequency cutoff.

        Returns
        -------
        BinaryArray
            Boolean mask array.
        """
        radius_squared = self._frequency_radius_squared()
        mask: BinaryArray = radius_squared > threshold
        self._clear_dc_components(mask)
        return mask
