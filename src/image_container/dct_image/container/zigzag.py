from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ...types import ImageArray
from ..utils.block_transform import BlockDctTransform
from ..utils.zigzag_scan import ZigzagScan
from .blocked import BlockedDctImage


@dataclass(kw_only=True)
class ZigzagDctCoefficients:
    """
    Per-block DCT coefficient vectors in JPEG zigzag order.

    This is the layout used when coefficients are serialized block by block,
    optionally truncated to the leading low-frequency slots. Convert it with
    to_blocked before applying the inverse DCT or frequency filters.

    Parameters
    ----------
    value : ImageArray
        Float32 array of shape (blocks_y, blocks_x, coefficient_count) holding
        the leading zigzag coefficients of every block.
    block_size : int
        Side length of each square DCT block in pixels; must be positive and
        even.
    source_shape : tuple[int, int] | None, default None
        Original (height, width) before block-size padding. It is carried over
        by to_blocked so that idct crops the restored image back to this shape.

    Examples
    --------
    >>> import numpy as np
    >>> coefficients = ZigzagDctCoefficients(
    ...     value=np.zeros((2, 3, 15), dtype=np.float32),
    ...     block_size=8,
    ... )
    >>> coefficients.to_blocked().shape
    (16, 24)
    """

    value: ImageArray
    block_size: int
    source_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """
        Post-initialization validation.

        Raises
        ------
        ValueError
            If value is not a 3D float32 array, block_size is not a positive
            even number, coefficient_count is not in [1, block_size ** 2], or
            source_shape does not fit inside the block grid.
        """
        if self.value.ndim != 3:
            raise ValueError(
                "Zigzag coefficients must have shape "
                + f"(blocks_y, blocks_x, coefficient_count), got {self.value.shape}"
            )
        if self.value.dtype != np.float32:
            raise ValueError(
                f"Zigzag coefficients must have float32 dtype, got {self.value.dtype}"
            )
        _ = BlockDctTransform(block_size=self.block_size)
        capacity = self.block_size * self.block_size
        if not 1 <= self.coefficient_count <= capacity:
            raise ValueError(
                f"coefficient_count must be in [1, {capacity}], "
                + f"got {self.coefficient_count}"
            )
        self._validate_source_shape()

    def _validate_source_shape(self) -> None:
        """
        Validate that source_shape fits inside the block grid.

        Raises
        ------
        ValueError
            If a dimension of source_shape is not positive or exceeds the
            padded extent blocks * block_size.
        """
        if self.source_shape is None:
            return
        height, width = self.source_shape
        padded_height = self.blocks_y * self.block_size
        padded_width = self.blocks_x * self.block_size
        if not (0 < height <= padded_height and 0 < width <= padded_width):
            raise ValueError(
                f"source_shape {self.source_shape} must be positive and fit "
                + f"inside the padded shape ({padded_height}, {padded_width})"
            )

    @property
    def blocks_y(self) -> int:
        """
        Get the number of blocks along the vertical axis.

        Returns
        -------
        int
            First dimension of value.
        """
        return int(self.value.shape[0])

    @property
    def blocks_x(self) -> int:
        """
        Get the number of blocks along the horizontal axis.

        Returns
        -------
        int
            Second dimension of value.
        """
        return int(self.value.shape[1])

    @property
    def coefficient_count(self) -> int:
        """
        Get the number of zigzag coefficients stored per block.

        Returns
        -------
        int
            Last dimension of value.
        """
        return int(self.value.shape[2])

    def to_blocked(self) -> BlockedDctImage:
        """
        Place the coefficients into the spatial block layout.

        Zigzag slots beyond coefficient_count are zero in the result, so idct
        yields a low-pass approximation when the vectors are truncated.

        Returns
        -------
        BlockedDctImage
            Block DCT coefficients of shape
            (blocks_y * block_size, blocks_x * block_size) with the same
            coefficient_count and source_shape.
        """
        layout = ZigzagScan(block_size=self.block_size).scatter(self.value)
        return BlockedDctImage(
            value=layout,
            block_size=self.block_size,
            coefficient_count=self.coefficient_count,
            source_shape=self.source_shape,
        )

    @classmethod
    def from_blocked(
        cls,
        image: BlockedDctImage,
        coefficient_count: int | None = None,
    ) -> ZigzagDctCoefficients:
        """
        Read the leading zigzag coefficients of every block.

        Parameters
        ----------
        image : BlockedDctImage
            Block DCT coefficients in spatial layout.
        coefficient_count : int | None, default None
            Number of leading zigzag coefficients to keep per block. None keeps
            every coefficient available in image.

        Returns
        -------
        ZigzagDctCoefficients
            Zigzag coefficient vectors of shape
            (image.height // block_size, image.width // block_size,
            coefficient_count) with the source_shape of image.

        Raises
        ------
        ValueError
            If coefficient_count is not in [1, image.coefficient_count].
        """
        kept_count = (
            image.coefficient_count if coefficient_count is None else coefficient_count
        )
        if not 1 <= kept_count <= image.coefficient_count:
            raise ValueError(
                f"coefficient_count must be in [1, {image.coefficient_count}] "
                + f"for this image, got {kept_count}"
            )
        scan = ZigzagScan(block_size=image.block_size)
        return cls(
            value=np.asarray(scan.gather(image.value, kept_count), dtype=np.float32),
            block_size=image.block_size,
            source_shape=image.source_shape,
        )
