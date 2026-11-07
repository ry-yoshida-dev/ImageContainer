from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from ...types import ImageArray


@dataclass(frozen=True)
class ZigzagScan:
    """
    JPEG zigzag scan order of the coefficients in a square DCT block.

    The scan walks the anti-diagonals from the DC corner towards the highest
    frequency, alternating direction and starting to the right, as defined in
    ITU-T T.81. Truncating a zigzag-ordered coefficient vector therefore drops
    only the highest frequencies.

    Parameters
    ----------
    block_size : int
        Side length of each square DCT block.
    """

    block_size: int

    def __post_init__(self) -> None:
        """
        Validate block size.

        Raises
        ------
        ValueError
            If block_size is not positive.
        """
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

    @property
    def coefficient_capacity(self) -> int:
        """
        Get the number of coefficients in one block.

        Returns
        -------
        int
            block_size squared.
        """
        return self.block_size * self.block_size

    @cached_property
    def raster_indices(self) -> NDArray[np.intp]:
        """
        Get the row-major in-block index of each zigzag slot.

        Returns
        -------
        NDArray[np.intp]
            Array of shape (block_size ** 2,) whose entry k is the row-major
            position inside a block that zigzag slot k belongs at.
        """
        positions: list[int] = []
        for diagonal in range(2 * self.block_size - 1):
            first_row = max(0, diagonal - self.block_size + 1)
            last_row = min(diagonal, self.block_size - 1)
            cells = [
                row * self.block_size + (diagonal - row)
                for row in range(first_row, last_row + 1)
            ]
            if diagonal % 2 == 0:
                cells.reverse()
            positions.extend(cells)
        return np.asarray(positions, dtype=np.intp)

    def scatter(self, coefficients: ImageArray) -> ImageArray:
        """
        Place zigzag-ordered coefficients into the spatial block layout.

        Zigzag slots beyond the given coefficient count are filled with zero.

        Parameters
        ----------
        coefficients : ImageArray
            Array of shape (blocks_y, blocks_x, coefficient_count) holding the
            leading zigzag coefficients of every block.

        Returns
        -------
        ImageArray
            Float32 array of shape (blocks_y * block_size, blocks_x * block_size)
            where each block holds its coefficients in row-major layout.

        Raises
        ------
        ValueError
            If coefficients is not 3D or coefficient_count is not in
            [1, block_size ** 2].
        """
        if coefficients.ndim != 3:
            raise ValueError(
                "Zigzag coefficients must have shape "
                + f"(blocks_y, blocks_x, coefficient_count), got {coefficients.shape}"
            )
        blocks_y, blocks_x, coefficient_count = coefficients.shape
        self._validate_coefficient_count(coefficient_count)
        blocks = np.zeros(
            (blocks_y, blocks_x, self.coefficient_capacity),
            dtype=np.float32,
        )
        blocks[..., self.raster_indices[:coefficient_count]] = coefficients
        layout: ImageArray = (
            blocks.reshape(blocks_y, blocks_x, self.block_size, self.block_size)
            .transpose(0, 2, 1, 3)
            .reshape(blocks_y * self.block_size, blocks_x * self.block_size)
        )
        return layout

    def gather(self, layout: ImageArray, coefficient_count: int) -> ImageArray:
        """
        Read the leading zigzag coefficients of every block.

        Parameters
        ----------
        layout : ImageArray
            Array of shape (height, width) with height and width divisible by
            block_size, each block holding coefficients in row-major layout.
        coefficient_count : int
            Number of leading zigzag coefficients to read per block.

        Returns
        -------
        ImageArray
            Array of shape (height // block_size, width // block_size,
            coefficient_count) with the dtype of layout.

        Raises
        ------
        ValueError
            If layout is not 2D, its dimensions are not divisible by block_size,
            or coefficient_count is not in [1, block_size ** 2].
        """
        self._validate_coefficient_count(coefficient_count)
        gathered: ImageArray = self._flatten_blocks(layout)[
            ...,
            self.raster_indices[:coefficient_count],
        ]
        return gathered

    def validate_truncated_layout(
        self,
        layout: ImageArray,
        coefficient_count: int,
    ) -> None:
        """
        Validate that only the leading zigzag slots of every block are non-zero.

        Only the dropped slots are inspected, and nothing is inspected when
        coefficient_count covers the whole block.

        Parameters
        ----------
        layout : ImageArray
            Array of shape (height, width) with height and width divisible by
            block_size, each block holding coefficients in row-major layout.
        coefficient_count : int
            Number of leading zigzag coefficients per block allowed to be
            non-zero.

        Raises
        ------
        ValueError
            If layout is not 2D, its dimensions are not divisible by block_size,
            coefficient_count is not in [1, block_size ** 2], or a coefficient
            beyond coefficient_count is non-zero or NaN.
        """
        self._validate_coefficient_count(coefficient_count)
        flattened_blocks = self._flatten_blocks(layout)
        if coefficient_count == self.coefficient_capacity:
            return
        dropped = flattened_blocks[..., self.raster_indices[coefficient_count:]]
        if np.any(dropped != 0):
            raise ValueError(
                "Coefficients beyond the leading "
                + f"{coefficient_count} zigzag slots must be zero"
            )

    def _flatten_blocks(self, layout: ImageArray) -> ImageArray:
        """
        View each block of a spatial layout as a flat row-major vector.

        Parameters
        ----------
        layout : ImageArray
            Array of shape (height, width) with height and width divisible by
            block_size.

        Returns
        -------
        ImageArray
            Array of shape (height // block_size, width // block_size,
            block_size ** 2) with the dtype of layout.

        Raises
        ------
        ValueError
            If layout is not 2D or its dimensions are not divisible by
            block_size.
        """
        if layout.ndim != 2:
            raise ValueError(f"Layout must have 2 dimensions, got {layout.ndim}")
        height, width = layout.shape
        if height % self.block_size != 0 or width % self.block_size != 0:
            raise ValueError(
                f"Layout shape {layout.shape} must be divisible by "
                + f"block_size {self.block_size}"
            )
        blocks_y = height // self.block_size
        blocks_x = width // self.block_size
        flattened_blocks: ImageArray = (
            layout.reshape(blocks_y, self.block_size, blocks_x, self.block_size)
            .transpose(0, 2, 1, 3)
            .reshape(blocks_y, blocks_x, self.coefficient_capacity)
        )
        return flattened_blocks

    def _validate_coefficient_count(self, coefficient_count: int) -> None:
        """
        Validate a per-block zigzag coefficient count.

        Parameters
        ----------
        coefficient_count : int
            Number of zigzag coefficients per block.

        Raises
        ------
        ValueError
            If coefficient_count is not in [1, block_size ** 2].
        """
        if not 1 <= coefficient_count <= self.coefficient_capacity:
            raise ValueError(
                f"coefficient_count must be in [1, {self.coefficient_capacity}], "
                + f"got {coefficient_count}"
            )
