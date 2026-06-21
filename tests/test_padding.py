from __future__ import annotations

import numpy as np
import pytest

from image_container.utils.padding import padding_to_block_size, padding_to_even


class TestPaddingToEven:
  def test_even_dimensions_unchanged(self) -> None:
      array = np.zeros((32, 64), dtype=np.uint8)
      padded, source_shape = padding_to_even(array)
      assert padded is array
      assert source_shape is None

  def test_odd_dimensions_padded(self) -> None:
      array = np.arange(33 * 65, dtype=np.uint8).reshape(33, 65)
      padded, source_shape = padding_to_even(array)
      assert padded.shape == (34, 66)
      assert source_shape == (33, 65)
      assert np.array_equal(padded[:33, :65], array)
      assert np.all(padded[33, :] == padded[32, :])
      assert np.all(padded[:, 65] == padded[:, 64])

  def test_rejects_non_2d(self) -> None:
      with pytest.raises(ValueError, match="Array must be 2D"):
          padding_to_even(np.zeros((2, 2, 3), dtype=np.uint8))


class TestPaddingToBlockSize:
  def test_already_divisible_unchanged(self) -> None:
      array = np.zeros((32, 64), dtype=np.uint8)
      padded, source_shape = padding_to_block_size(array, block_size=8)
      assert padded is array
      assert source_shape is None

  def test_pads_to_block_multiple(self) -> None:
      array = np.arange(33 * 65, dtype=np.uint8).reshape(33, 65)
      block_size = 8
      padded, source_shape = padding_to_block_size(array, block_size=block_size)
      assert padded.shape == (40, 72)
      assert source_shape == (33, 65)
      assert padded.shape[0] % block_size == 0
      assert padded.shape[1] % block_size == 0
      assert np.array_equal(padded[:33, :65], array)

  def test_rejects_non_positive_block_size(self) -> None:
      array = np.zeros((8, 8), dtype=np.uint8)
      with pytest.raises(ValueError, match="block_size must be positive"):
          padding_to_block_size(array, block_size=0)

  def test_rejects_non_2d(self) -> None:
      with pytest.raises(ValueError, match="Array must be 2D"):
          padding_to_block_size(np.zeros((2, 2, 3), dtype=np.uint8), block_size=8)
