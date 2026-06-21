from __future__ import annotations

import numpy as np
import pytest

from image_container.binary_image import BinaryImage, ConnectedComponents, Connectivity


class TestBinaryImage:
  def test_accepts_bool_array(self) -> None:
      value = np.array([[True, False], [False, True]], dtype=bool)
      binary = BinaryImage(value=value)
      assert binary.value.dtype == bool
      assert binary.sum == 2

  def test_accepts_uint8_zeros_and_ones(self) -> None:
      value = np.array([[0, 1], [1, 0]], dtype=np.uint8)
      binary = BinaryImage(value=value)
      assert binary.value.dtype == bool

  def test_rejects_invalid_values(self) -> None:
      value = np.array([[0, 2]], dtype=np.uint8)
      with pytest.raises(ValueError, match="must contain only 0/1"):
          BinaryImage(value=value)

  def test_rejects_non_2d(self) -> None:
      with pytest.raises(ValueError, match="must have 2 dimensions"):
          BinaryImage(value=np.zeros((2, 2, 1), dtype=bool))

  def test_geometry_properties(self) -> None:
      value = np.zeros((10, 20), dtype=bool)
      value[3:7, 5:15] = True
      binary = BinaryImage(value=value)
      assert binary.shape == (10, 20)
      assert binary.height == 10
      assert binary.width == 20
      assert binary.size == (20, 10)
      assert binary.sum == 40
      assert binary.mean == pytest.approx(0.2)


class TestConnectedComponents:
  def test_four_connectivity(self) -> None:
      value = np.array(
          [
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 0],
          ],
          dtype=bool,
      )
      binary = BinaryImage(value=value)
      result = binary.connected_components(Connectivity.FOUR)
      assert isinstance(result, ConnectedComponents)
      assert result.n_labels == 2
      assert result.labels.shape == (3, 3)
      assert result.stats.shape == (2, 5)
      assert result.centroids.shape == (2, 2)

  def test_eight_connectivity_merges_diagonal(self) -> None:
      value = np.array(
          [
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 0],
          ],
          dtype=bool,
      )
      binary = BinaryImage(value=value)
      four = binary.connected_components(Connectivity.FOUR)
      eight = binary.connected_components(Connectivity.EIGHT)
      assert four.n_labels == 3
      assert eight.n_labels == 2
