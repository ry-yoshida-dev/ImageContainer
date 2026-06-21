from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from image_container.ch_order import ChannelOrder
from image_container.containers.array import ArrayImageContainer
from image_container.format import ImageFormat


class TestArrayImageContainerValidation:
  def test_bgr_requires_3d_uint8(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      assert container.value.dtype == np.uint8

  def test_gray_requires_2d(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      assert container.shape == (32, 64, 1)

  def test_rejects_wrong_channel_count(self) -> None:
      bad = np.zeros((8, 8, 4), dtype=np.uint8)
      with pytest.raises(ValueError, match="must have 3 channels"):
          ArrayImageContainer(value=bad, channel_order=ChannelOrder.BGR)

  def test_rejects_non_uint8(self, bgr_array: np.ndarray) -> None:
      bad = bgr_array.astype(np.float32)
      with pytest.raises(ValueError, match="must have uint8 dtype"):
          ArrayImageContainer(value=bad, channel_order=ChannelOrder.BGR)

  def test_value_is_read_only(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      with pytest.raises(ValueError):
          container.value[0, 0, 0] = 0


class TestArrayImageContainerGeometry:
  def test_shape_and_size(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      assert container.shape == (32, 64, 3)
      assert container.height == 32
      assert container.width == 64
      assert container.size == (64, 32)
      assert container.ch == 3
      assert container.format == ImageFormat.ARRAY

  def test_gray_geometry(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      assert container.ch == 1


class TestArrayImageContainerConvert:
  def test_to_array_identity_returns_copy(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      result = container.to_array(ChannelOrder.BGR)
      assert np.array_equal(result, bgr_array)
      assert result is not container.value

  def test_to_array_rgb_swaps_channels(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      rgb = container.to_array(ChannelOrder.RGB)
      assert np.array_equal(rgb, bgr_array[..., ::-1])

  def test_to_gray(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      gray = container.to_gray()
      expected = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
      assert np.array_equal(gray, expected)

  def test_to_pil_from_bgr(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      pil_image = container.to_PIL()
      assert pil_image.mode == "RGB"
      assert pil_image.size == (64, 32)

  def test_to_binary(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      binary = container.to_binary(threshold=128)
      assert binary.shape == gray_array.shape
      assert binary.sum == int(np.sum(gray_array >= 128))

  def test_to_ch_swapped_image(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      rgb = container.to_ch_swapped_image(ChannelOrder.RGB)
      assert np.array_equal(rgb, bgr_array[..., ::-1])


class TestArrayImageContainerCrop:
  def test_crop(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      cropped = container.crop((slice(4, 20), slice(8, 40)))
      assert cropped.shape == (16, 32, 3)
      assert np.array_equal(cropped.value, bgr_array[4:20, 8:40])


class TestArrayImageContainerStats:
  def test_gray_stats(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      assert container.mean_gray == pytest.approx(float(np.mean(gray_array)))
      assert container.max_gray == float(np.max(gray_array))
      assert container.min_gray == float(np.min(gray_array))

  def test_bgr_stats(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      assert container.mean_bgr.shape == (3,)
      assert container.max_bgr.shape == (3,)
      assert container.min_bgr.shape == (3,)

  def test_compute_histogram_gray(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      hist = container.compute_histogram()
      assert hist.shape == (256, 1)
      assert hist.dtype == np.float32

  def test_compute_histogram_bgr(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      hist = container.compute_histogram(histogram_order=ChannelOrder.BGR)
      assert hist.shape == (256, 3)

  def test_compute_histogram_rejects_unsupported_order(
      self,
      gray_array: np.ndarray,
  ) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      with pytest.raises(ValueError, match="histogram_order must be"):
          container.compute_histogram(histogram_order=ChannelOrder.HSV)


class TestArrayImageContainerFilter:
  def test_laplacian(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      result = container.laplacian()
      assert result.shape == gray_array.shape
      assert result.dtype == np.uint8

  def test_sobel(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      result = container.sobel()
      assert result.shape == (*gray_array.shape, 2)
      assert result.dtype == np.uint8


class TestArrayImageContainerProcess:
  def test_equalize_histogram_gray(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      result = container.equalize_histogram()
      assert result.shape == gray_array.shape
      assert result.dtype == np.uint8

  def test_clahe_gray(self, gray_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=gray_array, channel_order=ChannelOrder.GRAY)
      result = container.clahe()
      assert result.shape == gray_array.shape
      assert result.dtype == np.uint8

  def test_equalize_histogram_color(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      result = container.equalize_histogram()
      assert result.shape == bgr_array.shape


class TestArrayImageContainerHash:
  def test_phash_shape(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      result = container.phash
      assert result.shape == (1, 8)
      assert result.dtype == np.uint8

  def test_identical_images_same_phash(self, bgr_array: np.ndarray) -> None:
      first = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      second = ArrayImageContainer(value=bgr_array.copy(), channel_order=ChannelOrder.BGR)
      assert np.array_equal(first.phash, second.phash)


class TestArrayImageContainerIO:
  def test_from_path_and_save(
      self,
      temp_image_dir: Path,
      tmp_path: Path,
  ) -> None:
      image_path = temp_image_dir / "color.png"
      container = ArrayImageContainer.from_path(str(image_path))
      assert container.channel_order == ChannelOrder.BGR
      assert container.shape[2] == 3

      save_path = tmp_path / "saved.png"
      container.save(str(save_path))
      assert save_path.is_file()

  def test_from_path_missing_raises(self) -> None:
      with pytest.raises(FileNotFoundError, match="Image not found"):
          ArrayImageContainer.from_path("/nonexistent/image.png")

  def test_str_representation(self, bgr_array: np.ndarray) -> None:
      container = ArrayImageContainer(value=bgr_array, channel_order=ChannelOrder.BGR)
      text = str(container)
      assert "ArrayImageContainer" in text
      assert "BGR" in text
