from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_container.ch_order import ChannelOrder
from image_container.containers.pil import PILImageContainer
from image_container.format import ImageFormat


class TestPILImageContainerValidation:
  def test_rgb_mode_matches_channel_order(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      assert container.value.mode == "RGB"

  def test_rejects_mode_mismatch(self, rgb_pil_image: Image.Image) -> None:
      with pytest.raises(ValueError, match="expects image.mode"):
          PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.GRAY)

  def test_gray_container(self, gray_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=gray_pil_image, channel_order=ChannelOrder.GRAY)
      assert container.value.mode == "L"


class TestPILImageContainerGeometry:
  def test_shape_and_size(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      assert container.shape == (32, 64, 3)
      assert container.height == 32
      assert container.width == 64
      assert container.size == (64, 32)
      assert container.ch == 3
      assert container.format == ImageFormat.PIL

  def test_gray_geometry(self, gray_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=gray_pil_image, channel_order=ChannelOrder.GRAY)
      assert container.ch == 1
      assert container.shape == (32, 64, 1)


class TestPILImageContainerConvert:
  def test_to_array_bgr(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      bgr = container.to_array(ChannelOrder.BGR)
      assert bgr.shape == (32, 64, 3)
      rgb = container.to_array(ChannelOrder.RGB)
      assert np.array_equal(bgr[..., ::-1], rgb)

  def test_to_array_gray(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      gray = container.to_array(ChannelOrder.GRAY)
      assert gray.shape == (32, 64)

  def test_to_pil_returns_same_image(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      assert container.to_PIL() is rgb_pil_image

  def test_to_binary(self, gray_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=gray_pil_image, channel_order=ChannelOrder.GRAY)
      binary = container.to_binary(threshold=128)
      gray = np.asarray(gray_pil_image)
      assert binary.sum == int(np.sum(gray >= 128))

  def test_to_ch_swapped_image_gray(self, gray_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=gray_pil_image, channel_order=ChannelOrder.GRAY)
      result = container.to_ch_swapped_image(ChannelOrder.GRAY)
      assert result.mode == "L"
      assert result.size == gray_pil_image.size


class TestPILImageContainerCrop:
  def test_crop(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      cropped = container.crop((slice(4, 20), slice(8, 40)))
      assert cropped.shape == (16, 32, 3)
      assert cropped.value.size == (32, 16)


class TestPILImageContainerStats:
  def test_gray_stats(self, gray_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=gray_pil_image, channel_order=ChannelOrder.GRAY)
      gray = np.asarray(gray_pil_image)
      assert container.mean_gray == pytest.approx(float(np.mean(gray)))
      assert container.max_gray == float(np.max(gray))
      assert container.min_gray == float(np.min(gray))

  def test_bgr_stats(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      assert container.mean_bgr.shape == (3,)
      assert container.max_bgr.shape == (3,)
      assert container.min_bgr.shape == (3,)

  def test_compute_histogram_gray(self, gray_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=gray_pil_image, channel_order=ChannelOrder.GRAY)
      hist = container.compute_histogram()
      assert hist.shape == (256, 1)
      assert hist.dtype == np.float32

  def test_compute_histogram_bgr(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      hist = container.compute_histogram(histogram_order=ChannelOrder.BGR)
      assert hist.shape == (256, 3)


class TestPILImageContainerIO:
  def test_from_path_and_save(
      self,
      temp_image_dir: Path,
      tmp_path: Path,
  ) -> None:
      image_path = temp_image_dir / "color.png"
      container = PILImageContainer.from_path(str(image_path))
      assert container.channel_order == ChannelOrder.RGB
      assert container.value.mode == "RGB"

      save_path = tmp_path / "saved.png"
      container.save(str(save_path))
      assert save_path.is_file()

  def test_str_representation(self, rgb_pil_image: Image.Image) -> None:
      container = PILImageContainer(value=rgb_pil_image, channel_order=ChannelOrder.RGB)
      text = str(container)
      assert "PILImageContainer" in text
