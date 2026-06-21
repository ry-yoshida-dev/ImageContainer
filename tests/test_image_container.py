from __future__ import annotations

import pytest
from PIL import Image

from image_container.ch_order import ChannelOrder
from image_container.container import ImageContainer
from image_container.containers.array import ArrayImageContainer
from image_container.containers.pil import PILImageContainer


class TestImageContainerRegister:
  def test_register_ndarray(self, bgr_array: np.ndarray) -> None:
      container = ImageContainer.register(bgr_array, ChannelOrder.BGR)
      assert isinstance(container, ArrayImageContainer)
      assert container.channel_order == ChannelOrder.BGR

  def test_register_pil_image(self, rgb_pil_image: Image.Image) -> None:
      container = ImageContainer.register(rgb_pil_image, ChannelOrder.RGB)
      assert isinstance(container, PILImageContainer)
      assert container.channel_order == ChannelOrder.RGB

  def test_register_unsupported_type_raises(self) -> None:
      with pytest.raises(ValueError, match="Unsupported image type"):
          ImageContainer.register("not an image", ChannelOrder.BGR)  # type: ignore[arg-type]
