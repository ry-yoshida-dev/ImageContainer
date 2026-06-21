from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_container.ch_order import ChannelOrder


class TestChannelOrderProperties:
  @pytest.mark.parametrize(
      ("order", "expected_mode", "is_3ch", "is_1ch"),
      [
          (ChannelOrder.RGB, "RGB", True, False),
          (ChannelOrder.BGR, "RGB", True, False),
          (ChannelOrder.GRAY, "L", False, True),
          (ChannelOrder.HSV, "HSV", True, False),
          (ChannelOrder.LAB, "LAB", True, False),
      ],
  )
  def test_properties(
      self,
      order: ChannelOrder,
      expected_mode: str,
      is_3ch: bool,
      is_1ch: bool,
  ) -> None:
      assert order.pil_mode == expected_mode
      assert order.is_3ch is is_3ch
      assert order.is_1ch is is_1ch


class TestChannelOrderConversion:
  @pytest.fixture
  def bgr_sample(self) -> np.ndarray:
      return np.array([[[0, 128, 255]]], dtype=np.uint8)

  def test_identity_returns_copy(self, bgr_sample: np.ndarray) -> None:
      converter = ChannelOrder.BGR.cv2_array_converter(ChannelOrder.BGR)
      result = converter(bgr_sample)
      assert np.array_equal(result, bgr_sample)
      assert result is not bgr_sample

  def test_bgr_to_rgb(self, bgr_sample: np.ndarray) -> None:
      converter = ChannelOrder.RGB.cv2_array_converter(ChannelOrder.BGR)
      result = converter(bgr_sample)
      assert result[0, 0].tolist() == [255, 128, 0]

  def test_bgr_to_gray(self, bgr_sample: np.ndarray) -> None:
      converter = ChannelOrder.GRAY.cv2_array_converter(ChannelOrder.BGR)
      result = converter(bgr_sample)
      assert result.shape == (1, 1)
      expected = cv2.cvtColor(bgr_sample, cv2.COLOR_BGR2GRAY)
      assert np.array_equal(result, expected)

  def test_gray_to_bgr(self) -> None:
      gray = np.array([[42]], dtype=np.uint8)
      converter = ChannelOrder.BGR.cv2_array_converter(ChannelOrder.GRAY)
      result = converter(gray)
      assert result.shape == (1, 1, 3)
      assert np.all(result == 42)

  def test_roundtrip_bgr_rgb_bgr(self, bgr_sample: np.ndarray) -> None:
      to_rgb = ChannelOrder.RGB.cv2_array_converter(ChannelOrder.BGR)
      to_bgr = ChannelOrder.BGR.cv2_array_converter(ChannelOrder.RGB)
      restored = to_bgr(to_rgb(bgr_sample))
      assert np.array_equal(restored, bgr_sample)

  def test_multi_step_conversion_flags(self) -> None:
      flags = ChannelOrder.LAB.cv2_conversion_flags(ChannelOrder.HSV)
      assert len(flags) == 2

  def test_identity_flags_are_empty(self) -> None:
      assert ChannelOrder.RGB.cv2_conversion_flags(ChannelOrder.RGB) == ()
