from __future__ import annotations

from image_container.format import ImageFormat


class TestImageFormat:
  def test_enum_values(self) -> None:
      assert ImageFormat.ARRAY.value == "array"
      assert ImageFormat.PIL.value == "pil"
      assert ImageFormat.TORCH_TENSOR.value == "torch_tensor"

  def test_members_are_unique(self) -> None:
      values = [member.value for member in ImageFormat]
      assert len(values) == len(set(values))
