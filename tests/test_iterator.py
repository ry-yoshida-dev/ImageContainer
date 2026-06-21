from __future__ import annotations

from pathlib import Path

import pytest

from image_container.containers.array import ArrayImageContainer
from image_container.containers.pil import PILImageContainer
from image_container.format import ImageFormat
from image_container.iterator import ImageIterator


class TestImageIterator:
  def test_from_dir_collects_images(self, temp_image_dir: Path) -> None:
      iterator = ImageIterator.from_dir(str(temp_image_dir), ImageFormat.ARRAY)
      assert len(iterator) == 3
      assert all(path.endswith((".png", ".jpg")) for path in iterator.paths)

  def test_iter_yields_array_containers(self, temp_image_dir: Path) -> None:
      iterator = ImageIterator.from_dir(str(temp_image_dir), ImageFormat.ARRAY)
      containers = list(iterator)
      assert len(containers) == 3
      assert all(isinstance(c, ArrayImageContainer) for c in containers)

  def test_iter_yields_pil_containers_for_color_images(
      self,
      temp_image_dir: Path,
  ) -> None:
      iterator = ImageIterator(
          image_format=ImageFormat.PIL,
          paths=[
              str(temp_image_dir / "color.png"),
              str(temp_image_dir / "nested" / "nested_color.jpg"),
          ],
      )
      containers = list(iterator)
      assert len(containers) == 2
      assert all(isinstance(c, PILImageContainer) for c in containers)

  def test_iter_gray_pil_from_path_raises(self, temp_image_dir: Path) -> None:
      iterator = ImageIterator(
          image_format=ImageFormat.PIL,
          paths=[str(temp_image_dir / "gray.png")],
      )
      with pytest.raises(ValueError, match="expects image.mode='RGB'"):
          next(iter(iterator))

  def test_from_dir_missing_raises(self) -> None:
      with pytest.raises(ValueError, match="Directory does not exist"):
          ImageIterator.from_dir("/nonexistent/dir", ImageFormat.ARRAY)

  def test_from_dir_empty_raises(self, tmp_path: Path) -> None:
      empty_dir = tmp_path / "empty"
      empty_dir.mkdir()
      with pytest.raises(ValueError, match="No image files found"):
          ImageIterator.from_dir(str(empty_dir), ImageFormat.ARRAY)

  def test_empty_paths_raises(self) -> None:
      with pytest.raises(ValueError, match="paths must contain at least one"):
          ImageIterator(image_format=ImageFormat.ARRAY, paths=[])

  def test_torch_tensor_not_implemented(self, temp_image_dir: Path) -> None:
      iterator = ImageIterator.from_dir(str(temp_image_dir), ImageFormat.TORCH_TENSOR)
      with pytest.raises(NotImplementedError, match="TORCH_TENSOR"):
          next(iter(iterator))
