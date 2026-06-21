from __future__ import annotations

import pytest

from image_container.extention import Extention


class TestExtention:
  @pytest.mark.parametrize(
      ("ext", "is_lossless"),
      [
          (Extention.PNG, True),
          (Extention.TIFF, True),
          (Extention.TIF, True),
          (Extention.BMP, True),
          (Extention.GIF, True),
          (Extention.JPG, False),
          (Extention.JPEG, False),
          (Extention.HEIC, False),
          (Extention.WEBP, False),
          (Extention.AVIF, False),
      ],
  )
  def test_is_lossless(self, ext: Extention, is_lossless: bool) -> None:
      assert ext.is_lossless is is_lossless

  @pytest.mark.parametrize(
      ("ext", "is_transparency_supported"),
      [
          (Extention.PNG, True),
          (Extention.GIF, True),
          (Extention.HEIC, True),
          (Extention.WEBP, True),
          (Extention.AVIF, True),
          (Extention.JPG, False),
          (Extention.JPEG, False),
          (Extention.TIFF, False),
          (Extention.TIF, False),
          (Extention.BMP, False),
      ],
  )
  def test_is_transparency_supported(
      self,
      ext: Extention,
      is_transparency_supported: bool,
  ) -> None:
      assert ext.is_transparency_supported is is_transparency_supported
