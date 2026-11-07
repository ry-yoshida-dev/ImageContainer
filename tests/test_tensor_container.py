from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from image_container.ch_order import ChannelOrder
from image_container.containers.tensor import TensorImageContainer
from image_container.format import ImageFormat


class TestTensorImageContainerValidation:
    def test_bgr_requires_3d_uint8(self, bgr_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        assert container.value.dtype == torch.uint8

    def test_gray_requires_1_channel(self, gray_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=gray_tensor, channel_order=ChannelOrder.GRAY
        )
        assert container.shape == (32, 64, 1)

    def test_rejects_wrong_channel_count(self) -> None:
        bad = torch.zeros((4, 8, 8), dtype=torch.uint8)
        with pytest.raises(ValueError, match="must have 3 channel"):
            _ = TensorImageContainer(value=bad, channel_order=ChannelOrder.BGR)

    def test_rejects_non_uint8(self, bgr_tensor: torch.Tensor) -> None:
        bad = bgr_tensor.to(torch.float32)
        with pytest.raises(ValueError, match="must have uint8 dtype"):
            _ = TensorImageContainer(value=bad, channel_order=ChannelOrder.BGR)

    def test_rejects_non_chw(self, bgr_tensor: torch.Tensor) -> None:
        bad = bgr_tensor.unsqueeze(0)
        with pytest.raises(ValueError, match="3 dimensions"):
            _ = TensorImageContainer(value=bad, channel_order=ChannelOrder.BGR)


class TestTensorImageContainerGeometry:
    def test_shape_and_size(self, bgr_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        assert container.shape == (32, 64, 3)
        assert container.height == 32
        assert container.width == 64
        assert container.size == (64, 32)
        assert container.ch == 3
        assert container.format == ImageFormat.TORCH_TENSOR

    def test_gray_geometry(self, gray_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=gray_tensor, channel_order=ChannelOrder.GRAY
        )
        assert container.ch == 1
        assert container.shape == (32, 64, 1)


class TestTensorImageContainerConvert:
    def test_to_array_bgr(
        self, bgr_array: np.ndarray, bgr_tensor: torch.Tensor
    ) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        bgr = container.to_array(ChannelOrder.BGR)
        assert np.array_equal(bgr, bgr_array)

    def test_to_array_rgb_matches_channel_flip(self, bgr_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        bgr = container.to_array(ChannelOrder.BGR)
        rgb = container.to_array(ChannelOrder.RGB)
        assert np.array_equal(bgr[..., ::-1], rgb)

    def test_to_array_gray(self, bgr_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        gray = container.to_array(ChannelOrder.GRAY)
        assert gray.shape == (32, 64)

    def test_to_pil_roundtrips_through_rgb(
        self, bgr_array: np.ndarray, bgr_tensor: torch.Tensor
    ) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        pil = container.to_PIL()
        assert pil.mode == "RGB"
        assert np.array_equal(np.asarray(pil), bgr_array[..., ::-1])

    def test_to_binary(self, gray_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=gray_tensor, channel_order=ChannelOrder.GRAY
        )
        binary = container.to_binary(threshold=128)
        gray = gray_tensor[0].numpy()
        assert binary.sum == int(np.sum(gray >= 128))

    def test_to_ch_swapped_image_rgb_bgr_stays_on_device(
        self, bgr_tensor: torch.Tensor
    ) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        swapped = container.to_ch_swapped_image(ChannelOrder.RGB)
        assert swapped.device == bgr_tensor.device
        assert torch.equal(swapped, bgr_tensor.flip(0))

    def test_to_ch_swapped_image_gray(self, gray_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=gray_tensor, channel_order=ChannelOrder.GRAY
        )
        result = container.to_ch_swapped_image(ChannelOrder.GRAY)
        assert torch.equal(result, gray_tensor)


class TestTensorImageContainerCrop:
    def test_crop(self, bgr_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        cropped = container.crop((slice(4, 20), slice(8, 40)))
        assert cropped.shape == (16, 32, 3)
        assert torch.equal(cropped.value, bgr_tensor[:, 4:20, 8:40])


class TestTensorImageContainerIO:
    def test_from_path_and_save(
        self,
        temp_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        image_path = temp_image_dir / "color.png"
        container = TensorImageContainer.from_path(str(image_path))
        assert container.channel_order == ChannelOrder.RGB
        assert container.value.dtype == torch.uint8

        save_path = tmp_path / "saved.png"
        container.save(str(save_path))
        assert save_path.is_file()

    def test_str_representation(self, bgr_tensor: torch.Tensor) -> None:
        container = TensorImageContainer(
            value=bgr_tensor, channel_order=ChannelOrder.BGR
        )
        text = str(container)
        assert "TensorImageContainer" in text
