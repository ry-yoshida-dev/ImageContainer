from __future__ import annotations

import os
from typing import override

import cv2
import numpy as np
import torch
from PIL import Image

from ...binary_image import BinaryImage
from ...ch_order import ChannelOrder
from ...container import ImageContainer
from ...format import ImageFormat
from ...types import UInt8Image

_FLIPPABLE_ORDERS = {ChannelOrder.RGB, ChannelOrder.BGR}


class TensorImageContainer(ImageContainer[torch.Tensor]):
    """
    Container class for torch tensors.

    Stored channel-first (C, H, W), on whatever device the tensor already
    lives on: a container built from a CUDA decode stays on the GPU through
    crop and channel-order operations, and only crosses to the CPU for
    operations only OpenCV/Pillow provide (to_PIL, to_array, to_binary, save).

    Attributes
    ----------
    value: torch.Tensor
        The tensor, shaped (C, H, W).
    channel_order: ChannelOrder
        The channel order of the image.
    """

    @override
    def _validate_image(self) -> None:
        """
        Validate the image.

        Raises
        ------
        ValueError:
            If the tensor is not 3D (C, H, W).
            If the tensor's channel count does not match channel_order.
            If the tensor does not have uint8 dtype.
        """
        if self.value.ndim != 3:
            raise ValueError(
                f"Image must have 3 dimensions (C, H, W). Got {self.value.ndim}"
            )
        expected_channels = 1 if self.channel_order.is_1ch else 3
        if self.value.shape[0] != expected_channels:
            raise ValueError(
                f"Image must have {expected_channels} channel(s) in dim 0. Got {self.value.shape[0]}"
            )
        if self.value.dtype != torch.uint8:
            raise ValueError(f"Image must have uint8 dtype. Got {self.value.dtype}")

    @property
    def format(self) -> ImageFormat:
        """The image format (TORCH_TENSOR)."""
        return ImageFormat.TORCH_TENSOR

    @property
    @override
    def shape(self) -> tuple[int, int, int]:
        """
        Get the shape of the image.

        Returns
        -------
        tuple[int, int, int]: The shape (height, width, channels) of the image.
        """
        channels, height, width = self.value.shape
        return (height, width, channels)

    @property
    @override
    def width(self) -> int:
        """
        Get the width of the image.

        Returns
        -------
        int: The width of the image.
        """
        return int(self.value.shape[2])

    @property
    @override
    def height(self) -> int:
        """
        Get the height of the image.

        Returns
        -------
        int: The height of the image.
        """
        return int(self.value.shape[1])

    @property
    @override
    def size(self) -> tuple[int, int]:
        """
        Get the size of the image.

        Returns
        -------
        tuple[int, int]: The size (width, height), matching PIL Image.size.
        """
        return (self.width, self.height)

    @property
    @override
    def ch(self) -> int:
        """
        Get the number of channels of the image.

        Returns
        -------
        int: The number of channels of the image.
        """
        return int(self.value.shape[0])

    @override
    def crop(self, crop_slice: tuple[slice, slice]) -> TensorImageContainer:
        """
        Crop the image.

        Parameters
        ----------
        crop_slice: tuple[slice, slice]
            The slice to crop the image(y_slice, x_slice).
            example: (slice(100, 200), slice(300, 400))

        Returns
        -------
        TensorImageContainer: The cropped image container, on the same device.
        """
        y_slice, x_slice = crop_slice
        return TensorImageContainer(
            value=self.value[:, y_slice, x_slice].contiguous(),
            channel_order=self.channel_order,
        )

    @override
    def to_ch_swapped_image(
        self, output_order: ChannelOrder = ChannelOrder.BGR
    ) -> torch.Tensor:
        """
        Get the channel swapped image.

        RGB<->BGR stays on the tensor's own device (a channel flip); every
        other conversion goes through OpenCV on the CPU and back.

        An order already matching the stored one is answered with the stored
        tensor itself rather than a copy of it. A container is frozen and its
        tensor is never written through, so the copy protected nothing, while
        a decoded frame is megabytes and this is asked for it several times per
        frame -- once for the detector, once per downstream crop batch.

        Parameters
        ----------
        output_order: ChannelOrder
            The output channel order to get the image.

        Returns
        -------
        torch.Tensor: The channel swapped image, shaped (C, H, W).
        """
        if self.channel_order == output_order:
            return self.value
        if {self.channel_order, output_order} == _FLIPPABLE_ORDERS:
            return self.value.flip(0)
        hwc = self.to_array(output_order)
        chw = hwc.transpose(2, 0, 1) if hwc.ndim == 3 else hwc[None, ...]
        return torch.as_tensor(np.ascontiguousarray(chw), device=self.value.device)

    @override
    def to_PIL(self) -> Image.Image:
        """
        Get the PIL image(RGB ordered).

        Returns
        -------
        Image.Image: The PIL image.
        """
        match self.channel_order:
            case ChannelOrder.BGR:
                hwc = self.value.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
                return Image.fromarray(hwc, mode="RGB")
            case ChannelOrder.RGB:
                hwc = self.value.permute(1, 2, 0).contiguous().cpu().numpy()
                return Image.fromarray(hwc, mode="RGB")
            case ChannelOrder.GRAY:
                return Image.fromarray(
                    self.value[0].cpu().numpy(), mode=ChannelOrder.GRAY.pil_mode
                )
            case ChannelOrder.HSV:
                bgr = self.to_array(ChannelOrder.BGR)
                return Image.fromarray(bgr[..., ::-1], mode="RGB")
            case ChannelOrder.LAB:
                hwc = self.value.permute(1, 2, 0).contiguous().cpu().numpy()
                return Image.fromarray(hwc, mode=ChannelOrder.LAB.pil_mode)

    @override
    def to_array(self, ch_order: ChannelOrder = ChannelOrder.BGR) -> UInt8Image:
        """
        Get the array image.

        Returns
        -------
        UInt8Image: The array image.
        """
        hwc = self.value.permute(1, 2, 0).contiguous().cpu().numpy()
        if self.channel_order.is_1ch:
            hwc = hwc[..., 0]
        if ch_order == self.channel_order:
            return hwc.copy()
        convert = ch_order.cv2_array_converter(self.channel_order)
        return convert(hwc)

    @override
    def to_binary(self, threshold: int | float) -> BinaryImage:
        """
        Convert the image into a binary image using a threshold.

        Values greater than or equal to the threshold become 1, otherwise 0.
        If the input has 3 channels, it will be converted to gray first.
        """
        gray = self.to_array(ChannelOrder.GRAY)
        return BinaryImage(value=(gray >= threshold))

    @override
    def save(self, save_path: str) -> None:
        """
        Save with cv2.imwrite.

        Parameters
        ----------
        save_path: str
            Output path.
        """
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        to_write = (
            self.to_array(ChannelOrder.GRAY)
            if self.channel_order.is_1ch
            else self.to_array(ChannelOrder.BGR)
        )
        if not cv2.imwrite(save_path, to_write):
            raise OSError(f"Failed to write image to {save_path}")

    @override
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape={self.shape}, width={self.width}, "
            f"height={self.height}, channel_order={self.channel_order}, device={self.value.device})"
        )

    @classmethod
    @override
    def from_path(
        cls,
        image_path: str,
    ) -> TensorImageContainer:
        """
        Create a tensor image container from an image path.

        Not the hot path this container exists for -- decode a video via
        `VideoBackend.TORCHCODEC` for that -- so this reads through Pillow like
        `PILImageContainer.from_path` and lands on the CPU.

        Parameters
        ----------
        image_path: str
            The path to the image.

        Returns
        -------
        TensorImageContainer: The tensor image container, on the CPU.
        """
        image = Image.open(image_path).convert("RGB")
        chw = np.asarray(image).transpose(2, 0, 1)
        return cls(
            value=torch.as_tensor(np.ascontiguousarray(chw)),
            channel_order=ChannelOrder.RGB,
        )
