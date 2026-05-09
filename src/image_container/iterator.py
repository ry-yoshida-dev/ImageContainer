from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path



import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener  # type: ignore[import-untyped]

from .ch_order import ChannelOrder
from .containers.array import ArrayImageContainer
from .containers.pil import PILImageContainer
from .extention import Extention
from .format import ImageFormat


SupportedContainer = ArrayImageContainer | PILImageContainer


@dataclass(frozen=True)
class ImageIterator:
    """
    Iterate image files under one directory and yield containers.

    Attributes
    ----------
    image_format : ImageFormat
        Output container format.
    paths : list[str]
        Matched image file paths.
    """

    image_format: ImageFormat
    paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate initialization parameters.
        """
        if len(self.paths) == 0:
            raise ValueError("paths must contain at least one file path.")

    @classmethod
    def from_dir(
        cls,
        dir_path: str,
        image_format: ImageFormat,
    ) -> ImageIterator:
        """
        Build an iterator by collecting valid image paths from a directory.

        Parameters
        ----------
        dir_path : str
            Target directory path.
        image_format : ImageFormat
            Output container format.

        Returns
        -------
        ImageIterator
            Configured iterator with collected file paths.
        """
        target_dir = Path(dir_path)
        if not target_dir.is_dir():
            raise ValueError(f"Directory does not exist: {dir_path}")

        supported_suffixes = {f".{ext.value.lower()}" for ext in Extention}
        paths = [
            str(path)
            for path in sorted(target_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ]
        if len(paths) == 0:
            raise ValueError(f"No image files found in directory: {dir_path}")

        return cls(image_format=image_format, paths=paths)

    def __iter__(self) -> Iterator[SupportedContainer]:
        """
        Iterate over all collected image paths.

        Returns
        -------
        Iterator[ImageContainer]
            Iterator yielding loaded image containers.
        """
        for image_path in self.paths:
            yield self._create_container(image_path=image_path)

    def __len__(self) -> int:
        """
        Get number of matched image paths.

        Returns
        -------
        int
            Number of image file paths.
        """
        return len(self.paths)

    def _create_container(self, image_path: str) -> SupportedContainer:
        """
        Create container from image path according to target format.

        Parameters
        ----------
        image_path : str
            Input image file path.

        Returns
        -------
        ImageContainer
            Loaded image container.
        """
        path = Path(image_path)
        is_heic = path.suffix.lower() == f".{Extention.HEIC.value}"
        if is_heic:
            return self._create_heic_container(image_path=image_path)

        match self.image_format:
            case ImageFormat.ARRAY:
                return ArrayImageContainer.from_path(image_path=image_path)
            case ImageFormat.PIL:
                return PILImageContainer.from_path(image_path=image_path)
            case ImageFormat.TORCH_TENSOR:
                raise NotImplementedError(
                    "TORCH_TENSOR format is not implemented yet."
                )

    def _create_heic_container(self, image_path: str) -> SupportedContainer:
        """
        Create a container from a HEIC image path via pillow-heif.

        Parameters
        ----------
        image_path : str
            Input HEIC file path.

        Returns
        -------
        SupportedContainer
            Loaded image container.
        """
        register_heif_opener()
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        match self.image_format:
            case ImageFormat.PIL:
                return PILImageContainer(
                    value=rgb_image,
                    channel_order=ChannelOrder.RGB,
                )
            case ImageFormat.ARRAY:
                rgb_array = np.array(rgb_image)
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                return ArrayImageContainer(
                    value=bgr_array,
                    channel_order=ChannelOrder.BGR,
                )
            case ImageFormat.TORCH_TENSOR:
                raise NotImplementedError(
                    "TORCH_TENSOR format is not implemented yet."
                )
