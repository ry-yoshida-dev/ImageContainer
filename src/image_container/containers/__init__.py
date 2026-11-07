from .array import ArrayImageContainer
from .pil import PILImageContainer
from .tensor import TensorImageContainer

type ImageContainerType = ArrayImageContainer | PILImageContainer | TensorImageContainer

__all__ = [
    "ArrayImageContainer",
    "ImageContainerType",
    "PILImageContainer",
    "TensorImageContainer",
]
