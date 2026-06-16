
import torch

from PIL import Image
from enum import Enum

from .types import UInt8Image

ImageFormats = UInt8Image | Image.Image | torch.Tensor

class ImageFormat(Enum):
    """
    Image format.

    Attributes
    ----------
    ARRAY: UInt8Image
    PIL: Image.Image
    TORCH_TENSOR: torch.Tensor
    """
    ARRAY = "array"
    PIL = "pil"
    TORCH_TENSOR = "torch_tensor"
    # TF_TENSOR = "tensorflow_tensor"