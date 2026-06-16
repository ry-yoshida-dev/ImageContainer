
import torch

from PIL import Image
from enum import Enum

from .types import ImageArray

ImageFormats = ImageArray | Image.Image | torch.Tensor

class ImageFormat(Enum):
    """
    Image format.

    Attributes
    ----------
    ARRAY: ImageArray
    PIL: Image.Image
    TORCH_TENSOR: torch.Tensor
    """
    ARRAY = "array"
    PIL = "pil"
    TORCH_TENSOR = "torch_tensor"
    # TF_TENSOR = "tensorflow_tensor"