
import numpy as np
import torch

from PIL import Image
from enum import Enum

ImageFormats = np.ndarray | Image.Image | torch.Tensor

class ImageFormat(Enum):
    """
    Image format.

    Attributes
    ----------
    ARRAY: np.ndarray
    PIL: Image.Image
    TORCH_TENSOR: torch.Tensor
    """
    ARRAY = "array"
    PIL = "pil"
    TORCH_TENSOR = "torch_tensor"
    # TF_TENSOR = "tensorflow_tensor"