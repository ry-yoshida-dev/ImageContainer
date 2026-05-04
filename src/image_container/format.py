
import numpy as np
import torch

from PIL import Image
from typing import Union
from enum import Enum


ImageFormats = Union[np.ndarray, Image.Image, torch.Tensor] # tf.tensorflow

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