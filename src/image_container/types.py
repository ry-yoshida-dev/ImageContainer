"""NumPy array type aliases for image pixel data."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

type UInt8Image = NDArray[np.uint8]
type ImageArray = NDArray[np.integer[Any] | np.floating[Any]]
type BinaryArray = NDArray[np.bool_]
