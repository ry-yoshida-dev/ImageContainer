"""NumPy array type aliases for image pixel data."""

import numpy as np

# A concrete integer shape keeps dimensions typed as int instead of the Any of NDArray.
type UInt8Image = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
type ImageArray = np.ndarray[tuple[int, ...], np.dtype[np.integer | np.floating]]
type BinaryArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
