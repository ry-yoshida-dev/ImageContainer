from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ConnectedComponents:
    """
    A class to represent connected components.

    Attributes
    ----------
    n_labels: int
        The number of connected components.
    labels: NDArray[np.int32]
        Per-pixel component id map from cv2.connectedComponentsWithStats.
    stats: NDArray[np.int32]
        Per-label statistics table (n_labels, 5).
    centroids: NDArray[np.float64]
        Per-label centroid coordinates (n_labels, 2).
    """

    n_labels: int
    labels: NDArray[np.int32]
    stats: NDArray[np.int32]
    centroids: NDArray[np.float64]
