from dataclasses import dataclass
from typing import Any

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
    labels: NDArray[np.integer[Any]]
        Per-pixel component id map from cv2.connectedComponentsWithStats.
    stats: NDArray[np.integer[Any]]
        Per-label statistics table (n_labels, 5), dtype typically int32.
    centroids: NDArray[np.floating[Any]]
        Per-label centroid coordinates (n_labels, 2), dtype typically float64.
    """
    n_labels: int
    labels: NDArray[np.integer[Any]]
    stats: NDArray[np.integer[Any]]
    centroids: NDArray[np.floating[Any]]
