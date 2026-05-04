from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass

from .connectivity import Connectivity
from .connected_components import ConnectedComponents

@dataclass
class BinaryImage:
    """
    A container class representing a binary image.

    Parameters
    ----------
    value: np.ndarray
        The binary image represented as a 2D array of shape (height, width).
    """
    value: np.ndarray

    def __post_init__(self):
        """
        Post-initialization validation.
        
        Raises
        ------
        ValueError: If the binary image is not a 2D array.
        ValueError: If the binary image contains values other than 0/1 or True/False.
        """
        if self.value.ndim != 2:
            raise ValueError(f"Binary image must have 2 dimensions, got {self.value.ndim}")

        unique_values = np.unique(self.value)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                f"Binary image must contain only 0/1 or True/False, got {unique_values}"
            )
        self.value = self.value.astype(bool)

    def connected_components(
        self, 
        connectivity: Connectivity = Connectivity.EIGHT
        ) -> ConnectedComponents:
        """
        Get the connected components of the binary image.

        Parameters
        ----------
        connectivity: int
            The connectivity of the connected components.

        Returns
        -------
        ConnectedComponents: The connected components of the binary image.
        """
        value = self.value.astype(np.uint8)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(value, connectivity=connectivity.value)
        return ConnectedComponents(
            n_labels=n_labels, 
            labels=labels, 
            stats=stats, 
            centroids=centroids
            )

    @property
    def sum(self) -> int:
        """
        Get the sum of the binary image.

        Returns
        -------
        int: The sum of the True(1) values in the binary image.
        """
        return np.sum(self.value)
    
    @property
    def mean(self) -> float:
        """
        Get the mean of the binary image.

        Returns
        -------
        float: The mean of the True(1) and False(0) values in the binary image.
        """
        return float(np.mean(self.value))
    
    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the binary image.

        Returns
        -------
        tuple[int, int]: The shape(height, width) of the binary image.
        """
        return self.value.shape
    
    @property
    def width(self) -> int:
        """
        Get the width of the binary image.

        Returns
        -------
        int: The width of the binary image.
        """
        return self.shape[1]
    
    @property
    def height(self) -> int:
        """
        Get the height of the binary image.

        Returns
        -------
        int: The height of the binary image.
        """
        return self.shape[0]
    
    @property
    def size(self) -> tuple[int, int]:
        """
        Get the size of the binary image.

        Returns
        -------
        tuple[int, int]: The size(width, height) of the binary image.
        """
        return (self.width, self.height)

    