"""
Base class for grid probability normalizers.

This module defines the abstract base class for normalizers that enforce
row and column constraints on grid probability distributions.
"""

from abc import ABC, abstractmethod

import numpy as np


class GridNormalizer(ABC):
    """
    Abstract base class for grid probability normalizers.

    Normalizers enforce constraints on grid probability distributions,
    such as ensuring that each driver's probabilities sum to 1.0 (row constraint)
    and each position's probabilities sum to 1.0 across all drivers (column constraint).

    The normalization is performed on a matrix where:
    - Rows represent drivers
    - Columns represent positions
    - Each entry represents the probability of a driver finishing in a position
    """

    @abstractmethod
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize a probability matrix.

        Parameters
        ----------
        matrix : np.ndarray
            2D probability matrix where rows represent drivers and columns positions

        Returns
        -------
        np.ndarray
            Normalized matrix satisfying both row and column constraints
        """
        pass
