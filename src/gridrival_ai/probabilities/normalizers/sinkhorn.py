"""
Sinkhorn-Knopp normalizer for grid probabilities.

This module provides an implementation of the GridNormalizer interface
using the Sinkhorn-Knopp algorithm for biproportional fitting.
"""

import numpy as np

from gridrival_ai.probabilities.normalizers.base import GridNormalizer


class SinkhornNormalizer(GridNormalizer):
    """
    Sinkhorn-Knopp algorithm for normalizing grid probabilities.

    This normalizer enforces both row and column constraints by iteratively
    normalizing rows and columns until convergence.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of iterations, by default 20
    tolerance : float, optional
        Convergence tolerance for row and column sums, by default 1e-6

    Examples
    --------
    >>> import numpy as np
    >>> normalizer = SinkhornNormalizer()
    >>> # Create a 2x2 probability matrix
    >>> matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> normalized = normalizer.normalize(matrix)
    >>> # P1 and P2 now sum to 1.0 across drivers
    >>> normalized[:, 0].sum()
    1.0
    >>> normalized[:, 1].sum()
    1.0
    """

    def __init__(self, max_iter: int = 20, tolerance: float = 1e-6):
        """
        Initialize the normalizer.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations, by default 20
        tolerance : float, optional
            Convergence tolerance for row and column sums, by default 1e-6
        """
        self.max_iter = max_iter
        self.tolerance = tolerance

    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize matrix using Sinkhorn-Knopp algorithm.

        Parameters
        ----------
        matrix : np.ndarray
            2D probability matrix where rows represent drivers and columns positions

        Returns
        -------
        np.ndarray
            Normalized matrix satisfying both row and column constraints

        Raises
        ------
        ValueError
            If the matrix is not square or contains values outside the range [0, 1]
        """
        if matrix.size == 0:
            return matrix

        # Check that matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Input matrix must be square, got shape {matrix.shape}")

        # Check that all values are between 0 and 1
        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError("All values in the input matrix must be between 0 and 1")

        # Make a copy to avoid modifying the input
        result = matrix.copy()

        # Apply Sinkhorn-Knopp algorithm
        for _ in range(self.max_iter):
            # Normalize rows
            row_sums = result.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            result = result / row_sums

            # Normalize columns
            col_sums = result.sum(axis=0, keepdims=True)
            # Avoid division by zero
            col_sums[col_sums == 0] = 1.0
            result = result / col_sums

            # Check convergence
            row_deviation = np.abs(result.sum(axis=1) - 1.0).max()
            col_deviation = np.abs(result.sum(axis=0) - 1.0).max()

            if row_deviation < self.tolerance and col_deviation < self.tolerance:
                break

        return result
