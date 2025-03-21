"""
Sinkhorn-Knopp normalizer for grid probabilities.

This module provides an implementation of the GridNormalizer interface
using the Sinkhorn-Knopp algorithm for biproportional fitting.
"""

import numpy as np

from gridrival_ai.probabilities.distributions import (
    PositionDistribution,
    SessionDistribution,
)
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
    >>> normalizer = SinkhornNormalizer()
    >>> driver_distributions = {
    ...     "VER": PositionDistribution({1: 0.7, 2: 0.3}),
    ...     "HAM": PositionDistribution({1: 0.4, 2: 0.6})
    ... }
    >>> session = SessionDistribution(driver_distributions, "race")
    >>> normalized = normalizer.normalize(session)
    >>> # P1 and P2 now sum to 1.0 across drivers
    >>> (
    >>>     normalized.get_driver_distribution("VER")[1]
    >>>     + normalized.get_driver_distribution("HAM")[1]
    >>> )
    1.0
    >>> (
    >>>     normalized.get_driver_distribution("VER")[2]
    >>>     + normalized.get_driver_distribution("HAM")[2]
    >>> )
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

    def normalize(self, session: SessionDistribution) -> SessionDistribution:
        """
        Normalize distributions using Sinkhorn-Knopp algorithm.

        Parameters
        ----------
        session : SessionDistribution
            Session distribution containing driver position distributions

        Returns
        -------
        SessionDistribution
            Normalized session distribution satisfying both row and column constraints
        """
        distributions = session.driver_distributions
        session_type = session.session_type

        if not distributions:
            return SessionDistribution({}, session_type)

        # Get all drivers
        all_drivers = list(distributions.keys())

        # Get all positions (assuming all drivers have the same positions available)
        all_positions = set()
        for dist in distributions.values():
            all_positions.update(dist.position_probs.keys())
        all_positions = sorted(all_positions)

        # Create matrix representation
        matrix = np.zeros((len(all_drivers), len(all_positions)))
        pos_to_idx = {pos: idx for idx, pos in enumerate(all_positions)}

        for i, driver_id in enumerate(all_drivers):
            for pos, prob in distributions[driver_id].position_probs.items():
                j = pos_to_idx[pos]
                matrix[i, j] = prob

        # Apply Sinkhorn-Knopp algorithm
        for _ in range(self.max_iter):
            # Normalize rows
            row_sums = matrix.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            matrix = matrix / row_sums

            # Normalize columns
            col_sums = matrix.sum(axis=0, keepdims=True)
            # Avoid division by zero
            col_sums[col_sums == 0] = 1.0
            matrix = matrix / col_sums

            # Check convergence
            row_deviation = np.abs(matrix.sum(axis=1) - 1.0).max()
            col_deviation = np.abs(matrix.sum(axis=0) - 1.0).max()

            if row_deviation < self.tolerance and col_deviation < self.tolerance:
                break

        # Convert back to PositionDistribution objects
        result = {}
        for i, driver_id in enumerate(all_drivers):
            position_probs = {}
            for j, pos in enumerate(all_positions):
                if matrix[i, j] > 0:
                    position_probs[pos] = matrix[i, j]

            # Create PositionDistribution with validation turned off
            # since the values might be very close to 1.0 but not exactly
            result[driver_id] = PositionDistribution(position_probs, _validate=False)

        # Return a new SessionDistribution with the normalized driver distributions
        return SessionDistribution(result, session_type, _validate=False)
