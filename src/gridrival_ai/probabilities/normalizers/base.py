"""
Base class for grid probability normalizers.

This module defines the abstract base class for normalizers that enforce
row and column constraints on grid probability distributions.
"""

from abc import ABC, abstractmethod

from gridrival_ai.probabilities.distributions import SessionDistribution


class GridNormalizer(ABC):
    """
    Abstract base class for grid probability normalizers.

    Normalizers enforce constraints on grid probability distributions,
    such as ensuring that each driver's probabilities sum to 1.0 (row constraint)
    and each position's probabilities sum to 1.0 across all drivers (column constraint).
    """

    @abstractmethod
    def normalize(self, distributions: SessionDistribution) -> SessionDistribution:
        """
        Normalize a set of position distributions.

        Parameters
        ----------
        distributions : SessionDistribution
            Dictionary mapping driver IDs to position distributions

        Returns
        -------
        SessionDistribution
            Normalized distributions
        """
        pass
