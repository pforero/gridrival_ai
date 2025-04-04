"""
Odds conversion base class for GridRival AI.

This module defines the abstract base class for converting betting odds
to probabilities, providing a consistent interface for all converter
implementations.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from gridrival_ai.probabilities.distributions.position import PositionDistribution


class OddsConverter(ABC):
    """
    Abstract base class for odds-to-probabilities converters.

    This class defines the interface for converting betting odds to
    probabilities. Concrete implementations handle different methods
    for margin removal and bias adjustment.
    """

    @abstractmethod
    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.
            Use values > 1.0 for markets like "top N finish".

        Returns
        -------
        np.ndarray
            Array of probabilities summing to target_sum.
        """
        pass
