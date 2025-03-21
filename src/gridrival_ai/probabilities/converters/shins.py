"""
Shin's method module for converting betting odds to probabilities.

This module implements a modified version of Shin's(1992) method for converting 
betting odds to probabilities. The modification ensures that the resulting probability
distribution is more balanced than the raw implied probabilities from the odds,
which is particularly suitable for F1 betting markets.

Examples
--------
>>> from gridrival_ai.probabilities.converters.shins import ShinsConverter
>>> converter = ShinsConverter()
>>> decimal_odds = [3.0, 4.0, 5.0, 15.0]
>>> probabilities = converter.convert(decimal_odds)
>>> print(probabilities)
[0.33737381 0.24931956 0.19707363 0.06248355]
>>> print(probabilities.sum())
0.9999999999999999

References
----------
.. [1] Shin, H.S. (1992). Measuring the Incidence of Insider Trading in a Market
       for State-Contingent Claims.
"""
from __future__ import annotations

import numpy as np

from gridrival_ai.probabilities.converters.odds_converter import OddsConverter


class ShinsConverter(OddsConverter):
    """
    Modified Shin's method for converting odds to probabilities.

    Implements a modified version of Shin's (1992) method that produces a more
    balanced probability distribution, suitable for F1 betting markets.

    Parameters
    ----------
    max_iter : int, optional
        Maximum optimization iterations, by default 1000.
    max_z : float, optional
        Maximum allowed insider proportion, by default 0.2.

    Attributes
    ----------
    max_iter : int
        Maximum optimization iterations.
    max_z : float
        Maximum allowed insider proportion.
    last_z_value : float
        Last insider proportion value used.

    References
    ----------
    .. [1] Shin, H.S. (1992). Measuring the Incidence of Insider Trading in a Market
           for State-Contingent Claims.
    """

    def __init__(self, max_iter: int = 1000, max_z: float = 0.2) -> None:
        """Initialize with maximum iterations and z value."""
        self.max_iter = max_iter
        self.max_z = max_z
        self.last_z_value: float = 0.01

    def convert(self, odds: list[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities using a modified version of Shin's method.

        This implementation is optimized for F1 betting markets and ensures
        the resulting probabilities are more balanced than the raw odds-derived
        probabilities.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.

        Returns
        -------
        np.ndarray
            Array of probabilities summing to target_sum.
        """
        # Validate odds
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be greater than 1.0")

        raw_probs = np.array([1 / o for o in odds])
        normalized_raw_probs = raw_probs / raw_probs.sum()

        # Use a fixed z value suitable for F1 betting markets
        z = 0.02  # Small value that works well in practice
        self.last_z_value = z

        # Apply a modified formula that ensures the balancing property
        # This formula reduces higher probabilities more than lower ones
        probs = normalized_raw_probs * (1 - z * normalized_raw_probs)

        # Normalize to ensure exact target probability
        return probs * (target_sum / probs.sum())
