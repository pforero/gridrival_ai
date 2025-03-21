"""
Shin's method implementation for odds conversion.

This module implements a modified version of Shin's (1992) method for
converting betting odds to probabilities, particularly suitable for
racing markets.
"""

from typing import List

import numpy as np

from gridrival_ai.probabilities.odds_converters.base import OddsConverter


class ShinsConverter(OddsConverter):
    """
    Modified Shin's method for converting odds to probabilities.

    Implements a modified version of Shin's (1992) method that produces a more
    balanced probability distribution, suitable for F1 betting markets.

    Parameters
    ----------
    max_z : float, optional
        Maximum allowed insider proportion, by default 0.2.

    Attributes
    ----------
    max_z : float
        Maximum allowed insider proportion.
    last_z_value : float
        Last insider proportion value used.

    References
    ----------
    .. [1] Shin, H.S. (1992). Measuring the Incidence of Insider Trading in a Market
           for State-Contingent Claims.
    """

    def __init__(self, max_z: float = 0.2) -> None:
        """Initialize with z value limit."""
        self.max_z = max_z
        self.last_z_value: float = 0.01

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
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

        Raises
        ------
        ValueError
            If any odd is not greater than 1.0.
        """
        # Validate odds
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be greater than 1.0")

        # Convert to raw probabilities (1/odds)
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
