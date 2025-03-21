"""
Basic odds conversion implementation.

This module provides a simple implementation of the OddsConverter interface
using direct 1/odds transformation with normalization.
"""

from typing import List

import numpy as np

from gridrival_ai.probabilities.odds_converters.base import OddsConverter


class BasicConverter(OddsConverter):
    """
    Basic method for converting odds to probabilities.

    Uses simple multiplicative normalization by taking the reciprocal of
    each odd and normalizing to the target sum.

    Examples
    --------
    >>> converter = BasicConverter()
    >>> odds = [1.5, 3.0, 6.0]
    >>> probs = converter.convert(odds)
    >>> probs
    array([0.57142857, 0.28571429, 0.14285714])
    """

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities using basic method.

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

        # Normalize to target sum
        return raw_probs * (target_sum / raw_probs.sum())
