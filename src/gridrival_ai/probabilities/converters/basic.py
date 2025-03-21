from __future__ import annotations

import numpy as np

from gridrival_ai.probabilities.converters.odds_converter import OddsConverter


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

    def convert(self, odds: list[float], target_sum: float = 1.0) -> np.ndarray:
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
        """
        # Validate odds
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be greater than 1.0")

        raw_probs = np.array([1 / o for o in odds])
        return raw_probs * (target_sum / raw_probs.sum())
