"""
Odds ratio conversion module for converting raw betting odds to probabilities.

This module implements the odds ratio method from Cheung (2015) for converting
betting odds to probabilities. The method models the relationship between true
probabilities and raw probabilities using an odds ratio function:

    OR = p(1-r)/(r(1-p))

where p is the true probability and r is the raw probability derived from betting odds.

The module provides the OddsRatioConverter class which handles the optimization
process to find the best odds ratio value that ensures the resulting probabilities
sum to a target value (typically 1.0).

Examples
--------
>>> from gridrival_ai.probabilities.converters.ratio import OddsRatioConverter
>>> converter = OddsRatioConverter()
>>> decimal_odds = [3.0, 4.0, 5.0, 15.0]
>>> probabilities = converter.convert(decimal_odds)
>>> print(probabilities)
[0.32876712 0.24657534 0.19726027 0.06575342]
>>> print(probabilities.sum())
0.9999999

References
----------
.. [1] Cheung (2015). Fixed-odds betting and traditional odds.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from gridrival_ai.probabilities.converters.odds_converter import OddsConverter


class OddsRatioConverter(OddsConverter):
    """
    Odds ratio method for converting odds to probabilities.

    Implements the odds ratio method from Cheung (2015) which models the
    relationship between true probabilities and raw probabilities using
    an odds ratio function: OR = p(1-r)/(r(1-p))
    where p is true probability and r is raw probability.

    Parameters
    ----------
    max_iter : int, optional
        Maximum optimization iterations, by default 1000.

    Attributes
    ----------
    max_iter : int
        Maximum optimization iterations.
    last_or_value : float
        Last optimized odds ratio value.

    References
    ----------
    .. [1] Cheung (2015). Fixed-odds betting and traditional odds.
    """

    def __init__(self, max_iter: int = 1000) -> None:
        """Initialize with maximum iterations."""
        self.max_iter = max_iter
        self.last_or_value: float = 1.0

    def convert(self, odds: list[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities using odds ratio method.

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

        def objective(or_value: float) -> float:
            probs = raw_probs / (or_value + raw_probs - (or_value * raw_probs))
            return abs(target_sum - probs.sum())

        result = minimize(
            objective,
            x0=self.last_or_value,
            method="Nelder-Mead",
            options={"maxiter": self.max_iter},
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")

        self.last_or_value = float(result.x[0])

        # Calculate final probabilities using optimal OR value
        probs = raw_probs / (
            self.last_or_value + raw_probs - (self.last_or_value * raw_probs)
        )

        # Normalize to ensure exact target probability
        return probs * (target_sum / probs.sum())
