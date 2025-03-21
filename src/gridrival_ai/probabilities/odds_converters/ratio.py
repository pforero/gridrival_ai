"""
Odds ratio method implementation for odds conversion.

This module implements the odds ratio method from Cheung (2015) for converting
betting odds to probabilities.
"""

import warnings
from typing import List

import numpy as np
from scipy.optimize import minimize

from gridrival_ai.probabilities.odds_converters.base import OddsConverter


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

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
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

        # Define objective function for optimization
        def objective(or_value: float) -> float:
            probs = raw_probs / (or_value + raw_probs - (or_value * raw_probs))
            return abs(target_sum - probs.sum())

        # Optimize to find best odds ratio value
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
