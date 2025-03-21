"""
Power method implementation for odds conversion.

This module implements the power method for converting betting odds to
probabilities, which models true probabilities as a power function of raw
probabilities.
"""

import warnings
from typing import List

import numpy as np
from scipy.optimize import minimize

from gridrival_ai.probabilities.odds_converters.base import OddsConverter


class PowerConverter(OddsConverter):
    """
    Power method for converting odds to probabilities.

    Models true probabilities as a power function of raw probabilities:
    p[i] = r[i]^(1/k)
    where k is optimized to achieve the target probability sum.

    Parameters
    ----------
    max_iter : int, optional
        Maximum optimization iterations, by default 1000.

    Attributes
    ----------
    max_iter : int
        Maximum optimization iterations.
    last_k_value : float
        Last optimized power parameter value.
    """

    def __init__(self, max_iter: int = 1000) -> None:
        """Initialize with maximum iterations."""
        self.max_iter = max_iter
        self.last_k_value: float = 1.0

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities using power method.

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
        def objective(k: float) -> float:
            if k <= 0:
                return float("inf")
            probs = raw_probs ** (1 / k)
            return abs(target_sum - probs.sum())

        # Try different optimization approaches
        methods = [
            # First try Nelder-Mead
            {
                "method": "Nelder-Mead",
                "options": {"maxiter": self.max_iter, "xatol": 1e-6, "fatol": 1e-6},
            },
            # If that fails, try L-BFGS-B
            {
                "method": "L-BFGS-B",
                "bounds": [(0.5, 2.0)],
                "options": {
                    "maxiter": self.max_iter,
                    "ftol": 1e-6,
                    "gtol": 1e-5,
                    "maxls": 50,
                },
            },
        ]

        best_result = None
        min_obj_value = float("inf")

        # Try each optimization method
        for method_params in methods:
            try:
                result = minimize(objective, x0=self.last_k_value, **method_params)

                # Keep the best result based on objective value
                if result.fun < min_obj_value:
                    min_obj_value = result.fun
                    best_result = result

                # If we got a good enough result, stop trying
                if min_obj_value < 1e-6:
                    break

            except Exception:
                continue  # Try next method if this one fails

        if best_result is None:
            warnings.warn("All optimization methods failed, using fallback solution")
            k = 1.0  # Fallback to no transformation
        else:
            k = float(best_result.x[0])

        self.last_k_value = k

        # Calculate final probabilities using optimal k value
        probs = raw_probs ** (1 / k)

        # Normalize to ensure exact target probability
        return probs * (target_sum / probs.sum())
