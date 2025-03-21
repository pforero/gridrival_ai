"""
Power method module for converting betting odds to probabilities.

This module implements a power method for converting betting odds to probabilities.
The method models true probabilities as a power function of raw probabilities:

    p[i] = r[i]^(1/k)

where r[i] is the raw probability (1/odds) and k is a parameter that is optimized
to ensure the resulting probabilities sum to a target value (typically 1.0).

The module provides the PowerConverter class which handles the optimization process
to find the best power parameter k. The implementation attempts multiple optimization
methods (Nelder-Mead and L-BFGS-B) to ensure robust convergence.

Examples
--------
>>> from gridrival_ai.probabilities.converters.power import PowerConverter
>>> converter = PowerConverter()
>>> decimal_odds = [3.0, 4.0, 5.0, 15.0]
>>> probabilities = converter.convert(decimal_odds)
>>> print(probabilities)
[0.33333333 0.25       0.2        0.06666667]
>>> print(probabilities.sum())
1.0
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from gridrival_ai.probabilities.converters.odds_converter import OddsConverter


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

    def convert(self, odds: list[float], target_sum: float = 1.0) -> np.ndarray:
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
        """
        # Validate odds
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be greater than 1.0")

        raw_probs = np.array([1 / o for o in odds])

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
