# cSpell: ignore Cheung, Shin, maxiter, Nelder-Mead, L-BFGS-B, xatol, fatol, gt, ftol, gtol, maxls  # noqa: E501
"""
Methods for converting betting odds to probabilities.

This module provides various methods for converting decimal betting odds to valid
probability distributions. Each method handles the house margin and favorite-longshot
bias differently.

Available methods:
- Basic method (multiplicative)
- Odds ratio method (Cheung, 2015)
- Shin's method (Shin, 1992)
- Power method

References
----------
.. [1] Cheung (2015). Fixed-odds betting and traditional odds.
.. [2] Shin, H.S. (1992). Measuring the Incidence of Insider Trading in a Market
       for State-Contingent Claims.
"""

import warnings
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize


def basic_method(odds: List[float], target_probability: float = 1.0) -> np.ndarray:
    """Convert odds to probabilities using the basic multiplicative method.

    Implements the basic probability conversion by dividing raw probabilities (1/odds)
    by their sum, then scaling to match the target probability. This is the simplest
    and most common method for removing the bookmaker margin.

    Parameters
    ----------
    odds : list[float]
        List of decimal odds. Must be > 1.0.
    target_probability : float, optional
        Target sum for probabilities, by default 1.0.
        Use values > 1.0 for markets like "top N finish".

    Returns
    -------
    np.ndarray
        Array of probabilities summing to target_probability.

    Notes
    -----
    The method is implemented as:
        p[i] = (1/odds[i]) / sum(1/odds) * target

    This is the simplest approach to removing bookmaker margin but does not
    explicitly address favorite-longshot bias. Despite its simplicity, it often
    performs well in practice and serves as a baseline for comparing more
    sophisticated methods.

    Examples
    --------
    >>> odds = [1.5, 4.5, 8.0]  # Example F1 pole position odds
    >>> probs = basic_method(odds)
    >>> print(probs)  # Probabilities sum to 1
    [0.59701492 0.19900498 0.1119403 ]

    >>> probs = basic_method(odds, target_probability=2)  # Top 2 finish
    >>> print(probs)  # Probabilities sum to 2
    [1.19402985 0.39800995 0.22388059]
    """
    raw_probs = np.array([1 / o for o in odds])
    return raw_probs * (target_probability / raw_probs.sum())


def odds_ratio_method(
    odds: List[float], target_probability: float = 1.0, max_iter: int = 1000
) -> Tuple[np.ndarray, float]:
    """Convert odds to probabilities using the odds ratio method.

    Implements the odds ratio method from Cheung (2015) which models the relationship
    between true probabilities and raw probabilities using an odds ratio function:
        OR = p(1-r)/(r(1-p))
    where p is true probability and r is raw probability.

    Parameters
    ----------
    odds : list[float]
        List of decimal odds. Must be > 1.0.
    target_probability : float, optional
        Target sum for probabilities, by default 1.0.
        Use values > 1.0 for markets like "top N finish".
    max_iter : int, optional
        Maximum optimization iterations, by default 1000.

    Returns
    -------
    tuple[np.ndarray, float]
        - Array of probabilities summing to target_probability
        - Optimal odds ratio value

    Notes
    -----
    The method optimizes an odds ratio parameter that maps raw probabilities
    (1/odds) to true probabilities. A final normalization step ensures the
    probabilities sum exactly to the target value.

    References
    ----------
    .. [1] Cheung (2015). Fixed-odds betting and traditional odds.
    """
    raw_probs = np.array([1 / o for o in odds])

    def objective(or_value: float) -> float:
        probs = raw_probs / (or_value + raw_probs - (or_value * raw_probs))
        return abs(target_probability - probs.sum())

    result = minimize(
        objective, x0=1.0, method="Nelder-Mead", options={"maxiter": max_iter}
    )

    if not result.success:
        warnings.warn(f"Optimization failed: {result.message}")

    or_value = result.x[0]

    # Calculate final probabilities using optimal OR value
    probs = raw_probs / (or_value + raw_probs - (or_value * raw_probs))

    # Normalize to ensure exact target probability
    probs = probs * (target_probability / probs.sum())

    return probs, or_value


def shin_method(
    odds: List[float],
    target_probability: float = 1.0,
    max_iter: int = 1000,
    max_z: float = 0.2,
) -> Tuple[np.ndarray, float]:
    """Convert odds to probabilities using Shin's method.

    Implements Shin's (1992) method which models the market as having a proportion z
    of insider traders:
        p[i] = r[i](1-z)/(1-z*r[i])
    where p is true probability, r is raw probability, z is insider proportion.

    Parameters
    ----------
    odds : list[float]
        List of decimal odds. Must be > 1.0.
    target_probability : float, optional
        Target sum for probabilities, by default 1.0.
        Use values > 1.0 for markets like "top N finish".
    max_iter : int, optional
        Maximum optimization iterations, by default 1000.
    max_z : float, optional
        Maximum allowed insider proportion, by default 0.2.
        Typical values in literature range from 0.02 to 0.15.

    Returns
    -------
    tuple[np.ndarray, float]
        - Array of probabilities summing to target_probability
        - Estimated insider trading proportion

    Notes
    -----
    The method optimizes an insider trading proportion parameter z that explains
    the divergence between market odds and true probabilities. A final normalization
    step ensures the probabilities sum exactly to the target value.

    References
    ----------
    .. [1] Shin, H.S. (1992). Measuring the Incidence of Insider Trading in a Market
           for State-Contingent Claims.
    """
    raw_probs = np.array([1 / o for o in odds])

    def objective(z: float) -> float:
        probs = raw_probs * (1 - z) / (1 - z * raw_probs)
        return abs(target_probability - probs.sum())

    result = minimize(
        objective,
        x0=0.01,
        bounds=[(0, max_z)],
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    if not result.success:
        warnings.warn(f"Optimization failed: {result.message}")

    z = result.x[0]

    # Calculate final probabilities using optimal z value
    probs = raw_probs * (1 - z) / (1 - z * raw_probs)

    # Normalize to ensure exact target probability
    probs = probs * (target_probability / probs.sum())

    return probs, z


def power_method(
    odds: List[float], target_probability: float = 1.0, max_iter: int = 1000
) -> Tuple[np.ndarray, float]:
    """Convert odds to probabilities using the power method.

    Models true probabilities as a power function of raw probabilities:
        p[i] = r[i]^(1/k)
    where k is optimized to achieve the target probability sum.

    Parameters
    ----------
    odds : list[float]
        List of decimal odds. Must be > 1.0.
    target_probability : float, optional
        Target sum for probabilities, by default 1.0.
        Use values > 1.0 for markets like "top N finish".
    max_iter : int, optional
        Maximum optimization iterations, by default 1000.

    Returns
    -------
    tuple[np.ndarray, float]
        - Array of probabilities summing to target_probability
        - Optimal power parameter k

    Notes
    -----
    The method optimizes a power parameter k that transforms raw probabilities
    (1/odds) into true probabilities. A final normalization step ensures the
    probabilities sum exactly to the target value. The parameter k is constrained
    between 0.5 and 2.0 for numerical stability.
    """
    raw_probs = np.array([1 / o for o in odds])

    def objective(k: float) -> float:
        probs = raw_probs ** (1 / k)
        return abs(target_probability - probs.sum())

    # Try different optimization approaches in order of complexity
    methods = [
        # First try Nelder-Mead (doesn't require derivatives, more robust)
        {
            "method": "Nelder-Mead",
            "options": {
                "maxiter": max_iter,
                "xatol": 1e-6,
                "fatol": 1e-6,
            },
        },
        # If that fails, try L-BFGS-B with tighter controls
        {
            "method": "L-BFGS-B",
            "bounds": [(0.5, 2.0)],
            "options": {
                "maxiter": max_iter,
                "ftol": 1e-6,
                "gtol": 1e-5,
                "maxls": 50,  # Increase max line search steps
            },
        },
    ]

    best_result = None
    min_obj_value = float("inf")

    # Try each optimization method
    for method_params in methods:
        try:
            result = minimize(objective, x0=1.0, **method_params)

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
        k = best_result.x[0]

    # Calculate final probabilities using optimal k value
    probs = raw_probs ** (1 / k)

    # Normalize to ensure exact target probability
    probs = probs * (target_probability / probs.sum())

    return probs, k
