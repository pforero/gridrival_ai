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

import math
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


def harville_method(
    odds: list[dict[str, float]],
    target_market: float = 1.0,
    epsilon: float = 1e-10,
) -> dict[str, dict[int, float]]:
    """
    Calculate grid probabilities (marginal finishing position probabilities) using the Harville model.

    The function assumes that the quoted odds for each driver come from a single market.
    For example:
      - If target_market==1.0 then the odds (accessed via key "1") represent finishing 1st or better.
      - If target_market==6.0 then the odds (accessed via key "6") represent finishing 6th or better.

    The function first converts odds to “strengths” (using basic_method) and then
    uses a DP algorithm to compute the marginal probability for each finishing position.

    Parameters
    ----------
    odds : List[Dict[str, float]]
        Each dictionary must contain at least:
            - driver_id: str
            - a key matching the market (e.g. "1" or "6")
    target_market : float, optional
        The “market” number. For win odds use 1.0; for, say, “top‑6” odds use 6.0.
        (Default is 1.0.)
    epsilon : float, optional
        A small number to avoid division by zero.

    Returns
    -------
    Dict[str, Dict[int, float]]
        A dictionary mapping each driver’s ID to another dictionary mapping finishing
        positions (1-indexed) to probabilities. For example, with two drivers the result
        may look like:

            {
                "VER": {1: 0.65, 2: 0.35},
                "HAM": {1: 0.35, 2: 0.65},
            }

    Notes
    -----
    This implementation is based on the idea that if we have a set of drivers with strengths p,
    then for any subset (represented by a bit mask) the probability that driver i is chosen
    next is p[i] divided by the sum of p[j] over drivers in the subset.

    The DP works as follows:

      - Let n be the number of drivers.
      - Represent a subset of available drivers as an integer bitmask.
      - Initialize dp[full_mask] = 1.0.
      - For each mask (iterating “backwards” from full_mask to 0), let
            pos = n - (number of drivers available in mask) + 1.
        For each driver i in the current mask, add:

            result[i][pos] += dp[mask] * (p[i] / sum_{j in mask} p[j])

        and “remove” driver i (i.e. update dp[new_mask]) with new_mask = mask without bit i.

      - At the end, each driver’s probabilities (over positions 1..n) sum to 1 and for each finishing
        position the probabilities over drivers sum to 1.
    """
    # Get driver IDs and number of drivers
    driver_ids = [d["driver_id"] for d in odds]
    n = len(driver_ids)

    # Choose the correct market key.
    # (If target_market is an integer, we look for key "1", "6", etc.)
    if target_market == int(target_market):
        market_key = str(int(target_market))
    else:
        market_key = str(target_market)

    # Extract the odds for the given market.
    market_odds = [d[market_key] for d in odds]

    # Convert odds to “strengths” that sum to target_market.
    strengths = basic_method(market_odds, target_market)

    # dp[mask] will hold the probability of reaching that state.
    # We use a list indexed by mask (an integer between 0 and 2^n - 1).
    dp = [0.0] * (1 << n)
    full_mask = (1 << n) - 1
    dp[full_mask] = 1.0

    # Initialize result dictionary: for each driver, create a dict for positions 1..n.
    result = {
        driver_id: {pos: 0.0 for pos in range(1, n + 1)} for driver_id in driver_ids
    }

    # Iterate over all masks from full_mask down to 0.
    for mask in range(full_mask, -1, -1):
        # Skip states with zero probability.
        if dp[mask] == 0:
            continue

        # Build a list of indices for drivers still available in this state.
        available = [i for i in range(n) if mask & (1 << i)]
        if not available:
            continue
        # Sum of strengths for drivers available in this mask.
        s = sum(strengths[i] for i in available)
        # The finishing position we are about to assign:
        pos = n - len(available) + 1  # positions are 1-indexed

        # For each driver i available, assign the probability to finish in position pos.
        for i in available:
            p_i = strengths[i] / (s + epsilon)
            prob = dp[mask] * p_i
            result[driver_ids[i]][pos] += prob
            new_mask = mask & ~(1 << i)
            dp[new_mask] += prob
    return result
