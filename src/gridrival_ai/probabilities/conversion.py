"""
Odds conversion functions for probability distributions.

This module provides a consolidated set of tools for converting betting odds
to probability distributions. It implements various methods for handling
bookmaker margins and favorite-longshot biases.

Classes
-------
OddsConverter : ABC
    Abstract base class for odds conversion strategies.
BasicConverter : OddsConverter
    Basic method using multiplicative normalization.
OddsRatioConverter : OddsConverter
    Odds ratio method (Cheung, 2015).
ShinsConverter : OddsConverter
    Shin's method for insider trading (Shin, 1992).
PowerConverter : OddsConverter
    Power method using variable exponent.
HarvilleConverter : OddsConverter
    Harville method with dynamic programming for grid probabilities.
ConverterFactory
    Factory for obtaining odds converters.

Examples
--------
>>> # Convert odds using basic method
>>> odds = [1.5, 3.0, 6.0]
>>> converter = ConverterFactory.get('basic')
>>> probs = converter.convert(odds)
>>> probs
[0.6666666666666666, 0.3333333333333333, 0.16666666666666666]

>>> # Create position distribution from odds
>>> from gridrival_ai.probabilities.core import PositionDistribution
>>> dist = odds_to_position_distribution(odds, method='basic')
>>> dist[1]  # Probability of P1
0.5714285714285714
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Union

import numpy as np
from scipy.optimize import minimize

from gridrival_ai.probabilities.core import PositionDistribution


class OddsConverter(ABC):
    """
    Abstract base class for odds conversion strategies.

    This class defines the interface for converting betting odds to
    probabilities. Concrete implementations handle different methods
    for margin removal and bias adjustment.
    """

    @abstractmethod
    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.
            Use values > 1.0 for markets like "top N finish".

        Returns
        -------
        np.ndarray
            Array of probabilities summing to target_sum.
        """
        pass

    def convert_to_dict(
        self, odds: List[float], target_sum: float = 1.0
    ) -> Dict[int, float]:
        """
        Convert odds to probability dictionary.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping positions (1-based) to probabilities.
        """
        probs = self.convert(odds, target_sum)
        return {i + 1: float(p) for i, p in enumerate(probs)}

    def to_position_distribution(
        self, odds: List[float], target_sum: float = 1.0, validate: bool = True
    ) -> PositionDistribution:
        """
        Convert odds to position distribution.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.
        validate : bool, optional
            Whether to validate the distribution, by default True.

        Returns
        -------
        PositionDistribution
            Distribution over positions.

        Examples
        --------
        >>> converter = BasicConverter()
        >>> odds = [1.5, 3.0, 6.0]
        >>> dist = converter.to_position_distribution(odds)
        >>> dist[1]  # Probability of P1
        0.5714285714285714
        """
        probs_dict = self.convert_to_dict(odds, target_sum)
        return PositionDistribution(probs_dict, _validate=validate)


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
        """
        # Validate odds
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be greater than 1.0")

        raw_probs = np.array([1 / o for o in odds])
        return raw_probs * (target_sum / raw_probs.sum())


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


class ShinsConverter(OddsConverter):
    """
    Shin's method for converting odds to probabilities.

    Implements Shin's (1992) method which models the market as having a
    proportion z of insider traders:
    p[i] = r[i](1-z)/(1-z*r[i])
    where p is true probability, r is raw probability, z is insider proportion.

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
        Last optimized insider proportion value.

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

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities using Shin's method.

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

        def objective(z: float) -> float:
            probs = raw_probs * (1 - z) / (1 - z * raw_probs)
            return abs(target_sum - probs.sum())

        result = minimize(
            objective,
            x0=self.last_z_value,
            bounds=[(0, self.max_z)],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")

        self.last_z_value = float(result.x[0])

        # Calculate final probabilities using optimal z value
        probs = (
            raw_probs * (1 - self.last_z_value) / (1 - self.last_z_value * raw_probs)
        )

        # Normalize to ensure exact target probability
        return probs * (target_sum / probs.sum())


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


class HarvilleConverter(OddsConverter):
    """
    Harville method for converting odds to grid probabilities.

    The Harville method is a dynamic programming approach that computes
    a complete grid of finishing position probabilities. It uses "strengths"
    derived from win odds and ensures constraints across rows and columns.

    Parameters
    ----------
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-10.

    Notes
    -----
    This implementation assumes that the input odds represent win probabilities.
    For other markets (e.g., top 3, top 6), use target_sum parameter and the
    method will adjust accordingly.
    """

    def __init__(self, epsilon: float = 1e-10) -> None:
        """Initialize with epsilon parameter."""
        self.epsilon = epsilon

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert win odds to win probabilities (row 1 of grid).

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.

        Returns
        -------
        np.ndarray
            Array of win probabilities summing to target_sum.
            This is only the first row of the full grid.

        Notes
        -----
        For a complete grid of probabilities, use convert_to_grid().
        """
        # Validate odds
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be greater than 1.0")

        # Convert odds to strengths that sum to target_sum
        raw_probs = np.array([1 / o for o in odds])
        return raw_probs * (target_sum / raw_probs.sum())

    def convert_to_grid(
        self, odds: List[float], driver_ids: Optional[List[str]] = None
    ) -> Dict[Union[str, int], Dict[int, float]]:
        """
        Convert win odds to a complete grid of position probabilities.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds for win market. Must be > 1.0.
        driver_ids : List[str], optional
            List of driver identifiers. If None, positions are used.

        Returns
        -------
        Dict[Union[str, int], Dict[int, float]]
            Nested dictionary mapping drivers to their position probabilities.
            {driver_id: {position: probability, ...}, ...}

        Examples
        --------
        >>> converter = HarvilleConverter()
        >>> odds = [1.5, 3.0]
        >>> grid = converter.convert_to_grid(odds, ["VER", "HAM"])
        >>> grid["VER"][1]  # Probability VER finishes P1
        0.6666666666666667
        >>> grid["VER"][2]  # Probability VER finishes P2
        0.33333333333333337
        >>> grid["HAM"][1]  # Probability HAM finishes P1
        0.33333333333333337
        >>> grid["HAM"][2]  # Probability HAM finishes P2
        0.6666666666666667
        """
        n = len(odds)
        if driver_ids is None:
            driver_ids = [str(i) for i in range(1, n + 1)]

        if len(driver_ids) != n:
            raise ValueError(
                f"Length of driver_ids ({len(driver_ids)}) must match odds ({n})"
            )

        # Convert odds to "strengths" using basic method
        strengths = self.convert(odds, target_sum=1.0)

        # Initialize result dictionary with empty dicts for each driver
        result = {driver_id: {} for driver_id in driver_ids}

        # dp[mask] holds the probability of reaching that state
        dp = np.zeros(1 << n)
        full_mask = (1 << n) - 1
        dp[full_mask] = 1.0

        # Iterate over all masks from full_mask down to 0
        for mask in range(full_mask, -1, -1):
            if dp[mask] == 0:
                continue

            # Build list of available drivers
            available = [i for i in range(n) if mask & (1 << i)]
            if not available:
                continue

            # Sum of strengths for drivers available in this mask
            s = sum(strengths[i] for i in available)

            # The finishing position we are about to assign:
            pos = n - len(available) + 1  # positions are 1-indexed

            # For each available driver, assign the probability to finish in pos
            for i in available:
                p_i = strengths[i] / (s + self.epsilon)
                prob = dp[mask] * p_i

                # Initialize position probability if it doesn't exist
                if pos not in result[driver_ids[i]]:
                    result[driver_ids[i]][pos] = 0

                # Accumulate the probability instead of just assigning
                result[driver_ids[i]][pos] += prob

                new_mask = mask & ~(1 << i)
                dp[new_mask] += prob

        return result

    def grid_to_position_distributions(
        self, grid: Dict[str, Dict[int, float]]
    ) -> Dict[str, PositionDistribution]:
        """
        Convert grid of probabilities to position distributions.

        Parameters
        ----------
        grid : Dict[str, Dict[int, float]]
            Grid of probabilities from convert_to_grid().

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions.

        Examples
        --------
        >>> converter = HarvilleConverter()
        >>> odds = [1.5, 3.0]
        >>> grid = converter.convert_to_grid(odds, ["VER", "HAM"])
        >>> dists = converter.grid_to_position_distributions(grid)
        >>> dists["VER"][1]  # Probability VER finishes P1
        0.6666666666666667
        """
        return {
            driver_id: PositionDistribution(positions)
            for driver_id, positions in grid.items()
        }


class ConverterFactory:
    """
    Factory for obtaining odds converters.

    This class provides a central registry for odds conversion strategies
    and a factory method to get the appropriate converter.

    Class Attributes
    ---------------
    _converters : Dict[str, Type[OddsConverter]]
        Registry of available converters.

    Examples
    --------
    >>> converter = ConverterFactory.get('basic')
    >>> odds = [1.5, 3.0, 6.0]
    >>> probs = converter.convert(odds)
    >>> probs
    array([0.57142857, 0.28571429, 0.14285714])

    >>> # Register custom converter
    >>> class MyConverter(OddsConverter):
    ...     def convert(self, odds, target_sum=1.0):
    ...         return np.array([1.0] + [0.0] * (len(odds) - 1))
    >>> ConverterFactory.register('my_method', MyConverter)
    >>> converter = ConverterFactory.get('my_method')
    >>> probs = converter.convert([1.5, 3.0, 6.0])
    >>> probs
    array([1., 0., 0.])
    """

    _converters: Dict[str, Type[OddsConverter]] = {
        "basic": BasicConverter,
        "odds_ratio": OddsRatioConverter,
        "shin": ShinsConverter,
        "power": PowerConverter,
        "harville": HarvilleConverter,
    }

    @classmethod
    def get(cls, method: str = "basic", **kwargs) -> OddsConverter:
        """
        Get a converter instance for the specified method.

        Parameters
        ----------
        method : str, optional
            Conversion method name, by default "basic".
            Options: "basic", "odds_ratio", "shin", "power", "harville".
        **kwargs
            Additional parameters for the converter constructor.

        Returns
        -------
        OddsConverter
            Instance of the appropriate converter.

        Raises
        ------
        ValueError
            If method is not recognized.
        """
        if method not in cls._converters:
            valid_methods = ", ".join(cls._converters.keys())
            raise ValueError(
                f"Unknown conversion method: {method}. "
                f"Valid methods are: {valid_methods}"
            )

        converter_class = cls._converters[method]
        return converter_class(**kwargs)

    @classmethod
    def register(cls, name: str, converter_class: Type[OddsConverter]) -> None:
        """
        Register a new converter class.

        Parameters
        ----------
        name : str
            Name for the conversion method.
        converter_class : Type[OddsConverter]
            Converter class to register.

        Raises
        ------
        TypeError
            If converter_class is not a subclass of OddsConverter.
        """
        if not issubclass(converter_class, OddsConverter):
            raise TypeError(
                f"Converter class must be a subclass of OddsConverter, "
                f"got {converter_class.__name__}"
            )

        cls._converters[name] = converter_class


def odds_to_position_distribution(
    odds: List[float], method: str = "basic", target_sum: float = 1.0, **kwargs
) -> PositionDistribution:
    """
    Convert odds to position distribution.

    Utility function for creating a position distribution directly from odds.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds. Must be > 1.0.
    method : str, optional
        Conversion method name, by default "basic".
    target_sum : float, optional
        Target sum for probabilities, by default 1.0.
    **kwargs
        Additional parameters for the converter.

    Returns
    -------
    PositionDistribution
        Position distribution.

    Examples
    --------
    >>> odds = [1.5, 3.0, 6.0]
    >>> dist = odds_to_position_distribution(odds)
    >>> dist[1]  # Probability of P1
    0.5714285714285714

    >>> # Use a different method
    >>> dist = odds_to_position_distribution(odds, method='shin')
    >>> dist[1] > 0.5
    True
    """
    converter = ConverterFactory.get(method, **kwargs)
    return converter.to_position_distribution(odds, target_sum)


def odds_to_grid(
    odds: List[float], driver_ids: Optional[List[str]] = None, **kwargs
) -> Dict[Union[str, int], Dict[int, float]]:
    """
    Convert win odds to a complete grid of position probabilities.

    Utility function for creating a grid of position probabilities
    directly from win odds.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for win market. Must be > 1.0.
    driver_ids : List[str], optional
        List of driver identifiers. If None, positions are used.
    **kwargs
        Additional parameters for the Harville converter.

    Returns
    -------
    Dict[Union[str, int], Dict[int, float]]
        Nested dictionary mapping drivers to their position probabilities.
        {driver_id: {position: probability, ...}, ...}

    Examples
    --------
    >>> odds = [1.5, 3.0]
    >>> grid = odds_to_grid(odds, ["VER", "HAM"])
    >>> grid["VER"][1]  # Probability VER finishes P1
    0.6666666666666667
    """
    converter = HarvilleConverter(**kwargs)
    return converter.convert_to_grid(odds, driver_ids)


def odds_to_distributions(
    odds: List[float], driver_ids: Optional[List[str]] = None, **kwargs
) -> Dict[str, PositionDistribution]:
    """
    Convert win odds to position distributions for each driver.

    Utility function for creating a set of position distributions
    directly from win odds.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for win market. Must be > 1.0.
    driver_ids : List[str], optional
        List of driver identifiers. If None, positions are used.
    **kwargs
        Additional parameters for the Harville converter.

    Returns
    -------
    Dict[str, PositionDistribution]
        Dictionary mapping driver IDs to position distributions.

    Examples
    --------
    >>> odds = [1.5, 3.0]
    >>> dists = odds_to_distributions(odds, ["VER", "HAM"])
    >>> dists["VER"][1]  # Probability VER finishes P1
    0.6666666666666667
    """
    converter = HarvilleConverter(**kwargs)
    grid = converter.convert_to_grid(odds, driver_ids)
    return converter.grid_to_position_distributions(grid)
