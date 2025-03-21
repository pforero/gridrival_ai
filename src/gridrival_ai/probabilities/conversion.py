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

from typing import Dict, List, Optional, Type, Union

from gridrival_ai.probabilities.converters import (
    BasicConverter,
    CumulativeMarketConverter,
    HarvilleConverter,
    OddsRatioConverter,
    PowerConverter,
    ShinsConverter,
)
from gridrival_ai.probabilities.converters.odds_converter import OddsConverter
from gridrival_ai.probabilities.core import PositionDistribution


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
        "cumulative": CumulativeMarketConverter,
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
