"""
Factory for obtaining odds converters.

This module provides a factory function for creating odds converter instances
with a simple interface.
"""

from typing import Dict, Type

from gridrival_ai.probabilities.odds_converters.base import OddsConverter
from gridrival_ai.probabilities.odds_converters.basic import BasicConverter
from gridrival_ai.probabilities.odds_converters.power import PowerConverter
from gridrival_ai.probabilities.odds_converters.ratio import OddsRatioConverter
from gridrival_ai.probabilities.odds_converters.shins import ShinsConverter

# Registry of available converters
_CONVERTERS: Dict[str, Type[OddsConverter]] = {
    "basic": BasicConverter,
    "power": PowerConverter,
    "odds_ratio": OddsRatioConverter,
    "shin": ShinsConverter,
}


def get_odds_converter(method: str = "basic", **kwargs) -> OddsConverter:
    """
    Get an odds converter instance for the specified method.

    Parameters
    ----------
    method : str, optional
        Conversion method name, by default "basic".
        Options: "basic", "power", "odds_ratio", "shin"
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

    Examples
    --------
    >>> converter = get_odds_converter("basic")
    >>> probs = converter.convert([2.0, 4.0, 6.0])
    >>> probs.sum()
    1.0

    >>> # With custom parameters
    >>> converter = get_odds_converter("power", max_iter=2000)
    >>> probs = converter.convert([2.0, 4.0, 6.0])
    """
    if method not in _CONVERTERS:
        valid_methods = ", ".join(_CONVERTERS.keys())
        raise ValueError(
            f"Unknown conversion method: {method}. Valid methods are: {valid_methods}"
        )

    converter_class = _CONVERTERS[method]
    return converter_class(**kwargs)
