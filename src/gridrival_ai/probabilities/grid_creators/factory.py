"""
Factory for obtaining odds converters.

This module provides a factory function for creating odds converter instances
with a simple interface.
"""

from typing import Dict, Type

from gridrival_ai.probabilities.grid_creators.base import GridCreator
from gridrival_ai.probabilities.grid_creators.cumulative import (
    CumulativeMarketConverter,
)
from gridrival_ai.probabilities.grid_creators.harville import HarvilleGridCreator

# Registry of available creators
_CREATORS: Dict[str, Type[GridCreator]] = {
    "cumulative": CumulativeMarketConverter,
    "harville": HarvilleGridCreator,
}


def get_grid_creator(method: str = "basic", **kwargs) -> GridCreator:
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
    GridCreator
        Instance of the appropriate converter.

    Raises
    ------
    ValueError
        If method is not recognized.

    Examples
    --------
    >>> converter = get_grid_creator("basic")
    >>> probs = converter.convert([2.0, 4.0, 6.0])
    >>> probs.sum()
    1.0

    >>> # With custom parameters
    >>> converter = get_grid_creator("power", max_iter=2000)
    >>> probs = converter.convert([2.0, 4.0, 6.0])
    """
    if method not in _CREATORS:
        valid_methods = ", ".join(_CREATORS.keys())
        raise ValueError(
            f"Unknown conversion method: {method}. "
            f"Valid methods are: {valid_methods}"
        )

    converter_class = _CREATORS[method]
    return converter_class(**kwargs)
