"""
Factory for obtaining grid creators.

This module provides a factory function for creating grid creator instances
with a simple interface.
"""

from typing import Dict, Type

from gridrival_ai.probabilities.grid_creators.base import GridCreator
from gridrival_ai.probabilities.grid_creators.cumulative import CumulativeGridCreator
from gridrival_ai.probabilities.grid_creators.harville import HarvilleGridCreator

# Registry of available creators
_CREATORS: Dict[str, Type[GridCreator]] = {
    "cumulative": CumulativeGridCreator,
    "harville": HarvilleGridCreator,
}


def get_grid_creator(method: str = "harville", **kwargs) -> GridCreator:
    """
    Get a grid creator instance for the specified method.

    Parameters
    ----------
    method : str, optional
        Grid creator method name, by default "harville".
        Options: "harville", "cumulative"
    **kwargs
        Additional parameters for the grid creator constructor.

    Returns
    -------
    GridCreator
        Instance of the appropriate grid creator.

    Raises
    ------
    ValueError
        If method is not recognized.

    Examples
    --------
    >>> creator = get_grid_creator("harville")
    >>> odds = {"race": {1: {"VER": 2.0, "HAM": 4.0}}}
    >>> session_dist = creator.create_session_distribution(odds)
    >>> session_dist.get_driver_distribution("VER").get(1)
    0.6666666666666666

    >>> # With custom parameters
    >>> creator = get_grid_creator("cumulative", max_position=10)
    >>> race_dist = creator.create_race_distribution(odds)
    """
    if method not in _CREATORS:
        valid_methods = ", ".join(_CREATORS.keys())
        raise ValueError(
            f"Unknown grid creator method: {method}. "
            f"Valid methods are: {valid_methods}"
        )

    creator_class = _CREATORS[method]
    return creator_class(**kwargs)
