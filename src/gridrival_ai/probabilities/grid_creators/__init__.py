"""
Odds conversion utilities for GridRival AI.

This package provides tools for converting betting odds to probability
distributions with various mathematical models.
"""

from gridrival_ai.probabilities.grid_creators.base import GridCreator
from gridrival_ai.probabilities.grid_creators.cumulative import (
    CumulativeMarketConverter,
)
from gridrival_ai.probabilities.grid_creators.factory import get_grid_creator
from gridrival_ai.probabilities.grid_creators.harville import HarvilleGridCreator

__all__ = [
    # Base class
    "GridCreator",
    # Implementations
    "CumulativeMarketConverter",
    "HarvilleGridCreator",
    # Factory
    "get_grid_creator",
]
