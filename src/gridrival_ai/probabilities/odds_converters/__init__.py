"""
Odds conversion utilities for GridRival AI.

This package provides tools for converting betting odds to probability
distributions with various mathematical models.
"""

from gridrival_ai.probabilities.odds_converters.base import OddsConverter
from gridrival_ai.probabilities.odds_converters.basic import BasicConverter
from gridrival_ai.probabilities.odds_converters.factory import get_odds_converter
from gridrival_ai.probabilities.odds_converters.power import PowerConverter
from gridrival_ai.probabilities.odds_converters.ratio import OddsRatioConverter
from gridrival_ai.probabilities.odds_converters.shins import ShinsConverter

__all__ = [
    # Base class
    "OddsConverter",
    # Implementations
    "BasicConverter",
    "PowerConverter",
    "OddsRatioConverter",
    "ShinsConverter",
    # Factory
    "get_odds_converter",
]
