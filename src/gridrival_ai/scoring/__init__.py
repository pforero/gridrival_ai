"""
GridRival AI scoring package.

This package provides functionality for calculating F1 fantasy points
based on race results and scoring rules.
"""

from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import RaceFormat

__all__ = [
    "ScoringConfig",
    "Scorer",
    "RaceFormat",
    "create_points_table",
]
