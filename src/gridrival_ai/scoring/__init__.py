"""
GridRival AI scoring package.

This package provides functionality for calculating F1 fantasy points
based on race results and scoring rules.
"""

from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.config import ScoringConfig
from gridrival_ai.scoring.engine import ScoringEngine
from gridrival_ai.scoring.types import (
    ConstructorPositions,
    ConstructorWeekendData,
    DriverPointsBreakdown,
    DriverPositions,
    DriverWeekendData,
    RaceFormat,
)

__all__ = [
    # Main classes
    "ScoringCalculator",
    "ScoringConfig",
    "ScoringEngine",
    # Data types
    "DriverPointsBreakdown",
    "DriverPositions",
    "DriverWeekendData",
    "ConstructorPositions",
    "ConstructorWeekendData",
    "RaceFormat",
]
