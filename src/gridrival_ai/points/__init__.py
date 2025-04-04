"""
GridRival AI points calculation package.

This package provides components for calculating expected fantasy points
based on probability distributions. It is designed to work with the
probability API and handles all aspects of the GridRival scoring system.

Main Components
--------------
- PointsCalculator: Main interface for points calculations
- DriverPointsCalculator: Calculator for driver points
- ConstructorPointsCalculator: Calculator for constructor points
- Component calculators: Specialized calculators for each scoring element

Usage
-----
>>> from gridrival_ai.points import PointsCalculator
>>> from gridrival_ai.probabilities.distributions import RaceDistribution
>>> from gridrival_ai.scoring.calculator import Scorer
>>>
>>> # Create calculator with RaceDistribution, scorer, and driver stats
>>> race_dist = RaceDistribution.from_structured_odds(odds_data)
>>> scorer = Scorer(config)
>>> driver_stats = {"VER": 1.5, "PER": 3.2}
>>>
>>> calculator = PointsCalculator(scorer, race_dist, driver_stats)
>>>
>>> # Calculate expected points
>>> ver_points = calculator.calculate_driver_points("VER")
>>> rbr_points = calculator.calculate_constructor_points("RBR")
"""

from gridrival_ai.points.calculator import PointsCalculator
from gridrival_ai.points.components import (
    CompletionPointsCalculator,
    ImprovementPointsCalculator,
    OvertakePointsCalculator,
    PositionPointsCalculator,
    TeammatePointsCalculator,
)
from gridrival_ai.points.constructor import ConstructorPointsCalculator
from gridrival_ai.points.driver import DriverPointsCalculator

__all__ = [
    # Main calculator
    "PointsCalculator",
    # Specialized calculators
    "DriverPointsCalculator",
    "ConstructorPointsCalculator",
    # Component calculators
    "PositionPointsCalculator",
    "OvertakePointsCalculator",
    "TeammatePointsCalculator",
    "CompletionPointsCalculator",
    "ImprovementPointsCalculator",
]
