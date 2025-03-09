"""
GridRival AI points calculation package.

This package provides components for calculating expected fantasy points
based on probability distributions. It is designed to work with the
new probability API and handles all aspects of the GridRival scoring system.

Main Components
--------------
- PointsCalculator: Main interface for points calculations
- DriverPointsCalculator: Calculator for driver points
- ConstructorPointsCalculator: Calculator for constructor points
- Component calculators: Specialized calculators for each scoring element
- DistributionAdapter: Bridge to the probability distribution API

Usage
-----
>>> from gridrival_ai.points import PointsCalculator
>>> from gridrival_ai.probabilities.registry import DistributionRegistry
>>> from gridrival_ai.scoring.calculator import Scorer
>>>
>>> # Create calculator with registry, scorer, and driver stats
>>> registry = DistributionRegistry()
>>> scorer = Scorer(config)
>>> driver_stats = {"VER": 1.5, "PER": 3.2}
>>>
>>> calculator = PointsCalculator(scorer, registry, driver_stats)
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
from gridrival_ai.points.distributions import DistributionAdapter
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
    # Adapter
    "DistributionAdapter",
]
