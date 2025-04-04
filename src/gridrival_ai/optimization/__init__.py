"""
GridRival AI optimization package.

This package provides functionality for optimizing F1 fantasy team compositions
based on expected points and budget constraints.
"""

from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.optimization.types import (
    ConstructorScoring,
    DriverScoring,
    OptimizationResult,
    TeamSolution,
)

__all__ = [
    "TeamOptimizer",
    "ConstructorScoring",
    "DriverScoring",
    "OptimizationResult",
    "TeamSolution",
]
