"""
GridRival AI optimization package.

This package provides functionality for optimizing F1 fantasy team compositions
based on expected points and budget constraints.
"""

from gridrival_ai.optimization.team_optimizer import TeamOptimizer
from gridrival_ai.optimization.types import ElementData, TeamComposition

__all__ = ["TeamOptimizer", "ElementData", "TeamComposition"]
