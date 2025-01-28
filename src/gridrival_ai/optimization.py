"""
Optimization module for GridRival AI.

This module provides the Optimizer class for finding optimal team compositions.
"""

from typing import Optional

from gridrival_ai.team import Team


class Optimizer:
    """
    A class to optimize F1 fantasy team composition.

    Parameters
    ----------
    budget : float
        Total budget available for team creation.
    """

    def __init__(self, budget: float) -> None:
        """Initialize the optimizer with a budget."""
        self.budget = budget
