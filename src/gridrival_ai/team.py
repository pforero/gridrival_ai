"""
Team management module for GridRival AI.

This module provides the Team class for managing F1 fantasy teams.
"""

from typing import List, Optional


class Team:
    """
    A class to represent and manage an F1 fantasy team.

    Parameters
    ----------
    budget : float
        Total budget available for the team.
    """

    def __init__(self, budget: float) -> None:
        """Initialize a new team instance."""
        self.budget = budget
