"""Data structures for F1 fantasy team optimization.

This module contains the data structures used for optimizing F1 fantasy team
composition under budget and roster constraints.
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple


@dataclass(frozen=True)
class DriverScoring:
    """Driver scoring information for optimization.

    Parameters
    ----------
    regular_points : float
        Expected points without talent driver bonus
    points_dict : Dict[str, float]
        Breakdown of points by component
    salary : float
        Driver salary in millions
    can_be_talent : bool
        Whether driver is eligible for talent driver selection

    Notes
    -----
    Talent driver points are calculated by doubling regular points.
    Salary threshold for talent drivers is 18M.
    """

    regular_points: float
    points_dict: Dict[str, float]
    salary: float
    can_be_talent: bool


class TeamSolution(NamedTuple):
    """Represents a valid team solution.

    Parameters
    ----------
    drivers : frozenset[str]
        Set of 5 driver IDs
    constructor : str
        Constructor ID
    talent_driver : str
        ID of designated talent driver
    expected_points : float
        Total expected points
    total_cost : float
        Total team cost including any penalties
    points_breakdown : dict[str, float]
        Expected points contribution per element

    Notes
    -----
    Points breakdown includes both drivers and constructor contributions.
    Total cost includes any penalties for unused locked-in drivers.
    """

    drivers: frozenset[str]
    constructor: str
    talent_driver: str
    expected_points: float
    total_cost: float
    points_breakdown: dict[str, float]


@dataclass
class OptimizationResult:
    """Result of team optimization.

    Parameters
    ----------
    best_solution : TeamSolution | None
        Best team found, None if no valid solution exists
    alternative_solutions : list[TeamSolution]
        Other solutions with same expected points (within tolerance)
    remaining_budget : float
        Unspent budget with best solution
    error_message : str | None, optional
        Description of why no solution found, by default None

    Notes
    -----
    Solutions are considered equal if their expected points differ by less
    than a small tolerance (typically 1e-6).
    """

    best_solution: TeamSolution | None
    alternative_solutions: list[TeamSolution]
    remaining_budget: float
    error_message: str | None = None
