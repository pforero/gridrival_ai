"""Data structures for F1 fantasy team optimization.

This module contains the data structures used for optimizing F1 fantasy team
composition under budget and roster constraints.
"""

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class DriverScoring:
    """Pre-calculated scoring data for a driver.

    Parameters
    ----------
    regular_points : float
        Expected points for the race weekend
    salary : float
        Current driver salary
    can_be_talent : bool
        Whether eligible to be talent driver (salary <= 18M)

    Notes
    -----
    Talent driver points are calculated by doubling regular points.
    Salary threshold for talent drivers is 18M.
    """

    regular_points: float
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
