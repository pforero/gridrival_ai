"""
Data structures for F1 fantasy team optimization.

This module contains the data structures used for optimizing F1 fantasy team
composition under budget and roster constraints. These types are used throughout
the optimization process to track driver/constructor scoring information, team
solutions, and optimization results.
"""

from dataclasses import dataclass
from typing import FrozenSet, NamedTuple


@dataclass(frozen=True)
class DriverScoring:
    """Driver scoring information for optimization.

    This class stores pre-calculated scoring information for drivers to be used
    in the optimization process.

    Parameters
    ----------
    regular_points : float
        Expected points without talent driver bonus
    points_dict : Dict[str, float]
        Breakdown of points by component (qualifying, race, sprint, etc.)
    salary : float
        Driver salary in millions
    can_be_talent : bool
        Whether driver is eligible for talent driver selection

    Notes
    -----
    Talent driver points are calculated by doubling regular points.
    Typical salary threshold for talent drivers is 18M.

    Examples
    --------
    >>> points_breakdown = {"qualifying": 40.0, "race": 60.0}
    >>> driver_scoring = DriverScoring(
    ...     regular_points=100.0,
    ...     points_dict=points_breakdown,
    ...     salary=15.0,
    ...     can_be_talent=True
    ... )
    """

    regular_points: float
    points_dict: dict[str, float]
    salary: float
    can_be_talent: bool


@dataclass(frozen=True)
class ConstructorScoring:
    """Constructor scoring information for optimization.

    This class stores pre-calculated scoring information for constructors to be used
    in the optimization process.

    Parameters
    ----------
    points : float
        Total expected points for the constructor
    points_dict : Dict[str, float]
        Breakdown of points by component (qualifying, race)
    salary : float
        Constructor salary in millions

    Examples
    --------
    >>> points_breakdown = {"qualifying": 50.0, "race": 100.0}
    >>> constructor_scoring = ConstructorScoring(
    ...     points=150.0,
    ...     points_dict=points_breakdown,
    ...     salary=20.0
    ... )
    """

    points: float
    points_dict: dict[str, float]
    salary: float


class TeamSolution(NamedTuple):
    """Represents a valid team solution.

    This class represents a complete team composition that satisfies all constraints
    and has been evaluated for expected points.

    Parameters
    ----------
    drivers : FrozenSet[str]
        Set of 5 driver IDs
    constructor : str
        Constructor ID
    talent_driver : str
        ID of designated talent driver (empty string if none)
    expected_points : float
        Total expected points
    total_cost : float
        Total team cost including any penalties
    points_breakdown : Dict[str, Union[Dict[str, float], float]]
        Expected points contribution per element

    Notes
    -----
    Points breakdown includes both drivers and constructor contributions.
    Total cost includes any penalties for unused locked-in drivers.

    Examples
    --------
    >>> drivers = frozenset(["VER", "HAM", "NOR", "ALO", "PIA"])
    >>> solution = TeamSolution(
    ...     drivers=drivers,
    ...     constructor="RBR",
    ...     talent_driver="ALO",
    ...     expected_points=450.0,
    ...     total_cost=98.5,
    ...     points_breakdown={"VER": 120.0, "HAM": 90.0, "RBR": 150.0, ...}
    ... )
    """

    drivers: FrozenSet[str]
    constructor: str
    talent_driver: str
    expected_points: float
    total_cost: float
    points_breakdown: dict[str, dict[str, float] | float]


@dataclass
class OptimizationResult:
    """Result of team optimization.

    This class contains the results of the team optimization process, including
    the best team composition found and any alternative solutions.

    Parameters
    ----------
    best_solution : Optional[TeamSolution]
        Best team found, None if no valid solution exists
    alternative_solutions : List[TeamSolution]
        Other solutions with same expected points (within tolerance)
    remaining_budget : float
        Unspent budget with best solution
    error_message : Optional[str]
        Description of why no solution found, by default None

    Notes
    -----
    Solutions are considered equal if their expected points differ by less
    than a small tolerance (typically 1e-6).

    Examples
    --------
    >>> result = OptimizationResult(
    ...     best_solution=team_solution,
    ...     alternative_solutions=[other_solution1, other_solution2],
    ...     remaining_budget=1.5,
    ...     error_message=None
    ... )
    >>> if result.best_solution:
    ...     print(f"Best team found with {result.best_solution.expected_points} points")
    ... else:
    ...     print(f"Error: {result.error_message}")
    """

    best_solution: TeamSolution | None
    alternative_solutions: list[TeamSolution]
    remaining_budget: float
    error_message: str | None = None
