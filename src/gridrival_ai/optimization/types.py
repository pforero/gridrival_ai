"""
Data structures for F1 fantasy team optimization.

This module contains the data structures used for optimizing F1 fantasy team
composition under budget and roster constraints. These types are used throughout
the optimization process to track driver/constructor scoring information, team
solutions, and optimization results.
"""

from dataclasses import dataclass, field
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

    This class contains all valid team solutions within budget and provides methods
    to filter and analyze them based on different constraints.

    Parameters
    ----------
    all_solutions : list[TeamSolution]
        All valid team solutions within budget
    locked_in : set[str]
        Elements that must be included in filtered solutions
    locked_out : set[str]
        Elements that must be excluded from filtered solutions
    budget : float
        Total available budget

    Notes
    -----
    The class maintains both the complete set of solutions within budget
    and a filtered subset based on current constraints.

    Examples
    --------
    >>> # Create result with all solutions and initial filters
    >>> result = OptimizationResult(
    ...     all_solutions=all_valid_solutions,
    ...     locked_in={"VER"},
    ...     locked_out={"HAM"}
    ... )
    >>>
    >>> # Get best solution with current filters
    >>> best = result.best_solution
    >>> print(f"Best team has {best.expected_points} points")
    >>>
    >>> # Get top 5 solutions with current filters
    >>> top_5 = result.top_n(5)
    >>>
    >>> # Create new result with different filters
    >>> ferrari_teams = result.with_elements({"FER"})
    >>> no_restrictions = result.without_restrictions()
    """

    all_solutions: list[TeamSolution]
    locked_in: set[str] = field(default_factory=set)
    locked_out: set[str] = field(default_factory=set)
    budget: float = 100.0

    def __post_init__(self):
        """Initialize by sorting solutions by expected points."""
        # Sort all solutions by expected points (highest first)
        self.all_solutions.sort(key=lambda x: x.expected_points, reverse=True)

    @property
    def filtered_solutions(self) -> list[TeamSolution]:
        """Return solutions matching current constraints.

        Returns
        -------
        list[TeamSolution]
            Solutions that satisfy both locked_in and locked_out constraints
        """
        solutions = self.all_solutions

        if self.locked_in:
            solutions = [
                sol
                for sol in solutions
                if all(
                    elem in sol.drivers or elem == sol.constructor
                    for elem in self.locked_in
                )
            ]

        if self.locked_out:
            solutions = [
                sol
                for sol in solutions
                if all(
                    elem not in sol.drivers and elem != sol.constructor
                    for elem in self.locked_out
                )
            ]

        return solutions

    @property
    def best_solution(self) -> TeamSolution | None:
        """Return the best solution in the filtered set.

        Returns
        -------
        TeamSolution or None
            Best solution by expected points, None if no valid solutions
        """
        solutions = self.filtered_solutions
        return solutions[0] if solutions else None

    @property
    def remaining_budget(self) -> float:
        """Return remaining budget with best solution.

        Returns
        -------
        float
            Unspent budget with the best solution in filtered set
        """
        if self.best_solution:
            return self.budget - self.best_solution.total_cost
        return self.budget

    def top_n(self, n: int) -> list[TeamSolution]:
        """Return top N solutions in filtered set.

        Parameters
        ----------
        n : int
            Number of solutions to return

        Returns
        -------
        list[TeamSolution]
            Top N solutions sorted by expected points
        """
        return self.filtered_solutions[:n]

    def without_restrictions(self) -> "OptimizationResult":
        """Return a new result with no restrictions.

        Returns
        -------
        OptimizationResult
            New result object with no locked_in or locked_out elements
        """
        return OptimizationResult(
            all_solutions=self.all_solutions,
            locked_in=set(),
            locked_out=set(),
            budget=self.budget,
        )

    def with_elements(self, elements: set[str]) -> "OptimizationResult":
        """Return a new result requiring specific elements.

        Parameters
        ----------
        elements : set[str]
            Elements that must be included

        Returns
        -------
        OptimizationResult
            New result object with updated locked_in elements
        """
        new_locked_in = self.locked_in | elements
        return OptimizationResult(
            all_solutions=self.all_solutions,
            locked_in=new_locked_in,
            locked_out=self.locked_out,
            budget=self.budget,
        )

    def without_elements(self, elements: set[str]) -> "OptimizationResult":
        """Return a new result excluding specific elements.

        Parameters
        ----------
        elements : set[str]
            Elements that must be excluded

        Returns
        -------
        OptimizationResult
            New result object with updated locked_out elements
        """
        new_locked_out = self.locked_out | elements
        return OptimizationResult(
            all_solutions=self.all_solutions,
            locked_in=self.locked_in,
            locked_out=new_locked_out,
            budget=self.budget,
        )
