"""
Team optimization for F1 fantasy leagues with the GridRival format.

This module provides functionality for optimizing F1 fantasy team compositions
based on expected points and budget constraints using a brute force approach
with progressive pruning for efficiency.
"""

from itertools import combinations
from typing import Dict, Iterator, Optional, Set

import numpy as np

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.types import (
    ConstructorScoring,
    DriverScoring,
    OptimizationResult,
    TeamSolution,
)
from gridrival_ai.points.calculator import PointsCalculator
from gridrival_ai.probabilities.registry import DistributionRegistry
from gridrival_ai.scoring.types import RaceFormat

# Constants
TALENT_SALARY_THRESHOLD = 18.0  # Maximum salary for talent drivers
LOCKED_IN_PENALTY = 0.03  # 3% penalty for not using locked-in drivers


class TeamOptimizer:
    """
    Optimizer for F1 fantasy team composition.

    This class finds the optimal team composition for a GridRival fantasy league
    by evaluating combinations of drivers and constructors to maximize expected points
    while respecting budget and roster constraints.

    Parameters
    ----------
    league_data : FantasyLeagueData
        Current league state including salaries and constraints
    points_calculator : PointsCalculator
        Calculator for expected points using the new points API
    probability_registry : DistributionRegistry
        Registry containing probability distributions
    driver_stats : Dict[str, float]
        Dictionary mapping driver IDs to rolling averages
    budget : float, optional
        Total budget available, by default 100.0

    Notes
    -----
    Team composition rules:
    - 5 drivers and 1 constructor
    - Total cost must be within budget
    - One driver must be designated as talent driver (salary <= 18M)
    - Locked-in drivers must be included or incur 3% penalty
    - Locked-out drivers cannot be selected

    Examples
    --------
    >>> from gridrival_ai.data.fantasy import FantasyLeagueData
    >>> from gridrival_ai.points.calculator import PointsCalculator
    >>> from gridrival_ai.probabilities.registry import DistributionRegistry
    >>>
    >>> # Create prerequisites
    >>> league_data = FantasyLeagueData.from_dicts(
    ...     driver_salaries={"VER": 33.0, "HAM": 26.2},
    ...     constructor_salaries={"RBR": 30.0, "MER": 27.2},
    ...     rolling_averages={"VER": 1.5, "HAM": 3.0}
    ... )
    >>>
    >>> # Create registry and points calculator
    >>> registry = DistributionRegistry()
    >>> # Populate registry with distributions...
    >>>
    >>> points_calculator = PointsCalculator(scorer, registry, driver_stats)
    >>>
    >>> # Create optimizer
    >>> optimizer = TeamOptimizer(
    ...     league_data=league_data,
    ...     points_calculator=points_calculator,
    ...     probability_registry=registry,
    ...     driver_stats=driver_stats
    ... )
    >>>
    >>> # Optimize team
    >>> result = optimizer.optimize(race_format=RaceFormat.STANDARD)
    >>>
    >>> # Access best team
    >>> if result.best_solution:
    ...     print(f"Best team: {result.best_solution.drivers}")
    ...     print(f"Expected points: {result.best_solution.expected_points}")
    """

    def __init__(
        self,
        league_data: FantasyLeagueData,
        points_calculator: PointsCalculator,
        probability_registry: DistributionRegistry,
        driver_stats: Dict[str, float],
        budget: float = 100.0,
    ) -> None:
        """Initialize optimizer with league state."""
        self.league_data = league_data
        self.points_calculator = points_calculator
        self.probability_registry = probability_registry
        self.driver_stats = driver_stats
        self.budget = budget

    def optimize(
        self,
        race_format: RaceFormat = RaceFormat.STANDARD,
        locked_in: Optional[Set[str]] = None,
        locked_out: Optional[Set[str]] = None,
    ) -> OptimizationResult:
        """
        Find optimal team composition.

        This method evaluates combinations of drivers and constructors to find
        the team with the highest expected points that satisfies all constraints.

        Parameters
        ----------
        race_format : RaceFormat, optional
            Format of the race weekend, by default RaceFormat.STANDARD
        locked_in : Optional[Set[str]], optional
            Set of element IDs that must be included, by default None
        locked_out : Optional[Set[str]], optional
            Set of element IDs that cannot be included, by default None

        Returns
        -------
        OptimizationResult
            Results of the optimization, including best and alternative solutions

        Notes
        -----
        The optimization uses a brute force approach with pruning to eliminate
        invalid combinations early. Solutions are considered equal if their expected
        points differ by less than 1e-6.
        """
        # Initialize constraints
        locked_in = locked_in or set()
        locked_out = locked_out or set()

        # Pre-calculate driver scores
        driver_scores = self._calculate_driver_scores(race_format, locked_out)

        # Pre-calculate constructor scores
        constructor_scores = self._calculate_constructor_scores(race_format, locked_out)

        # Generate and evaluate team combinations
        best_solution = None
        alternative_solutions = []
        max_points = 0.0

        try:
            for solution in self._generate_valid_combinations(
                driver_scores, constructor_scores, locked_in, locked_out
            ):
                if best_solution is None:
                    # First valid solution
                    best_solution = solution
                    max_points = solution.expected_points
                    continue

                if np.isclose(
                    solution.expected_points,
                    max_points,
                    rtol=1e-6,
                ):
                    # Equal to current best
                    alternative_solutions.append(solution)
                elif solution.expected_points > max_points:
                    # New best solution
                    best_solution = solution
                    max_points = solution.expected_points
                    alternative_solutions = []

            if best_solution is None:
                return OptimizationResult(
                    best_solution=None,
                    alternative_solutions=[],
                    remaining_budget=self.budget,
                    error_message="Optimization failed: No valid team composition found"
                )

            remaining = self.budget - best_solution.total_cost

            return OptimizationResult(
                best_solution=best_solution,
                alternative_solutions=alternative_solutions,
                remaining_budget=remaining,
                error_message=None,
            )

        except Exception as e:
            return OptimizationResult(
                best_solution=None,
                alternative_solutions=[],
                remaining_budget=self.budget,
                error_message=f"Optimization failed: {str(e)}",
            )

    def _calculate_driver_scores(
        self, race_format: RaceFormat, locked_out: Set[str]
    ) -> Dict[str, DriverScoring]:
        """
        Calculate driver scoring information.

        Parameters
        ----------
        race_format : RaceFormat
            Format of the race weekend
        locked_out : Set[str]
            Set of element IDs that cannot be included

        Returns
        -------
        Dict[str, DriverScoring]
            Mapping of driver IDs to scoring information
        """
        driver_scores = {}

        # Get available drivers (not locked out)
        available_drivers = self.league_data.get_available_drivers() - locked_out

        # Calculate scores for each available driver
        for driver_id in available_drivers:
            # Check if driver has distributions in the registry
            if not self.probability_registry.has(driver_id, "race"):
                continue

            # Get driver salary and check talent eligibility
            salary = self.league_data.salaries.drivers[driver_id]
            can_be_talent = salary <= TALENT_SALARY_THRESHOLD

            # Calculate expected points using the points calculator
            points_dict = self.points_calculator.calculate_driver_points(
                driver_id=driver_id, race_format=race_format
            )

            # Total points across all components
            regular_points = sum(points_dict.values())

            driver_scores[driver_id] = DriverScoring(
                regular_points=regular_points,
                points_dict=points_dict,
                salary=salary,
                can_be_talent=can_be_talent,
            )

        return driver_scores

    def _calculate_constructor_scores(
        self, race_format: RaceFormat, locked_out: Set[str]
    ) -> Dict[str, ConstructorScoring]:
        """
        Calculate constructor scoring information.

        Parameters
        ----------
        race_format : RaceFormat
            Format of the race weekend
        locked_out : Set[str]
            Set of element IDs that cannot be included

        Returns
        -------
        Dict[str, ConstructorScoring]
            Mapping of constructor IDs to scoring information
        """
        constructor_scores = {}

        # Get available constructors (not locked out)
        available_constructors = (
            self.league_data.get_available_constructors() - locked_out
        )

        # Calculate scores for each available constructor
        for constructor_id in available_constructors:
            # Get constructor salary
            salary = self.league_data.salaries.constructors[constructor_id]

            # Calculate expected points using the points calculator
            try:
                points_dict = self.points_calculator.calculate_constructor_points(
                    constructor_id=constructor_id, race_format=race_format
                )

                # Total points across all components
                total_points = sum(points_dict.values())

                constructor_scores[constructor_id] = ConstructorScoring(
                    points=total_points, points_dict=points_dict, salary=salary
                )
            except Exception:
                # Skip constructors that can't be scored
                continue

        return constructor_scores

    def _generate_valid_combinations(
        self,
        driver_scores: Dict[str, DriverScoring],
        constructor_scores: Dict[str, ConstructorScoring],
        locked_in: Set[str],
        locked_out: Set[str],
    ) -> Iterator[TeamSolution]:
        """
        Generate valid team combinations.

        Parameters
        ----------
        driver_scores : Dict[str, DriverScoring]
            Scoring information for available drivers
        constructor_scores : Dict[str, ConstructorScoring]
            Scoring information for available constructors
        locked_in : Set[str]
            Set of element IDs that must be included
        locked_out : Set[str]
            Set of element IDs that cannot be included

        Yields
        ------
        TeamSolution
            Valid team composition

        Notes
        -----
        Uses progressive pruning:
        1. Generate driver combinations
        2. Check budget and constraints
        3. Select best talent driver
        4. Add valid constructors
        """
        # Get driver IDs for drivers and constructors
        driver_ids = set(driver_scores.keys())
        constructor_ids = set(constructor_scores.keys())

        # Split locked-in elements into drivers and constructors
        locked_in_drivers = locked_in & driver_ids
        locked_in_constructors = locked_in & constructor_ids

        # Must include locked-in drivers
        required_drivers = locked_in_drivers
        optional_drivers = driver_ids - required_drivers

        # Need to select (5 - len(required_drivers)) additional drivers
        remaining_driver_slots = 5 - len(required_drivers)

        if remaining_driver_slots < 0:
            # Too many locked-in drivers
            return

        # Calculate penalty for excluded locked-in drivers
        locked_in_cost = sum(
            driver_scores[d].salary * LOCKED_IN_PENALTY for d in locked_in_drivers
        )

        # Try each valid driver combination
        for extra_drivers in combinations(optional_drivers, remaining_driver_slots):
            drivers = frozenset(required_drivers | set(extra_drivers))

            # Calculate driver cost
            driver_cost = sum(driver_scores[d].salary for d in drivers)
            base_cost = driver_cost + locked_in_cost

            if base_cost > self.budget:
                # Skip if over budget before constructor
                continue

            # Try each constructor within budget
            remaining_budget = self.budget - base_cost

            for const_id, const_score in constructor_scores.items():
                if const_id in locked_out:
                    continue

                if (
                    const_id not in locked_in_constructors
                    and const_score.salary > remaining_budget
                ):
                    continue

                # Find best talent driver(s)
                talent_eligible = [d for d in drivers if driver_scores[d].can_be_talent]

                if not talent_eligible:
                    # No eligible talent drivers, generate solution without talent
                    points_breakdown = {}
                    total_points = const_score.points
                    points_breakdown[const_id] = const_score.points_dict

                    # Add regular driver points
                    for driver_id in drivers:
                        driver_points = driver_scores[driver_id].points_dict
                        points_breakdown[driver_id] = driver_points
                        total_points += sum(driver_points.values())

                    total_cost = base_cost + const_score.salary

                    yield TeamSolution(
                        drivers=drivers,
                        constructor=const_id,
                        talent_driver="",
                        expected_points=total_points,
                        total_cost=total_cost,
                        points_breakdown=points_breakdown,
                    )
                    continue

                # Find maximum points among eligible drivers
                max_points = max(
                    driver_scores[d].regular_points for d in talent_eligible
                )

                # Get all drivers with maximum points
                best_talent_drivers = [
                    d
                    for d in talent_eligible
                    if np.isclose(
                        driver_scores[d].regular_points, max_points, rtol=1e-6
                    )
                ]

                # Generate solution for each potential talent driver
                for talent_id in best_talent_drivers:
                    points_breakdown = {}
                    total_points = const_score.points
                    points_breakdown[const_id] = const_score.points_dict

                    # Add driver points
                    for driver_id in drivers:
                        if driver_id == talent_id:
                            # Double points for talent driver
                            driver_points = {
                                k: v * 2
                                for k, v in driver_scores[driver_id].points_dict.items()
                            }
                            points_breakdown[driver_id] = driver_points
                            total_points += sum(driver_points.values())
                        else:
                            # Regular points
                            driver_points = driver_scores[driver_id].points_dict
                            points_breakdown[driver_id] = driver_points
                            total_points += sum(driver_points.values())

                    total_cost = base_cost + const_score.salary

                    yield TeamSolution(
                        drivers=drivers,
                        constructor=const_id,
                        talent_driver=talent_id,
                        expected_points=total_points,
                        total_cost=total_cost,
                        points_breakdown=points_breakdown,
                    )
