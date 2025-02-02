"""Team optimization for F1 fantasy league.

This module provides optimization functionality for finding the best possible
F1 fantasy team composition given budget and roster constraints.

Key Features
-----------
- Supports both sprint and standard race formats
- Handles locked-in and locked-out driver constraints
- Automatic talent driver selection (highest points among eligible drivers)
- Multiple optimal solutions tracking
- Progressive combination pruning for efficiency

Notes
-----
The optimizer uses a combination of pre-calculation and pruning:
1. Pre-calculates all driver and constructor points
2. Generates valid combinations with budget pruning
3. Evaluates remaining combinations for optimal solutions

Locked-in drivers incur a 3% salary penalty if not included in team.
Talent drivers double their regular points.
"""

from itertools import combinations
from typing import Iterator

import numpy as np

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.expected_points import ExpectedPointsCalculator
from gridrival_ai.optimization.types import (
    DriverScoring,
    OptimizationResult,
    TeamSolution,
)
from gridrival_ai.scoring.types import RaceFormat

# Constants
TALENT_SALARY_THRESHOLD = 18.0  # Maximum salary for talent drivers
LOCKED_IN_PENALTY = 0.03  # 3% penalty for not using locked-in drivers


class TeamOptimizer:
    """Optimizer for F1 fantasy team composition.

    Parameters
    ----------
    league_data : FantasyLeagueData
        Current league state including salaries and constraints
    points_calculator : ExpectedPointsCalculator
        Calculator for expected points
    race_format : RaceFormat
        Format of the race weekend (sprint/standard)
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
    >>> optimizer = TeamOptimizer(
    ...     league_data=league_data,
    ...     points_calculator=calculator,
    ...     race_format=RaceFormat.STANDARD,
    ...     budget=100.0
    ... )
    >>> result = optimizer.optimize()
    >>> if result.best_solution:
    ...     print(f"Best team: {result.best_solution.drivers}")
    ...     print(f"Expected points: {result.best_solution.expected_points}")
    """

    def __init__(
        self,
        league_data: FantasyLeagueData,
        points_calculator: ExpectedPointsCalculator,
        race_format: RaceFormat,
        budget: float = 100.0,
    ) -> None:
        """Initialize optimizer with league state."""
        self.league_data = league_data
        self.points_calculator = points_calculator
        self.race_format = race_format
        self.budget = budget

        # Pre-calculate expected points
        self.driver_scores: dict[str, DriverScoring] = {}
        self.constructor_points: dict[str, float] = {}
        self.constructor_salaries: dict[str, float] = {}

        self._precalculate_points()

    def _precalculate_points(self) -> None:
        """Pre-calculate expected points for all elements.

        Calculates and stores:
        - Regular driver points
        - Constructor points
        - Talent driver eligibility

        Notes
        -----
        Points are calculated once at initialization for efficiency.
        Talent driver points are computed by doubling regular points.
        """
        # Calculate driver points
        for driver_id in self.league_data.get_available_drivers():
            salary = self.league_data.salaries.drivers[driver_id]
            can_be_talent = salary <= TALENT_SALARY_THRESHOLD

            # Calculate regular points
            regular = self.points_calculator.calculate_driver_points(
                driver_id, format=self.race_format
            )

            self.driver_scores[driver_id] = DriverScoring(
                regular_points=regular,
                salary=salary,
                can_be_talent=can_be_talent,
            )

        # Calculate constructor points
        for const_id in self.league_data.get_available_constructors():
            points = self.points_calculator.calculate_constructor_points(const_id)
            self.constructor_points[const_id] = points
            self.constructor_salaries[const_id] = (
                self.league_data.salaries.constructors[const_id]
            )

    def _generate_valid_combinations(self) -> Iterator[TeamSolution]:
        """Generate valid team combinations using pruning.

        Yields
        ------
        TeamSolution
            Valid team composition

        Notes
        -----
        Uses progressive pruning:
        1. Generate driver combinations
        2. Check budget and constraints
        3. Select best talent driver(s):
           - If one driver has highest points, select them
           - If multiple drivers tie for highest points, try each
        4. Add valid constructors

        Combinations are pruned if they:
        - Exceed budget
        - Violate roster constraints
        """
        available_drivers = set(self.driver_scores.keys())

        # Must include locked-in drivers
        required = self.league_data.constraints.locked_in
        optional = available_drivers - required

        # Need to select (5 - len(required)) additional drivers
        remaining_slots = 5 - len(required)

        if remaining_slots < 0:
            # Too many locked-in drivers
            return

        # Calculate penalty for excluded locked-in drivers
        locked_in_cost = sum(
            self.driver_scores[d].salary * LOCKED_IN_PENALTY for d in required
        )

        # Try each valid combination
        for extra_drivers in combinations(optional, remaining_slots):
            drivers = frozenset(required | set(extra_drivers))

            # Calculate driver cost
            driver_cost = sum(self.driver_scores[d].salary for d in drivers)
            base_cost = driver_cost + locked_in_cost

            if base_cost >= self.budget:
                # Skip if over budget before constructor
                continue

            # Try each constructor within budget
            remaining_budget = self.budget - base_cost

            for const_id, const_salary in self.constructor_salaries.items():
                if const_salary > remaining_budget:
                    continue

                # Find best talent driver(s) if any are eligible
                talent_eligible = [
                    d for d in drivers if self.driver_scores[d].can_be_talent
                ]

                if not talent_eligible:
                    # No eligible talent drivers, generate solution without talent
                    points_breakdown = {}
                    total_points = self.constructor_points[const_id]
                    points_breakdown[const_id] = total_points

                    # Add regular driver points
                    for driver_id in drivers:
                        driver_points = self.driver_scores[driver_id].regular_points
                        points_breakdown[driver_id] = driver_points
                        total_points += driver_points

                    total_cost = base_cost + const_salary

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
                    self.driver_scores[d].regular_points for d in talent_eligible
                )

                # Get all drivers with maximum points
                best_talent_drivers = [
                    d
                    for d in talent_eligible
                    if np.isclose(
                        self.driver_scores[d].regular_points,
                        max_points,
                        rtol=1e-6,
                    )
                ]

                # Generate solution for each potential talent driver
                for talent_id in best_talent_drivers:
                    points_breakdown = {}
                    total_points = self.constructor_points[const_id]
                    points_breakdown[const_id] = total_points

                    # Add driver points
                    for driver_id in drivers:
                        if driver_id == talent_id:
                            # Double points for talent driver
                            driver_points = (
                                self.driver_scores[driver_id].regular_points * 2
                            )
                        else:
                            # Regular points
                            driver_points = self.driver_scores[driver_id].regular_points

                        points_breakdown[driver_id] = driver_points
                        total_points += driver_points

                    total_cost = base_cost + const_salary

                    yield TeamSolution(
                        drivers=drivers,
                        constructor=const_id,
                        talent_driver=talent_id,
                        expected_points=total_points,
                        total_cost=total_cost,
                        points_breakdown=points_breakdown,
                    )

    def optimize(self) -> OptimizationResult:
        """Find optimal team composition.

        Returns
        -------
        OptimizationResult
            Optimization result including best and alternative solutions

        Notes
        -----
        Evaluates all valid combinations to find global optimum.
        Tracks alternative solutions with equal expected points.
        Solutions are considered equal if their expected points differ
        by less than 1e-6.
        """
        best_solution = None
        alternative_solutions = []
        max_points = 0.0

        try:
            for solution in self._generate_valid_combinations():
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
                    error_message="No valid team composition found",
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
