"""
Team optimization for F1 fantasy leagues with the GridRival format.

This module provides functionality for optimizing F1 fantasy team compositions
based on expected points and budget constraints using a brute force approach
with progressive pruning for efficiency.
"""

from itertools import combinations
from typing import Dict, Iterator, Set

import numpy as np

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.data.reference import CONSTRUCTORS, get_teammate
from gridrival_ai.optimization.types import (
    ConstructorScoring,
    DriverScoring,
    OptimizationResult,
    TeamSolution,
)
from gridrival_ai.probabilities.distributions import RaceDistribution
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.constants import RaceFormat

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
    scorer : ScoringCalculator
        Calculator for expected points using the new points API
    race_distribution : RaceDistribution
        Distribution containing probabilities for all sessions and drivers
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
    >>> from gridrival_ai.scoring.calculator import ScoringCalculator
    >>> from gridrival_ai.probabilities.distributions import RaceDistribution
    >>>
    >>> # Create prerequisites
    >>> league_data = FantasyLeagueData.from_dicts(
    ...     driver_salaries={"VER": 33.0, "HAM": 26.2},
    ...     constructor_salaries={"RBR": 30.0, "MER": 27.2},
    ...     rolling_averages={"VER": 1.5, "HAM": 3.0}
    ... )
    >>>
    >>> # Create race distribution
    >>> odds_data = {...}  # Dictionary of betting odds
    >>> race_dist = RaceDistribution.from_structured_odds(odds_data)
    >>>
    >>> # Create scoring calculator
    >>> scorer = ScoringCalculator()
    >>>
    >>> # Create optimizer
    >>> optimizer = TeamOptimizer(
    ...     league_data=league_data,
    ...     scorer=scorer,
    ...     race_distribution=race_dist,
    ...     driver_stats=driver_stats
    ... )
    >>>
    >>> # Optimize team
    >>> result = optimizer.optimize(race_format="STANDARD")
    >>>
    >>> # Access best team
    >>> if result.best_solution:
    ...     print(f"Best team: {result.best_solution.drivers}")
    ...     print(f"Expected points: {result.best_solution.expected_points}")
    """

    def __init__(
        self,
        league_data: FantasyLeagueData,
        scorer: ScoringCalculator,
        race_distribution: RaceDistribution,
        driver_stats: Dict[str, float],
        budget: float = 100.0,
    ) -> None:
        """Initialize optimizer with league state."""
        self.league_data = league_data
        self.scorer = scorer
        self.race_distribution = race_distribution
        self.driver_stats = driver_stats
        self.budget = budget

    def optimize(
        self,
        race_format: RaceFormat = "STANDARD",
        locked_in: set[str] | None = None,
        locked_out: set[str] | None = None,
    ) -> OptimizationResult:
        """
        Find optimal team composition.

        This method calculates all valid team combinations within budget and returns
        an OptimizationResult that can be filtered based on constraints.

        Parameters
        ----------
        race_format : RaceFormat, optional
            Format of the race weekend, by default "STANDARD"
        locked_in : set[str] | None, optional
            Set of driver IDs that must be included, by default None
        locked_out : set[str] | None, optional
            Set of driver IDs that must be excluded, by default None

        Returns
        -------
        OptimizationResult
            Results object with all solutions and filtering capabilities

        Notes
        -----
        This method calculates ALL valid solutions within budget, including
        penalties for not using locked-in drivers.
        """
        # Initialize constraints
        locked_in = locked_in or set()
        locked_out = locked_out or set()

        # Pre-calculate driver scores for ALL drivers
        driver_scores = self._calculate_driver_scores(race_format)

        # Pre-calculate constructor scores for ALL constructors
        constructor_scores = self._calculate_constructor_scores()

        # Generate and evaluate all valid team combinations
        all_solutions = []

        # Generate all valid combinations within budget, applying locked-in penalties
        for solution in self._generate_valid_combinations(
            driver_scores, constructor_scores, locked_in
        ):
            all_solutions.append(solution)

        # Return result with all solutions and requested filters
        return OptimizationResult(
            all_solutions=all_solutions,
            locked_in=locked_in,
            locked_out=locked_out,
            budget=self.budget,
        )

    def _calculate_driver_scores(
        self, race_format: RaceFormat
    ) -> Dict[str, DriverScoring]:
        """
        Calculate driver scoring information.

        Parameters
        ----------
        race_format : RaceFormat
            Format of the race weekend

        Returns
        -------
        Dict[str, DriverScoring]
            Mapping of driver IDs to scoring information
        """
        driver_scores = {}

        # Get all available drivers
        available_drivers = self.league_data.get_available_drivers()

        # Calculate scores for each available driver
        for driver_id in available_drivers:
            # Check if driver has distributions
            try:
                self.race_distribution.get_driver_distribution(driver_id, "race")
            except KeyError:
                continue

            # Get driver salary and check talent eligibility
            salary = self.league_data.salaries.drivers[driver_id]
            can_be_talent = salary <= TALENT_SALARY_THRESHOLD

            # Get teammate ID using the reference function
            teammate_id = get_teammate(driver_id)

            # Calculate expected points using the scoring calculator
            points_breakdown = (
                self.scorer.expected_driver_points_from_race_distribution(
                    race_dist=self.race_distribution,
                    driver_id=driver_id,
                    rolling_avg=self.driver_stats[driver_id],
                    teammate_id=teammate_id,
                    race_format=race_format,
                )
            )

            # Convert points breakdown to dictionary for compatibility
            points_dict = {
                "qualifying": points_breakdown.qualifying,
                "race": points_breakdown.race,
                "sprint": points_breakdown.sprint,
                "overtake": points_breakdown.overtake,
                "improvement": points_breakdown.improvement,
                "teammate": points_breakdown.teammate,
                "completion": points_breakdown.completion,
            }

            driver_scores[driver_id] = DriverScoring(
                regular_points=points_breakdown.total,
                points_dict=points_dict,
                salary=salary,
                can_be_talent=can_be_talent,
            )

        return driver_scores

    def _calculate_constructor_scores(self) -> Dict[str, ConstructorScoring]:
        """
        Calculate constructor scoring information based on driver distributions.

        Returns
        -------
        Dict[str, ConstructorScoring]
            Mapping of constructor IDs to scoring information
        """
        constructor_scores = {}

        # Get all available constructors
        available_constructors = self.league_data.get_available_constructors()

        # Calculate scores for each available constructor
        for constructor_id in available_constructors:
            # Get constructor from reference data
            constructor = CONSTRUCTORS.get(constructor_id)
            if not constructor or len(constructor.drivers) < 2:
                continue

            # Get constructor salary
            salary = self.league_data.salaries.constructors[constructor_id]

            # Get the two drivers
            driver1_id, driver2_id = constructor.drivers

            # Get qualifying and race distributions for each driver
            try:
                driver1_qual_dist = self.race_distribution.get_driver_distribution(
                    driver1_id, "qualifying"
                )
                driver1_race_dist = self.race_distribution.get_driver_distribution(
                    driver1_id, "race"
                )
                driver2_qual_dist = self.race_distribution.get_driver_distribution(
                    driver2_id, "qualifying"
                )
                driver2_race_dist = self.race_distribution.get_driver_distribution(
                    driver2_id, "race"
                )
            except KeyError:
                # Skip if we're missing distributions
                continue

            # Calculate expected constructor points
            points_dict = self.scorer.expected_constructor_points(
                driver1_qual_dist=driver1_qual_dist,
                driver1_race_dist=driver1_race_dist,
                driver2_qual_dist=driver2_qual_dist,
                driver2_race_dist=driver2_race_dist,
            )

            # Sum up the points
            total_points = sum(points_dict.values())

            constructor_scores[constructor_id] = ConstructorScoring(
                points=total_points, points_dict=points_dict, salary=salary
            )

        return constructor_scores

    def _generate_valid_combinations(
        self,
        driver_scores: Dict[str, DriverScoring],
        constructor_scores: Dict[str, ConstructorScoring],
        locked_in: Set[str],
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
            Set of element IDs that must be included (for penalty calculation)

        Yields
        ------
        TeamSolution
            Valid team composition within budget

        Notes
        -----
        Uses progressive pruning:
        1. Generate driver combinations
        2. Apply penalties for locked-in drivers not included
        3. Check budget constraints
        4. Select best talent driver
        5. Add valid constructors
        """
        # Get driver IDs for drivers and constructors
        driver_ids = set(driver_scores.keys())
        constructor_ids = set(constructor_scores.keys())

        # Split locked-in elements into drivers and constructors
        locked_in_drivers = locked_in & driver_ids
        locked_in_constructors = locked_in & constructor_ids

        # Try each valid driver combination (always 5 drivers)
        for five_drivers in combinations(driver_ids, 5):
            drivers = frozenset(five_drivers)

            # Calculate driver cost
            driver_cost = sum(driver_scores[d].salary for d in drivers)

            # Calculate penalty for excluded locked-in drivers
            missing_locked_drivers = locked_in_drivers - drivers
            locked_in_penalty = sum(
                driver_scores[d].salary * LOCKED_IN_PENALTY
                for d in missing_locked_drivers
            )

            # Add penalty to base cost
            base_cost = driver_cost + locked_in_penalty

            if base_cost > self.budget:
                # Skip if over budget before constructor
                continue

            # Try each constructor within budget
            remaining_budget = self.budget - base_cost

            for const_id, const_score in constructor_scores.items():
                # Apply penalty for locked-in constructor if not selected
                missing_constructor_penalty = 0
                if const_id not in locked_in_constructors and locked_in_constructors:
                    # Constructor is locked in but we're using a different one
                    # Apply penalty for each locked-in constructor
                    for locked_const in locked_in_constructors:
                        if locked_const in constructor_scores:
                            missing_constructor_penalty += (
                                constructor_scores[locked_const].salary
                                * LOCKED_IN_PENALTY
                            )

                # Check if constructor is affordable including penalty
                if const_score.salary + missing_constructor_penalty > remaining_budget:
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

                    total_cost = (
                        base_cost + const_score.salary + missing_constructor_penalty
                    )

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

                    total_cost = (
                        base_cost + const_score.salary + missing_constructor_penalty
                    )

                    yield TeamSolution(
                        drivers=drivers,
                        constructor=const_id,
                        talent_driver=talent_id,
                        expected_points=total_points,
                        total_cost=total_cost,
                        points_breakdown=points_breakdown,
                    )
