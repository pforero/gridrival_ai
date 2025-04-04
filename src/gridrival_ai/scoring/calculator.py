"""
Simplified calculator for F1 fantasy points.

This module provides a streamlined calculator for F1 fantasy points without
complex validation, configuration loading, or unnecessary abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from gridrival_ai.probabilities.distributions import (
    JointDistribution,
    PositionDistribution,
    RaceDistribution,
)
from gridrival_ai.scoring.constants import (
    COMPLETION_STAGE_POINTS,
    COMPLETION_THRESHOLDS,
    CONSTRUCTOR_QUALIFYING_POINTS,
    CONSTRUCTOR_RACE_POINTS,
    IMPROVEMENT_POINTS,
    OVERTAKE_MULTIPLIER,
    QUALIFYING_POINTS,
    RACE_POINTS,
    SPRINT_POINTS,
    TEAMMATE_POINTS,
    RaceFormat,
)


@dataclass
class DriverPointsBreakdown:
    """Detailed breakdown of driver points by component."""

    qualifying: float = 0.0
    race: float = 0.0
    sprint: float = 0.0
    overtake: float = 0.0
    improvement: float = 0.0
    teammate: float = 0.0
    completion: float = 0.0

    @property
    def total(self) -> float:
        """Calculate total points across all components."""
        return (
            self.qualifying
            + self.race
            + self.sprint
            + self.overtake
            + self.improvement
            + self.teammate
            + self.completion
        )


class ScoringCalculator:
    """
    Simplified calculator for F1 fantasy points.

    This class provides methods for calculating fantasy points for drivers
    and constructors based on race results and expected points based on
    probability distributions.

    Examples
    --------
    >>> # Calculate points for a specific driver scenario
    >>> calculator = ScoringCalculator()
    >>> points = calculator.calculate_driver_points(
    ...     qualifying_pos=1,
    ...     race_pos=1,
    ...     rolling_avg=2.5,
    ...     teammate_pos=3,
    ...     completion_pct=1.0
    ... )
    >>> print(f"Total points: {points.total}")
    Total points: 166.0
    """

    def calculate_driver_points(
        self,
        qualifying_pos: int,
        race_pos: int,
        rolling_avg: float,
        teammate_pos: int,
        completion_pct: float = 1.0,
        sprint_pos: Optional[int] = None,
        race_format: RaceFormat = "STANDARD",
    ) -> DriverPointsBreakdown:
        """
        Calculate points for a driver with given positions.

        Parameters
        ----------
        qualifying_pos : int
            Qualifying position (1-20)
        race_pos : int
            Race finish position (1-20)
        rolling_avg : float
            8-race rolling average finish position
        teammate_pos : int
            Teammate's race finish position (1-20)
        completion_pct : float, optional
            Race completion percentage (0.0 to 1.0), by default 1.0
        sprint_pos : Optional[int], optional
            Sprint race position (1-8) for sprint weekends, by default None
        race_format : STANDARD or SPRINT, optional
            Race weekend format, by default STANDARD

        Returns
        -------
        DriverPointsBreakdown
            Detailed breakdown of points by component
        """
        # Initialize results
        breakdown = DriverPointsBreakdown()

        # Qualifying points
        breakdown.qualifying = QUALIFYING_POINTS.get(qualifying_pos, 0.0)

        # Race points
        breakdown.race = RACE_POINTS.get(race_pos, 0.0)

        # Sprint points if applicable
        if race_format == "SPRINT" and sprint_pos is not None:
            breakdown.sprint = SPRINT_POINTS.get(sprint_pos, 0.0)

        # Overtake points
        positions_gained = max(0, qualifying_pos - race_pos)
        breakdown.overtake = positions_gained * OVERTAKE_MULTIPLIER

        # Improvement points
        positions_ahead = max(0, round(rolling_avg) - race_pos)
        if positions_ahead > 0:
            breakdown.improvement = IMPROVEMENT_POINTS.get(
                positions_ahead,
                IMPROVEMENT_POINTS.get(max(IMPROVEMENT_POINTS.keys()), 0.0),
            )

        # Teammate points
        if race_pos < teammate_pos:
            margin = teammate_pos - race_pos
            # Find applicable threshold
            thresholds = sorted(TEAMMATE_POINTS.items())
            for threshold, points in thresholds:
                if margin <= threshold:
                    breakdown.teammate = points
                    break
            else:
                # If beyond all thresholds, use the last one
                if thresholds:
                    breakdown.teammate = thresholds[-1][1]

        # Completion points
        stages_completed = 0
        for threshold in sorted(COMPLETION_THRESHOLDS):
            if completion_pct >= threshold:
                stages_completed += 1
        breakdown.completion = stages_completed * COMPLETION_STAGE_POINTS

        return breakdown

    def calculate_constructor_points(
        self,
        driver1_qualifying: int,
        driver1_race: int,
        driver2_qualifying: int,
        driver2_race: int,
    ) -> Dict[str, float]:
        """
        Calculate points for a constructor with given positions.

        Parameters
        ----------
        driver1_qualifying : int
            First driver qualifying position (1-20)
        driver1_race : int
            First driver race position (1-20)
        driver2_qualifying : int
            Second driver qualifying position (1-20)
        driver2_race : int
            Second driver race position (1-20)

        Returns
        -------
        Dict[str, float]
            Points breakdown by component
        """
        # Initialize results
        result = {
            "qualifying": 0.0,
            "race": 0.0,
        }

        # Qualifying points (sum of both drivers)
        result["qualifying"] += CONSTRUCTOR_QUALIFYING_POINTS.get(
            driver1_qualifying, 0.0
        )
        result["qualifying"] += CONSTRUCTOR_QUALIFYING_POINTS.get(
            driver2_qualifying, 0.0
        )

        # Race points (sum of both drivers)
        result["race"] += CONSTRUCTOR_RACE_POINTS.get(driver1_race, 0.0)
        result["race"] += CONSTRUCTOR_RACE_POINTS.get(driver2_race, 0.0)

        return result

    def expected_driver_points(
        self,
        qual_dist: PositionDistribution,
        race_dist: PositionDistribution,
        rolling_avg: float,
        teammate_dist: PositionDistribution,
        completion_prob: float = 0.95,
        sprint_dist: Optional[PositionDistribution] = None,
        race_format: RaceFormat = "STANDARD",
        joint_qual_race: Optional[JointDistribution] = None,
    ) -> DriverPointsBreakdown:
        """
        Calculate expected points breakdown from probability distributions.

        Parameters
        ----------
        qual_dist : PositionDistribution
            Distribution over qualifying positions
        race_dist : PositionDistribution
            Distribution over race positions
        rolling_avg : float
            8-race rolling average finish position
        teammate_dist : PositionDistribution
            Distribution over teammate's race positions
        completion_prob : float, optional
            Probability of completing the race, by default 0.95
        sprint_dist : Optional[PositionDistribution], optional
            Distribution over sprint positions, by default None
        race_format : STANDARD or SPRINT, optional
            Race weekend format, by default STANDARD
        joint_qual_race : Optional[JointDistribution], optional
            Joint distribution between qualifying and race positions.
            If provided, this will be used for overtake points calculations.
            If None, assumes independence between qual_dist and race_dist.

        Returns
        -------
        DriverPointsBreakdown
            Detailed breakdown of expected points by component
        """
        # Initialize component points
        result = DriverPointsBreakdown()

        # Calculate qualifying points
        result.qualifying = self._expected_position_points(qual_dist, QUALIFYING_POINTS)

        # Calculate race points
        result.race = self._expected_position_points(race_dist, RACE_POINTS)

        # Calculate sprint points if applicable
        if race_format == "SPRINT" and sprint_dist:
            result.sprint = self._expected_position_points(sprint_dist, SPRINT_POINTS)

        # Calculate overtake points
        if joint_qual_race:
            # Use joint distribution for overtakes if provided
            result.overtake = self._expected_overtake_points(joint_qual_race)
        else:
            # Otherwise create independent joint distribution
            result.overtake = self._expected_overtake_points_from_marginals(
                qual_dist, race_dist
            )

        # Calculate improvement points
        result.improvement = self._expected_improvement_points(race_dist, rolling_avg)

        # Calculate teammate points
        result.teammate = self._expected_teammate_points(race_dist, teammate_dist)

        # Calculate completion points
        result.completion = self._expected_completion_points(completion_prob)

        return result

    def expected_constructor_points(
        self,
        driver1_qual_dist: PositionDistribution,
        driver1_race_dist: PositionDistribution,
        driver2_qual_dist: PositionDistribution,
        driver2_race_dist: PositionDistribution,
    ) -> Dict[str, float]:
        """
        Calculate expected constructor points from probability distributions.

        Parameters
        ----------
        driver1_qual_dist : PositionDistribution
            First driver's qualifying position distribution
        driver1_race_dist : PositionDistribution
            First driver's race position distribution
        driver2_qual_dist : PositionDistribution
            Second driver's qualifying position distribution
        driver2_race_dist : PositionDistribution
            Second driver's race position distribution

        Returns
        -------
        Dict[str, float]
            Expected points breakdown by component
        """
        # Initialize components
        result = {"qualifying": 0.0, "race": 0.0}

        # Calculate qualifying points for both drivers
        result["qualifying"] += self._expected_position_points(
            driver1_qual_dist, CONSTRUCTOR_QUALIFYING_POINTS
        )
        result["qualifying"] += self._expected_position_points(
            driver2_qual_dist, CONSTRUCTOR_QUALIFYING_POINTS
        )

        # Calculate race points for both drivers
        result["race"] += self._expected_position_points(
            driver1_race_dist, CONSTRUCTOR_RACE_POINTS
        )
        result["race"] += self._expected_position_points(
            driver2_race_dist, CONSTRUCTOR_RACE_POINTS
        )

        return result

    def expected_driver_points_from_race_distribution(
        self,
        race_dist: RaceDistribution,
        driver_id: str,
        rolling_avg: float,
        teammate_id: str,
        race_format: RaceFormat = "STANDARD",
        completion_prob: float = 0.95,  # Fallback if not in race_dist
    ) -> DriverPointsBreakdown:
        """
        Calculate expected points breakdown from a race distribution.

        Parameters
        ----------
        race_dist : RaceDistribution
            Distribution for all sessions in the race weekend
        driver_id : str
            Driver ID
        rolling_avg : float
            8-race rolling average finish position
        teammate_id : str
            ID of teammate driver
        race_format : STANDARD or SPRINT, optional
            Race weekend format, by default STANDARD
        completion_prob : float, optional
            Fallback completion probability if not found in race_dist, by default 0.95

        Returns
        -------
        DriverPointsBreakdown
            Detailed breakdown of expected points by component
        """
        # Get distributions
        qual_dist = race_dist.get_driver_distribution(driver_id, "qualifying")
        race_dist_driver = race_dist.get_driver_distribution(driver_id, "race")

        # Get sprint distribution if applicable
        sprint_dist = None
        if race_format == "SPRINT":
            try:
                sprint_dist = race_dist.get_driver_distribution(driver_id, "sprint")
            except (KeyError, ValueError):
                pass

        # Get teammate distribution
        try:
            teammate_dist = race_dist.get_driver_distribution(teammate_id, "race")
            joint_qual_race = race_dist.get_qualifying_race_distribution(driver_id)
        except (KeyError, ValueError):
            teammate_dist = None
            joint_qual_race = None

        # Try to get completion probability
        try:
            completion_prob_value = race_dist.get_completion_probability(driver_id)
        except KeyError:
            completion_prob_value = completion_prob

        # Calculate expected points
        if teammate_dist:
            return self.expected_driver_points(
                qual_dist=qual_dist,
                race_dist=race_dist_driver,
                rolling_avg=rolling_avg,
                teammate_dist=teammate_dist,
                completion_prob=completion_prob_value,
                sprint_dist=sprint_dist,
                race_format=race_format,
                joint_qual_race=joint_qual_race,
            )
        else:
            # Create a simple teammate distribution centered on the average position
            avg_pos = sum(p * pos for pos, p in race_dist_driver.items())
            simple_teammate_dist = PositionDistribution({int(avg_pos): 1.0})

            return self.expected_driver_points(
                qual_dist=qual_dist,
                race_dist=race_dist_driver,
                rolling_avg=rolling_avg,
                teammate_dist=simple_teammate_dist,
                completion_prob=completion_prob_value,
                sprint_dist=sprint_dist,
                race_format=race_format,
                joint_qual_race=joint_qual_race,
            )

    # Helper methods for expected value calculations
    def _expected_position_points(
        self, dist: PositionDistribution, points_mapping: Dict[int, float]
    ) -> float:
        """Calculate expected points for a position distribution."""
        return sum(prob * points_mapping.get(pos, 0.0) for pos, prob in dist.items())

    def _expected_overtake_points(self, joint_dist: JointDistribution) -> float:
        """Calculate expected overtake points from joint distribution."""
        return sum(
            prob * max(0, qual_pos - race_pos) * OVERTAKE_MULTIPLIER
            for (qual_pos, race_pos), prob in joint_dist.items()
        )

    def _expected_overtake_points_from_marginals(
        self, qual_dist: PositionDistribution, race_dist: PositionDistribution
    ) -> float:
        """Calculate expected overtake points assuming independence."""
        expected_points = 0.0
        for qual_pos, qual_prob in qual_dist.items():
            for race_pos, race_prob in race_dist.items():
                joint_prob = qual_prob * race_prob
                positions_gained = max(0, qual_pos - race_pos)
                expected_points += joint_prob * positions_gained * OVERTAKE_MULTIPLIER
        return expected_points

    def _expected_improvement_points(
        self, race_dist: PositionDistribution, rolling_avg: float
    ) -> float:
        """Calculate expected improvement points vs rolling average."""
        rounded_avg = round(rolling_avg)
        expected_points = 0.0

        for pos, prob in race_dist.items():
            positions_ahead = max(0, rounded_avg - pos)
            if positions_ahead > 0:
                improvement_points = IMPROVEMENT_POINTS.get(
                    positions_ahead,
                    IMPROVEMENT_POINTS.get(max(IMPROVEMENT_POINTS.keys()), 0.0),
                )
                expected_points += prob * improvement_points

        return expected_points

    def _expected_teammate_points(
        self, driver_dist: PositionDistribution, teammate_dist: PositionDistribution
    ) -> float:
        """Calculate expected teammate points from distributions."""
        expected_points = 0.0
        thresholds = sorted(TEAMMATE_POINTS.items())

        for driver_pos, driver_prob in driver_dist.items():
            for tm_pos, tm_prob in teammate_dist.items():
                if driver_pos < tm_pos:
                    margin = tm_pos - driver_pos
                    points = 0.0

                    for threshold, threshold_points in thresholds:
                        if margin <= threshold:
                            points = threshold_points
                            break

                    if not points and thresholds:
                        points = thresholds[-1][1]

                    expected_points += driver_prob * tm_prob * points

        return expected_points

    def _expected_completion_points(self, completion_prob: float) -> float:
        """Calculate expected completion points based on completion probability."""
        # Full completion points (all stages)
        max_points = len(COMPLETION_THRESHOLDS) * COMPLETION_STAGE_POINTS

        # Points for completing the race
        if completion_prob >= 1.0:
            return max_points

        points_if_completed = completion_prob * max_points

        # DNF probability
        dnf_prob = 1.0 - completion_prob

        # Expected DNF points assuming uniform distribution of DNFs across race distance
        expected_dnf_points = 0.0
        prev_threshold = 0.0

        # Calculate points for each stage
        for i, threshold in enumerate(sorted(COMPLETION_THRESHOLDS)):
            # Probability of DNF in this interval (relative to overall DNF prob)
            interval_prob = (threshold - prev_threshold) / 1.0 * dnf_prob

            # Points for completing all previous stages
            points = i * COMPLETION_STAGE_POINTS
            expected_dnf_points += interval_prob * points

            prev_threshold = threshold

        # DNF after last threshold but before finish
        last_interval_prob = (1.0 - prev_threshold) / 1.0 * dnf_prob
        expected_dnf_points += last_interval_prob * max_points

        return points_if_completed + expected_dnf_points
