"""
Expected points calculator for F1 fantasy optimization.

This module provides functionality to calculate expected fantasy points
based on probabilistic position distributions and scoring rules.
"""

from functools import lru_cache
from typing import Dict

import numpy as np

from gridrival_ai.data import get_teammate
from gridrival_ai.data.fantasy import RollingAverages
from gridrival_ai.data.reference import DriverId
from gridrival_ai.probabilities.position_distribution import PositionDistributions
from gridrival_ai.probabilities.types import JointProbabilities, SessionProbabilities
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import RaceFormat


class ExpectedPointsCalculator:
    """Calculator for expected fantasy points using position probabilities.

    This class calculates expected fantasy points for drivers and constructors
    based on their position probability distributions and GridRival scoring rules.

    Parameters
    ----------
    distributions : PositionDistributions
        Container with position probabilities
    scorer : Scorer
        Fantasy points calculator
    driver_stats : RollingAverages
        Historical driver statistics including rolling averages

    Notes
    -----
    Results are cached for efficiency using position probabilities as key.
    Teammate relationships are determined using data module.

    Examples
    --------
    >>> calculator = ExpectedPointsCalculator(distributions, scorer, driver_stats)
    >>> points = calculator.calculate_driver_points("VER")
    >>> print(points)  # Points breakdown by component
    {'qualifying': 45.2, 'race': 85.3, ...}
    """

    def __init__(
        self,
        distributions: PositionDistributions,
        scorer: Scorer,
        driver_stats: RollingAverages,
    ):
        """Initialize calculator with distributions and scoring rules."""
        self.distributions = distributions
        self.scorer = scorer
        self.driver_stats = driver_stats

    @lru_cache(maxsize=128)
    def calculate_driver_points(
        self, driver_id: DriverId, format: RaceFormat = RaceFormat.STANDARD
    ) -> Dict[str, float]:
        """Calculate expected points breakdown for a driver.

        Parameters
        ----------
        driver_id : DriverId
            Driver three-letter abbreviation
        format : RaceFormat, optional
            Race format, by default STANDARD

        Returns
        -------
        Dict[str, float]
            Expected points by component:
            - qualifying: Points from qualifying position
            - race: Points from race finish
            - sprint: Points from sprint (if applicable)
            - overtake: Points from positions gained
            - teammate: Points from beating teammate
            - completion: Points from race completion
            - improvement: Points from beating rolling average
        """
        qual_probs = self.distributions.get_session_probabilities(
            driver_id, "qualifying"
        )
        race_probs = self.distributions.get_session_probabilities(driver_id, "race")

        # Calculate base points components
        qual_points = self._calculate_position_points(qual_probs, session="qualifying")
        race_points = self._calculate_position_points(race_probs, session="race")

        # Calculate overtake points using joint distribution
        overtake_points = self._calculate_overtake_points(driver_id)

        # Calculate teammate points if teammate exists
        teammate_points = 0.0
        if teammate_id := get_teammate(driver_id):
            teammate_points = self._calculate_teammate_points(driver_id, teammate_id)

        # Calculate completion points
        completion_points = self._calculate_completion_points(driver_id)

        # Calculate improvement points using rolling average
        improvement_points = self._calculate_improvement_points(driver_id)

        # Combine results
        points = {
            "qualifying": qual_points,
            "race": race_points,
            "overtake": overtake_points,
            "teammate": teammate_points,
            "completion": completion_points,
            "improvement": improvement_points,
        }

        # Add sprint points if applicable
        if format == RaceFormat.SPRINT:
            sprint_probs = self.distributions.get_session_probabilities(
                driver_id, "sprint"
            )
            points["sprint"] = self._calculate_position_points(
                sprint_probs, session="sprint"
            )

        return points

    def _calculate_position_points(
        self, position_probs: SessionProbabilities, session: str
    ) -> float:
        """Calculate expected points for a single session.

        Parameters
        ----------
        position_probs : SessionProbabilities
            Position probabilities
        session : str
            Session name ("qualifying", "race", or "sprint")

        Returns
        -------
        float
            Expected points for session
        """
        if session == "qualifying":
            points_table = self.scorer.tables.driver_points[0]
        elif session == "race":
            points_table = self.scorer.tables.driver_points[1]
        else:  # sprint
            points_table = self.scorer.tables.driver_points[2]

        return sum(prob * points_table[pos] for pos, prob in position_probs.items())

    def _calculate_overtake_points(self, driver_id: DriverId) -> float:
        """Calculate expected overtake points.

        Parameters
        ----------
        driver_id : DriverId
            Driver three-letter abbreviation

        Returns
        -------
        float
            Expected overtake points
        """
        joint_probs: JointProbabilities = (
            self.distributions.get_driver_session_correlation(
                driver_id, "qualifying", "race"
            )
        )
        multiplier = self.scorer.tables.overtake_multiplier

        return sum(
            prob * max(0, qual_pos - race_pos) * multiplier
            for (qual_pos, race_pos), prob in joint_probs.items()
        )

    def _calculate_teammate_points(
        self, driver_id: DriverId, teammate_id: DriverId
    ) -> float:
        """Calculate expected points for beating teammate.

        Parameters
        ----------
        driver_id : DriverId
            Driver three-letter abbreviation
        teammate_id : DriverId
            Teammate's three-letter abbreviation

        Returns
        -------
        float
            Expected teammate points

        Notes
        -----
        Uses joint distribution between teammates that enforces the constraint
        that two drivers cannot finish in the same position.
        """
        # Get joint race probabilities for driver and teammate
        joint_probs = self.distributions.get_driver_pair_distribution(
            driver_id, teammate_id, session="race"
        )

        # Calculate points based on position differences
        total_points = 0.0
        thresholds = self.scorer.tables.teammate_thresholds

        for (d_pos, t_pos), prob in joint_probs.items():
            if d_pos < t_pos:  # Only if beating teammate
                margin = t_pos - d_pos
                # Find applicable threshold (first one not exceeded)
                idx = np.searchsorted(thresholds[:, 0], margin)
                if idx < len(thresholds):  # If we found a valid threshold
                    points = thresholds[idx, 1]
                    total_points += prob * points

        return total_points

    def _calculate_completion_points(self, driver_id: DriverId) -> float:
        """Calculate expected completion points for a driver.

        Parameters
        ----------
        driver_id : DriverId
            Driver three-letter abbreviation

        Returns
        -------
        float
            Expected completion points

        Notes
        -----
        The calculation assumes that DNFs occur uniformly across race laps.
        For a driver with completion probability p, the probability (1-p) of DNF
        is distributed uniformly across the race distance.

        For example, with thresholds [0.25, 0.5, 0.75, 0.9] and points_per_stage=3:
        - DNF before 0.25: 0 points (25% of DNF cases)
        - DNF between 0.25-0.5: 3 points (25% of DNF cases)
        - DNF between 0.5-0.75: 6 points (25% of DNF cases)
        - DNF between 0.75-0.9: 9 points (15% of DNF cases)
        - DNF between 0.9-1.0: 12 points (10% of DNF cases)
        - Complete race (no DNF): 12 points (100% of completion cases)
        """
        completion_prob = self.distributions.get_completion_probability(driver_id)
        thresholds = self.scorer.tables.completion_thresholds
        points_per_stage = self.scorer.tables.stage_points
        max_points = points_per_stage * len(thresholds)

        # Handle edge cases
        if completion_prob == 0.0:
            return 0.0
        if completion_prob == 1.0:
            return max_points

        # Points for completing the race
        total_points = completion_prob * max_points

        # Points for DNF cases
        dnf_prob = 1 - completion_prob
        prev_threshold = 0.0

        # For each threshold, calculate points if DNF occurs before reaching it
        for i, threshold in enumerate(thresholds):
            # Probability of DNF in this interval
            interval_prob = (threshold - prev_threshold) * dnf_prob
            # Points for completing all previous stages
            points = i * points_per_stage
            total_points += interval_prob * points
            prev_threshold = threshold

        # Points for DNF after last threshold (all stages completed)
        last_interval_prob = (1.0 - prev_threshold) * dnf_prob
        total_points += last_interval_prob * max_points

        return total_points

    def _calculate_improvement_points(self, driver_id: DriverId) -> float:
        """Calculate expected improvement points vs rolling average.

        Parameters
        ----------
        driver_id : DriverId
            Driver three-letter abbreviation

        Returns
        -------
        float
            Expected improvement points
        """
        rolling_avg = self.driver_stats.values[driver_id]
        race_probs = self.distributions.get_session_probabilities(driver_id, "race")

        # Calculate expected improvement points
        total_points = 0.0
        improvement_points = self.scorer.tables.improvement_points

        for pos, prob in race_probs.items():
            positions_ahead = max(0, round(rolling_avg) - pos)
            if positions_ahead > 0:
                # Find applicable improvement points
                points = improvement_points[
                    min(positions_ahead, len(improvement_points) - 1)
                ]
                total_points += prob * points

        return total_points

    @lru_cache(maxsize=128)
    def calculate_constructor_points(
        self, constructor_id: str, format: RaceFormat = RaceFormat.STANDARD
    ) -> Dict[str, float]:
        """Calculate expected points breakdown for a constructor.

        This method calculates the expected fantasy points for a constructor using
        joint probability distributions for the two drivers' positions. Qualifying
        points are calculated independently since positions can overlap, while race
        points use joint distributions to respect the constraint that drivers cannot
        finish in the same position.

        Parameters
        ----------
        constructor_id : str
            Constructor identifier (e.g., "RBR" for Red Bull Racing)
        format : RaceFormat, optional
            Race weekend format, by default RaceFormat.STANDARD
            Note: Format only affects race weekend structure, not scoring

        Returns
        -------
        Dict[str, float]
            Expected points breakdown by component:
            - qualifying: Points from both drivers' qualifying positions
            - race: Points from both drivers' race finishing positions

        Notes
        -----
        The calculation uses:
        - Independent distributions for qualifying (positions can overlap)
        - Joint distributions for race (positions must be different)
        - Pre-calculated lookup tables from the scoring configuration

        Points are calculated as expected values:
        E[Points] = Σ P(pos1, pos2) * (Points(pos1) + Points(pos2))

        Examples
        --------
        >>> calculator = ExpectedPointsCalculator(...)
        >>> points = calculator.calculate_constructor_points("RBR")
        >>> print(points)
        {'qualifying': 45.2, 'race': 85.3}
        """
        # Get drivers for this constructor
        drivers = self.distributions.get_constructor_drivers(constructor_id)
        if not drivers or len(drivers) != 2:
            return {"qualifying": 0.0, "race": 0.0}

        driver1_id, driver2_id = drivers

        # Calculate qualifying points (positions can overlap)
        qual_points = 0.0
        for driver_id in drivers:
            qual_probs = self.distributions.get_session_probabilities(
                driver_id, "qualifying"
            )
            qual_points += sum(
                prob * self.scorer.tables.constructor_points[0, pos]
                for pos, prob in qual_probs.items()
            )

        # Calculate race points using joint distribution (positions must differ)
        race_points = 0.0
        joint_race_probs = self.distributions.get_driver_pair_distribution(
            driver1_id, driver2_id, session="race"
        )

        # For each possible combination of positions
        for (pos1, pos2), prob in joint_race_probs.items():
            # Calculate points for this combination
            combo_points = (
                self.scorer.tables.constructor_points[1, pos1]  # Driver 1 points
                + self.scorer.tables.constructor_points[1, pos2]  # Driver 2 points
            )
            race_points += prob * combo_points

        return {
            "qualifying": qual_points,
            "race": race_points,
        }
