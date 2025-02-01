"""
Expected points calculator for F1 fantasy optimization.

This module provides functionality to calculate expected fantasy points
based on probabilistic position distributions and scoring rules.
"""

from functools import lru_cache
from typing import Dict

import numpy as np

from gridrival_ai.f1 import get_teammate
from gridrival_ai.f1.data import DriverId
from gridrival_ai.probabilities.position_distribution import PositionDistributions
from gridrival_ai.probabilities.types import JointProbabilities, SessionProbabilities
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import RaceFormat
from gridrival_ai.utils.driver_stats import DriverStats


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
    driver_stats : DriverStats
        Historical driver statistics including rolling averages

    Notes
    -----
    Results are cached for efficiency using position probabilities as key.
    Teammate relationships are determined using F1 data module.

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
        driver_stats: DriverStats,
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
            points_table = self.scorer.tables.base_points[0]
        elif session == "race":
            points_table = self.scorer.tables.base_points[1]
        else:  # sprint
            points_table = self.scorer.tables.base_points[2]

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
        """
        # Get race probabilities for both drivers
        driver_probs = self.distributions.get_session_probabilities(driver_id, "race")
        teammate_probs = self.distributions.get_session_probabilities(
            teammate_id, "race"
        )

        # Calculate probability of each position difference
        total_points = 0.0
        thresholds = self.scorer.tables.teammate_thresholds

        for d_pos, d_prob in driver_probs.items():
            for t_pos, t_prob in teammate_probs.items():
                if d_pos < t_pos:  # Only if beating teammate
                    margin = t_pos - d_pos
                    # Find applicable threshold (first one not exceeded)
                    idx = np.searchsorted(thresholds[:, 0], margin)
                    if idx < len(thresholds):  # If we found a valid threshold
                        points = thresholds[idx, 1]
                        total_points += d_prob * t_prob * points

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
        """
        completion_prob = self.distributions.get_completion_probability(driver_id)
        n_stages = len(self.scorer.tables.completion_thresholds)
        stage_prob = completion_prob ** np.arange(1, n_stages + 1)
        return np.sum(stage_prob) * self.scorer.tables.stage_points

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
        rolling_avg = self.driver_stats.rolling_averages[driver_id]
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
