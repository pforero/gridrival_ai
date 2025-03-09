"""
Driver points calculator for GridRival F1 fantasy.

This module provides the calculator for driver fantasy points,
handling all scoring components for individual drivers.
"""

from typing import Dict

from gridrival_ai.points.components import (
    CompletionPointsCalculator,
    ImprovementPointsCalculator,
    OvertakePointsCalculator,
    PositionPointsCalculator,
    TeammatePointsCalculator,
)
from gridrival_ai.points.distributions import DistributionAdapter
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import RaceFormat


class DriverPointsCalculator:
    """
    Calculate expected points for drivers.

    This calculator orchestrates the component calculators to compute
    the total expected points for a driver across all scoring elements.

    Parameters
    ----------
    distributions : DistributionAdapter
        Adapter for accessing probability distributions
    scorer : Scorer
        Scoring rules calculator

    Examples
    --------
    >>> adapter = DistributionAdapter(registry)
    >>> calculator = DriverPointsCalculator(adapter, scorer)
    >>> points = calculator.calculate(
    ...     driver_id="VER",
    ...     rolling_avg=1.5,
    ...     teammate_id="PER",
    ...     race_format=RaceFormat.STANDARD
    ... )
    >>> print(f"Total expected points: {sum(points.values()):.1f}")
    Total expected points: 156.5
    """

    def __init__(self, distributions: DistributionAdapter, scorer: Scorer):
        """Initialize with distributions and scorer."""
        self.distributions = distributions
        self.scorer = scorer

        # Initialize component calculators
        self.position_calculator = PositionPointsCalculator()
        self.overtake_calculator = OvertakePointsCalculator()
        self.teammate_calculator = TeammatePointsCalculator()
        self.completion_calculator = CompletionPointsCalculator()
        self.improvement_calculator = ImprovementPointsCalculator()

    def calculate(
        self,
        driver_id: str,
        rolling_avg: float,
        teammate_id: str,
        race_format: RaceFormat = RaceFormat.STANDARD,
    ) -> Dict[str, float]:
        """
        Calculate expected points breakdown for a driver.

        Parameters
        ----------
        driver_id : str
            Driver ID
        rolling_avg : float
            8-race rolling average finish position
        teammate_id : str
            Teammate driver ID
        race_format : RaceFormat, optional
            Type of race weekend, by default RaceFormat.STANDARD

        Returns
        -------
        Dict[str, float]
            Points breakdown by component

        Raises
        ------
        KeyError
            If required distributions not found
        """
        result = {}

        # Get qualifying distribution and calculate points
        qual_dist = self.distributions.get_position_distribution(
            driver_id, "qualifying"
        )
        result["qualifying"] = self.position_calculator.calculate(
            qual_dist, self.scorer.tables.driver_points[0]  # Qualifying points table
        )

        # Get race distribution and calculate points
        race_dist = self.distributions.get_position_distribution(driver_id, "race")
        result["race"] = self.position_calculator.calculate(
            race_dist, self.scorer.tables.driver_points[1]  # Race points table
        )

        # Calculate sprint points if applicable
        if race_format == RaceFormat.SPRINT:
            sprint_dist = self.distributions.get_position_distribution_safe(
                driver_id, "sprint"
            )
            if sprint_dist is not None:
                result["sprint"] = self.position_calculator.calculate(
                    sprint_dist,
                    self.scorer.tables.driver_points[2],  # Sprint points table
                )
            else:
                # No sprint distribution, use race as fallback
                result["sprint"] = self.position_calculator.calculate(
                    race_dist,
                    self.scorer.tables.driver_points[2],  # Sprint points table
                )

        # Calculate overtake points
        joint_qual_race = self.distributions.get_qualifying_race_distribution(driver_id)
        result["overtake"] = self.overtake_calculator.calculate(
            joint_qual_race, self.scorer.tables.overtake_multiplier
        )

        # Calculate teammate points
        joint_driver_teammate = self.distributions.get_joint_distribution_safe(
            driver_id, teammate_id, "race"
        )
        if joint_driver_teammate is not None:
            result["teammate"] = self.teammate_calculator.calculate(
                joint_driver_teammate, self.scorer.tables.teammate_thresholds
            )
        else:
            # If joint distribution not available, use 0 points
            result["teammate"] = 0.0

        # Calculate completion points
        completion_prob = self.distributions.get_completion_probability(driver_id)
        result["completion"] = self.completion_calculator.calculate(
            completion_prob,
            self.scorer.tables.completion_thresholds,
            self.scorer.tables.stage_points,
        )

        # Calculate improvement points
        result["improvement"] = self.improvement_calculator.calculate(
            race_dist, rolling_avg, self.scorer.tables.improvement_points
        )

        return result
