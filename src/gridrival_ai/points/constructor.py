"""
Constructor points calculator for GridRival F1 fantasy.

This module provides the calculator for constructor fantasy points,
handling the scoring for team performance.
"""

from typing import Dict

from gridrival_ai.data.reference import CONSTRUCTORS
from gridrival_ai.points.components import PositionPointsCalculator
from gridrival_ai.probabilities.distributions import RaceDistribution
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat


class ConstructorPointsCalculator:
    """
    Calculate expected points for constructors.

    This calculator computes the expected points for a constructor
    based on the performance of both drivers.

    Parameters
    ----------
    race_distribution : RaceDistribution
        Race distribution containing probabilities for all sessions
    scorer : ScoringCalculator
        Scoring rules calculator

    Examples
    --------
    >>> race_dist = RaceDistribution(race_session)
    >>> calculator = ConstructorPointsCalculator(race_dist, scorer)
    >>> points = calculator.calculate("RBR", RaceFormat.STANDARD)
    >>> print(f"Total constructor points: {sum(points.values()):.1f}")
    Total constructor points: 177.0
    """

    def __init__(self, race_distribution: RaceDistribution, scorer: ScoringCalculator):
        """Initialize with race distribution and scorer."""
        self.race_distribution = race_distribution
        self.scorer = scorer
        self.position_calculator = PositionPointsCalculator()

    def calculate(
        self, constructor_id: str, race_format: RaceFormat = RaceFormat.STANDARD
    ) -> Dict[str, float]:
        """
        Calculate expected points breakdown for a constructor.

        Parameters
        ----------
        constructor_id : str
            Constructor ID
        race_format : RaceFormat, optional
            Type of race weekend, by default RaceFormat.STANDARD

        Returns
        -------
        Dict[str, float]
            Points breakdown by component

        Notes
        -----
        Constructor points are calculated for each driver and summed.
        The race format does not affect constructor points directly.

        Raises
        ------
        KeyError
            If constructor or required distributions not found
        """
        result = {"qualifying": 0.0, "race": 0.0}

        driver1_id, driver2_id = CONSTRUCTORS.get(constructor_id).drivers

        # Calculate qualifying points for both drivers
        qual_points = 0.0
        for driver_id in (driver1_id, driver2_id):
            try:
                qual_dist = self.race_distribution.get_driver_distribution(
                    driver_id, "qualifying"
                )
                qual_points += self.position_calculator.calculate(
                    qual_dist,
                    self.scorer.tables.constructor_points[
                        0
                    ],  # Constructor qualifying points
                )
            except (KeyError, ValueError):
                # If distribution not found, continue with next driver
                continue

        result["qualifying"] = qual_points

        # Calculate race points for both drivers
        race_points = 0.0
        for driver_id in (driver1_id, driver2_id):
            try:
                race_dist = self.race_distribution.get_driver_distribution(
                    driver_id, "race"
                )
                race_points += self.position_calculator.calculate(
                    race_dist,
                    self.scorer.tables.constructor_points[1],  # Constructor race points
                )
            except (KeyError, ValueError):
                # If distribution not found, continue with next driver
                continue

        result["race"] = race_points

        return result
