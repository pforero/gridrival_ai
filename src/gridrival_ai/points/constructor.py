"""
Constructor points calculator for GridRival F1 fantasy.

This module provides the calculator for constructor fantasy points,
handling the scoring for team performance.
"""

from typing import Dict

from gridrival_ai.points.components import PositionPointsCalculator
from gridrival_ai.points.distributions import DistributionAdapter
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat


class ConstructorPointsCalculator:
    """
    Calculate expected points for constructors.

    This calculator computes the expected points for a constructor
    based on the performance of both drivers.

    Parameters
    ----------
    distributions : DistributionAdapter
        Adapter for accessing probability distributions
    scorer : ScoringCalculator
        Scoring rules calculator

    Examples
    --------
    >>> adapter = DistributionAdapter(registry)
    >>> calculator = ConstructorPointsCalculator(adapter, scorer)
    >>> points = calculator.calculate("RBR", RaceFormat.STANDARD)
    >>> print(f"Total constructor points: {sum(points.values()):.1f}")
    Total constructor points: 177.0
    """

    def __init__(self, distributions: DistributionAdapter, scorer: ScoringCalculator):
        """Initialize with distributions and scorer."""
        self.distributions = distributions
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

        # Get constructor drivers
        try:
            driver1_id, driver2_id = self.distributions.get_constructor_drivers(
                constructor_id
            )
        except KeyError:
            # If constructor not found, return zero points
            return result

        # Calculate qualifying points for both drivers
        qual_points = 0.0
        for driver_id in (driver1_id, driver2_id):
            try:
                qual_dist = self.distributions.get_position_distribution(
                    driver_id, "qualifying"
                )
                qual_points += self.position_calculator.calculate(
                    qual_dist,
                    self.scorer.tables.constructor_points[
                        0
                    ],  # Constructor qualifying points
                )
            except KeyError:
                # If distribution not found, continue with next driver
                continue

        result["qualifying"] = qual_points

        # Calculate race points for both drivers
        race_points = 0.0
        for driver_id in (driver1_id, driver2_id):
            try:
                race_dist = self.distributions.get_position_distribution(
                    driver_id, "race"
                )
                race_points += self.position_calculator.calculate(
                    race_dist,
                    self.scorer.tables.constructor_points[1],  # Constructor race points
                )
            except KeyError:
                # If distribution not found, continue with next driver
                continue

        result["race"] = race_points

        return result
