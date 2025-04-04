"""
Main calculator interface for GridRival F1 fantasy points.

This module provides the primary interface for calculating expected fantasy points
based on probability distributions. It orchestrates the driver and constructor
calculators and handles the integration with team data.
"""

from types import SimpleNamespace
from typing import Dict

from gridrival_ai.data.reference import get_teammate
from gridrival_ai.points.constructor import ConstructorPointsCalculator
from gridrival_ai.points.driver import DriverPointsCalculator
from gridrival_ai.probabilities.distributions import RaceDistribution
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat


class PointsCalculator:
    """
    Main calculator interface for expected points calculations.

    This class orchestrates the calculation of expected points for drivers
    and constructors using component calculators and probability distributions.

    Parameters
    ----------
    scorer : ScoringCalculator
        Scoring rules calculator
    race_distribution : RaceDistribution
        Distribution containing probabilities for all sessions and drivers
    driver_stats : Dict[str, float]
        Dictionary of driver rolling averages

    Examples
    --------
    >>> from gridrival_ai.probabilities.distributions import RaceDistribution
    >>> from gridrival_ai.scoring.calculator import ScoringCalculator
    >>>
    >>> # Create distributions from betting odds
    >>> odds = {...}  # Dictionary of betting odds
    >>> race_dist = RaceDistribution.from_structured_odds(odds)
    >>>
    >>> scorer = ScoringCalculator(config)
    >>> driver_stats = {"VER": 1.5, "PER": 3.2}
    >>>
    >>> calculator = PointsCalculator(scorer, race_dist, driver_stats)
    >>> ver_points = calculator.calculate_driver_points("VER")
    >>> rbr_points = calculator.calculate_constructor_points("RBR")
    """

    def __init__(
        self,
        scorer: ScoringCalculator,
        race_distribution: RaceDistribution,
        driver_stats: Dict[str, float],
    ):
        """Initialize with scoring rules and probability distributions."""
        self.scorer = scorer
        self.race_distribution = race_distribution
        self.driver_stats = driver_stats

        # Initialize component calculators
        self.driver_calculator = DriverPointsCalculator(self.race_distribution, scorer)

        self.constructor_calculator = ConstructorPointsCalculator(
            self.race_distribution, scorer
        )

    def calculate_driver_points(
        self, driver_id: str, race_format: RaceFormat = RaceFormat.STANDARD
    ) -> Dict[str, float]:
        """
        Calculate expected points breakdown for a driver.

        Parameters
        ----------
        driver_id : str
            Driver ID (e.g., "VER")
        race_format : RaceFormat, optional
            Type of race weekend (standard/sprint), by default RaceFormat.STANDARD

        Returns
        -------
        Dict[str, float]
            Points breakdown by component (qualifying, race, etc.)
        """
        # Get teammate ID and rolling average
        teammate_id = get_teammate(driver_id)
        rolling_avg = self.driver_stats[driver_id]

        # Calculate points
        return self.driver_calculator.calculate(
            driver_id=driver_id,
            rolling_avg=rolling_avg,
            teammate_id=teammate_id,
            race_format=race_format,
        )

    def calculate_constructor_points(
        self, constructor_id: str, race_format: RaceFormat = RaceFormat.STANDARD
    ) -> Dict[str, float]:
        """
        Calculate expected points breakdown for a constructor.

        Parameters
        ----------
        constructor_id : str
            Constructor ID (e.g., "RBR")
        race_format : RaceFormat, optional
            Type of race weekend, by default RaceFormat.STANDARD

        Returns
        -------
        Dict[str, float]
            Points breakdown by component

        Raises
        ------
        KeyError
            If constructor or required distributions not found
        """
        return self.constructor_calculator.calculate(
            constructor_id=constructor_id,
            race_format=race_format,
        )

    @property
    def tables(self):
        """
        Get scoring tables with attribute-style access.

        Returns
        -------
        SimpleNamespace
            Object for accessing tables via attributes
        """
        return SimpleNamespace(**self.scorer.tables)
