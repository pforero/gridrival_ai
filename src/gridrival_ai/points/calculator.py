"""
Main calculator interface for GridRival F1 fantasy points.

This module provides the primary interface for calculating expected fantasy points
based on probability distributions. It orchestrates the driver and constructor
calculators and handles the integration with team data.
"""

from typing import Dict

from gridrival_ai.data.reference import CONSTRUCTORS
from gridrival_ai.points.constructor import ConstructorPointsCalculator
from gridrival_ai.points.distributions import DistributionAdapter
from gridrival_ai.points.driver import DriverPointsCalculator
from gridrival_ai.probabilities.registry import DistributionRegistry
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import RaceFormat


class PointsCalculator:
    """
    Main calculator interface for expected points calculations.

    This class orchestrates the calculation of expected points for drivers
    and constructors using component calculators and probability distributions.

    Parameters
    ----------
    scorer : Scorer
        Scoring rules calculator
    probability_registry : DistributionRegistry
        Registry containing probability distributions
    driver_stats : Dict[str, float]
        Dictionary of driver rolling averages

    Examples
    --------
    >>> from gridrival_ai.probabilities.registry import DistributionRegistry
    >>> from gridrival_ai.scoring.calculator import Scorer
    >>>
    >>> registry = DistributionRegistry()
    >>> scorer = Scorer(config)
    >>> driver_stats = {"VER": 1.5, "PER": 3.2}
    >>>
    >>> calculator = PointsCalculator(scorer, registry, driver_stats)
    >>> ver_points = calculator.calculate_driver_points("VER")
    >>> rbr_points = calculator.calculate_constructor_points("RBR")
    """

    def __init__(
        self,
        scorer: Scorer,
        probability_registry: DistributionRegistry,
        driver_stats: Dict[str, float],
    ):
        """Initialize with scoring rules and probability distributions."""
        self.scorer = scorer
        self.distribution_adapter = DistributionAdapter(probability_registry)
        self.driver_stats = driver_stats

        # Initialize component calculators
        self.driver_calculator = DriverPointsCalculator(
            self.distribution_adapter, scorer
        )

        self.constructor_calculator = ConstructorPointsCalculator(
            self.distribution_adapter, scorer
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

        Raises
        ------
        KeyError
            If driver or required distributions not found
        """
        # Get teammate ID
        teammate_id = None
        for constructor in CONSTRUCTORS.values():
            if driver_id in constructor.drivers:
                teammate_id = (
                    constructor.drivers[0]
                    if constructor.drivers[1] == driver_id
                    else constructor.drivers[1]
                )
                break

        if teammate_id is None:
            raise KeyError(f"No teammate found for driver {driver_id}")

        # Get rolling average
        rolling_avg = self.driver_stats.get(
            driver_id, 10.0
        )  # Default to mid-field if not found

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
            constructor_id=constructor_id, race_format=race_format
        )
