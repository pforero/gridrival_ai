"""F1 fantasy scoring calculator optimized for Monte Carlo simulations.

This module provides the Scorer class for calculating both driver and constructor
points in F1 fantasy. It's optimized for Monte Carlo simulations with batch
processing capabilities.

Performance Characteristics
-------------------------
- First calculation includes JIT compilation (~100ms)
- Subsequent calculations ~1µs per race
- Batch processing up to 50x faster than individual calculations
- Thread-safe after JIT compilation
- Uses ~1KB memory for lookup tables
"""

import numpy as np

from gridrival_ai.scoring._calculator import (
    PointTables,
    calculate_constructor_points,
    calculate_driver_points,
    constructor_dtype,
    race_dtype,
)
from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.types import ConstructorWeekendData, DriverWeekendData


class Scorer:
    """High performance scorer for F1 fantasy points.

    This class provides an efficient implementation for calculating F1 fantasy
    points for both drivers and constructors. It's optimized for Monte Carlo
    simulations using JIT compilation and vectorized operations.

    Parameters
    ----------
    config : ScoringConfig
        Scoring configuration with point values

    Examples
    --------
    >>> # Calculate driver points
    >>> scorer = Scorer(config)
    >>> driver_data = DriverWeekendData(...)
    >>> points = scorer.calculate_driver(driver_data)
    >>> print(points)
    162.5

    >>> # Calculate constructor points
    >>> constructor_data = ConstructorWeekendData(...)
    >>> points = scorer.calculate_constructor(constructor_data)
    >>> print(points)
    245.0
    """

    def __init__(self, config: ScoringConfig):
        """Initialize scorer with pre-computed lookup tables."""
        # Create driver points table [type, position]
        driver_points = np.zeros((3, 21))  # qual, race, sprint

        # Fill qualifying points (type index 0)
        for pos, points in config.qualifying_points.items():
            driver_points[0, pos] = points

        # Fill race points (type index 1)
        for pos, points in config.race_points.items():
            driver_points[1, pos] = points

        # Fill sprint points (type index 2)
        for pos, points in config.sprint_points.items():
            driver_points[2, pos] = points

        # Create constructor points table [type, position]
        constructor_points = np.zeros((2, 21))  # qual, race only

        # Fill constructor qualifying points (type index 0)
        for pos, points in config.constructor_qualifying_points.items():
            constructor_points[0, pos] = points

        # Fill constructor race points (type index 1)
        for pos, points in config.constructor_race_points.items():
            constructor_points[1, pos] = points

        # Create improvement points array
        max_improve = max(config.improvement_points.keys())
        improvement_points = np.zeros(max_improve + 1)
        for pos, points in config.improvement_points.items():
            improvement_points[pos] = points

        # Create sorted teammate thresholds
        teammate_thresholds = np.array(
            sorted(config.teammate_points.items(), key=lambda x: x[0])
        )

        # Completion thresholds
        completion_thresholds = np.array([0.25, 0.50, 0.75, 0.90])

        # Create combined tables
        self.tables = PointTables(
            driver_points=driver_points,
            constructor_points=constructor_points,
            improvement_points=improvement_points,
            teammate_thresholds=teammate_thresholds,
            completion_thresholds=completion_thresholds,
            stage_points=config.completion_stage_points,
            overtake_multiplier=config.overtake_multiplier,
        )

    def convert_to_driver_array(self, data: DriverWeekendData) -> np.ndarray:
        """Convert DriverWeekendData to structured array format.

        Parameters
        ----------
        data : DriverWeekendData
            Driver's race weekend data

        Returns
        -------
        np.ndarray
            Structured array with race_dtype
        """
        return np.array(
            [
                (
                    data.format.value,
                    data.positions.qualifying,
                    data.positions.race,
                    data.positions.sprint_finish or -1,
                    data.completion_percentage,
                    data.rolling_average,
                    data.teammate_position,
                )
            ],
            dtype=race_dtype,
        )

    def convert_to_constructor_array(self, data: ConstructorWeekendData) -> np.ndarray:
        """Convert ConstructorWeekendData to structured array format.

        Parameters
        ----------
        data : ConstructorWeekendData
            Constructor's race weekend data

        Returns
        -------
        np.ndarray
            Structured array with constructor_dtype
        """
        return np.array(
            [
                (
                    data.format.value,
                    data.positions.driver1_qualifying,
                    data.positions.driver2_qualifying,
                    data.positions.driver1_race,
                    data.positions.driver2_race,
                )
            ],
            dtype=constructor_dtype,
        )

    def calculate_driver_batch(self, data: np.ndarray) -> np.ndarray:
        """Calculate driver points for multiple scenarios.

        Parameters
        ----------
        data : np.ndarray
            Structured array with race_dtype

        Returns
        -------
        np.ndarray
            Array of calculated points
        """
        return calculate_driver_points(data, self.tables)

    def calculate_driver(self, data: DriverWeekendData) -> float:
        """Calculate points for a single driver scenario.

        Parameters
        ----------
        data : DriverWeekendData
            Driver's race weekend data

        Returns
        -------
        float
            Calculated points
        """
        arr_data = self.convert_to_driver_array(data)
        return float(self.calculate_driver_batch(arr_data)[0])

    def calculate_constructor_batch(self, data: np.ndarray) -> np.ndarray:
        """Calculate constructor points for multiple scenarios.

        Parameters
        ----------
        data : np.ndarray
            Structured array with constructor_dtype

        Returns
        -------
        np.ndarray
            Array of calculated points
        """
        return calculate_constructor_points(data, self.tables)

    def calculate_constructor(self, data: ConstructorWeekendData) -> float:
        """Calculate points for a single constructor scenario.

        Parameters
        ----------
        data : ConstructorWeekendData
            Constructor's race weekend data

        Returns
        -------
        float
            Calculated points
        """
        arr_data = self.convert_to_constructor_array(data)
        return float(self.calculate_constructor_batch(arr_data)[0])
