"""F1 fantasy scoring calculator optimized for Monte Carlo simulations.

Performance Characteristics
-------------------------
- First calculation includes JIT compilation (~100ms)
- Subsequent calculations ~1µs per race
- Batch processing up to 50x faster than individual
- Thread-safe after JIT compilation
- Uses ~1KB memory for lookup tables

Notes
-----
For optimal performance:
- Use batch calculations when possible
- Reuse calculator instance
- Run warmup calculation before timing
"""

import numpy as np

from gridrival_ai.scoring._calculator import PointTables, calculate_points, race_dtype
from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.types import RaceWeekendData


class Scorer:
    """High performance scorer for F1 fantasy points.

    This class provides an efficient implementation for calculating F1 fantasy
    points, optimized for Monte Carlo simulations. It uses JIT compilation,
    structured arrays, and vectorized operations for maximum performance.

    Parameters
    ----------
    config : ScoringConfig
        Scoring configuration with point values

    Examples
    --------
    >>> scorer = Scorer(config)
    >>> data = RaceWeekendData(...)
    >>> points = scorer.calculate(data)
    >>> points
    162.0  # Example total points

    For batch calculations:
    >>> data_batch = np.array([...], dtype=race_dtype)
    >>> points = scorer.calculate_batch(data_batch)
    >>> points
    array([162.0, 145.5, ...])
    """

    def __init__(self, config: ScoringConfig):
        """Initialize scorer with pre-computed lookup tables."""
        # Create combined position points table [type, position]
        base_points = np.zeros((3, 21))  # qual, race, sprint

        # Fill qualifying points (type index 0)
        for pos, points in config.qualifying_points.items():
            base_points[0, pos] = points

        # Fill race points (type index 1)
        for pos, points in config.race_points.items():
            base_points[1, pos] = points

        # Fill sprint points (type index 2)
        for pos, points in config.sprint_points.items():
            base_points[2, pos] = points

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
            base_points=base_points,
            improvement_points=improvement_points,
            teammate_thresholds=teammate_thresholds,
            completion_thresholds=completion_thresholds,
            stage_points=config.completion_stage_points,
            overtake_multiplier=config.overtake_multiplier,
            talent_multiplier=config.talent_multiplier,
        )

    def convert_to_array(self, data: RaceWeekendData) -> np.ndarray:
        """Convert RaceWeekendData to structured array format.

        Parameters
        ----------
        data : RaceWeekendData
            Race weekend data

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

    def calculate_batch(self, data: np.ndarray, is_talent: bool = False) -> np.ndarray:
        """Calculate points for multiple race scenarios.

        Parameters
        ----------
        data : np.ndarray
            Structured array with race_dtype
        is_talent : bool, optional
            Whether calculating for talent driver

        Returns
        -------
        np.ndarray
            Array of calculated points
        """
        return calculate_points(data, self.tables, is_talent)

    def calculate(self, data: RaceWeekendData, is_talent: bool = False) -> float:
        """Calculate points for a single race scenario.

        Parameters
        ----------
        data : RaceWeekendData
            Race weekend data
        is_talent : bool, optional
            Whether calculating for talent driver

        Returns
        -------
        float
            Calculated points
        """
        arr_data = self.convert_to_array(data)
        return float(self.calculate_batch(arr_data, is_talent)[0])
