"""Internal calculator implementation optimized for JIT.

This module contains the core calculation functions and data structures
optimized for numba JIT compilation. Not intended for direct use.
"""

from typing import NamedTuple

import numpy as np
from numba import jit
from numpy.typing import NDArray

from gridrival_ai.scoring.types import RaceFormat


class PointTables(NamedTuple):
    """Combined lookup tables for point calculations.

    Using a NamedTuple provides clear field names while keeping
    the memory layout optimal for calculations.

    Parameters
    ----------
    base_points : NDArray[np.float64]
        Combined table for position-based points [type, position]
        type indices: 0=qualifying, 1=race, 2=sprint
    improvement_points : NDArray[np.float64]
        Points for positions gained vs average
    teammate_thresholds : NDArray[np.float64]
        Sorted [threshold, points] pairs for teammate comparison
    completion_thresholds : NDArray[np.float64]
        Sorted completion percentage thresholds
    stage_points : float
        Points per completed stage
    overtake_multiplier : float
        Points per position gained
    talent_multiplier : float
        Multiplier for talent drivers
    """

    base_points: NDArray[np.float64]
    improvement_points: NDArray[np.float64]
    teammate_thresholds: NDArray[np.float64]
    completion_thresholds: NDArray[np.float64]
    stage_points: float
    overtake_multiplier: float
    talent_multiplier: float


# Define structured dtype for race data
race_dtype = np.dtype(
    [
        ("format", "i4"),  # RaceFormat enum value
        ("qualifying", "i4"),  # Qualifying position
        ("race", "i4"),  # Race position
        ("sprint", "i4"),  # Sprint position (-1 if N/A)
        ("completion", "f8"),  # Race completion percentage
        ("rolling_avg", "f8"),  # 8-race rolling average
        ("teammate", "i4"),  # Teammate position
    ]
)


@jit(nopython=True)
def calculate_points(
    data: np.ndarray,  # Structured array with race_dtype
    tables: PointTables,
    is_talent: bool = False,
) -> np.ndarray:
    """JIT-compiled point calculation for multiple race scenarios.

    Parameters
    ----------
    data : np.ndarray
        Structured array of race data
    tables : PointTables
        Combined lookup tables
    is_talent : bool, optional
        Whether calculating for talent driver

    Returns
    -------
    np.ndarray
        Array of total points for each scenario
    """
    n_races = len(data)
    points = np.zeros(n_races)

    for i in range(n_races):
        # Base points (qualifying, race)
        points[i] = (
            tables.base_points[0, data[i]["qualifying"]]
            + tables.base_points[1, data[i]["race"]]  # Qualifying  # Race
        )

        # Sprint points if applicable
        if data[i]["format"] == RaceFormat.SPRINT.value and data[i]["sprint"] > 0:
            points[i] += tables.base_points[2, data[i]["sprint"]]

        # Overtake points
        positions_gained = max(0, data[i]["qualifying"] - data[i]["race"])
        points[i] += positions_gained * tables.overtake_multiplier

        # Improvement points
        positions_ahead = max(0, round(data[i]["rolling_avg"]) - data[i]["race"])
        if positions_ahead > 0:
            # Clip to maximum improvement points
            idx = min(positions_ahead, len(tables.improvement_points) - 1)
            points[i] += tables.improvement_points[idx]

        # Teammate points
        teammate_diff = max(0, data[i]["teammate"] - data[i]["race"])
        if teammate_diff > 0:
            # Use searchsorted for efficient threshold lookup
            idx = np.searchsorted(tables.teammate_thresholds[:, 0], teammate_diff)
            if idx == 0:
                points[i] += tables.teammate_thresholds[0, 1]
            else:
                points[i] += tables.teammate_thresholds[
                    min(idx, len(tables.teammate_thresholds) - 1), 1
                ]

        # Completion points
        completed_stages = (data[i]["completion"] >= tables.completion_thresholds).sum()
        points[i] += completed_stages * tables.stage_points

    # Apply talent multiplier if needed
    if is_talent:
        points *= tables.talent_multiplier

    return points
