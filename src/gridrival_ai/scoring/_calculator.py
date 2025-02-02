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

    Parameters
    ----------
    driver_points : NDArray[np.float64]
        Table for driver position-based points [type, position]
        type indices: 0=qualifying, 1=race, 2=sprint
    constructor_points : NDArray[np.float64]
        Table for constructor position-based points [type, position]
        type indices: 0=qualifying, 1=race
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
    """

    driver_points: NDArray[np.float64]
    constructor_points: NDArray[np.float64]
    improvement_points: NDArray[np.float64]
    teammate_thresholds: NDArray[np.float64]
    completion_thresholds: NDArray[np.float64]
    stage_points: float
    overtake_multiplier: float


# Define structured dtype for race data (unchanged)
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

# New dtype for constructor data
constructor_dtype = np.dtype(
    [
        ("format", "i4"),  # RaceFormat enum value
        ("qualifying1", "i4"),  # First driver qualifying
        ("qualifying2", "i4"),  # Second driver qualifying
        ("race1", "i4"),  # First driver race
        ("race2", "i4"),  # Second driver race
    ]
)


@jit(nopython=True)
def calculate_driver_points(
    data: np.ndarray,  # Structured array with race_dtype
    tables: PointTables,
) -> np.ndarray:
    """JIT-compiled point calculation for multiple race scenarios.

    Parameters
    ----------
    data : np.ndarray
        Structured array of race data
    tables : PointTables
        Combined lookup tables

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
            tables.driver_points[0, data[i]["qualifying"]]
            + tables.driver_points[1, data[i]["race"]]  # Qualifying  # Race
        )

        # Sprint points if applicable
        if data[i]["format"] == RaceFormat.SPRINT.value and data[i]["sprint"] > 0:
            points[i] += tables.driver_points[2, data[i]["sprint"]]

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

    return points


@jit(nopython=True)
def calculate_constructor_points(
    data: np.ndarray,  # Structured array with constructor_dtype
    tables: PointTables,
) -> np.ndarray:
    """JIT-compiled constructor point calculation for multiple scenarios.

    Parameters
    ----------
    data : np.ndarray
        Structured array of constructor data
    tables : PointTables
        Combined lookup tables

    Returns
    -------
    np.ndarray
        Array of total constructor points for each scenario
    """
    n_scenarios = len(data)
    points = np.zeros(n_scenarios)

    for i in range(n_scenarios):
        # Qualifying points (sum of both drivers)
        points[i] = (
            tables.constructor_points[0, data[i]["qualifying1"]]
            + tables.constructor_points[0, data[i]["qualifying2"]]
        )

        # Race points (sum of both drivers)
        points[i] += (
            tables.constructor_points[1, data[i]["race1"]]
            + tables.constructor_points[1, data[i]["race2"]]
        )

    return points
