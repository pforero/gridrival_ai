"""
Optimized calculation engine for F1 fantasy points.

This module provides the low-level implementation for calculating F1 fantasy points
using optimized data structures and algorithms. It's designed for batch processing
and high-performance calculations.

Note: This is an internal implementation module. Most users should use the
higher-level ScoringCalculator interface from calculator.py.
"""

from __future__ import annotations

import numpy as np

from gridrival_ai.scoring.config import ScoringConfig
from gridrival_ai.scoring.types import (
    ConstructorWeekendData,
    DriverWeekendData,
    RaceFormat,
)


class ScoringEngine:
    """
    Optimized calculation engine for F1 fantasy points.

    This is an internal class that handles the optimized calculations.
    Users should typically use ScoringCalculator instead.

    Parameters
    ----------
    config : ScoringConfig
        Scoring configuration

    Attributes
    ----------
    tables : dict
        Optimized lookup tables for scoring calculations
    """

    def __init__(self, config: ScoringConfig):
        """Initialize engine with scoring configuration."""
        self.config = config
        self.tables = self._create_tables()

    def _create_tables(self) -> dict:
        """Convert configuration to optimized lookup tables.

        Returns
        -------
        dict
            Dictionary of numpy arrays for fast lookups
        """
        tables = {}

        # Driver points tables for qualifying, race, sprint
        driver_points = np.zeros((3, 21))  # [type, position]

        # Fill qualifying points (index 0)
        for pos, points in self.config.qualifying_points.items():
            driver_points[0, pos] = points

        # Fill race points (index 1)
        for pos, points in self.config.race_points.items():
            driver_points[1, pos] = points

        # Fill sprint points (index 2)
        for pos, points in self.config.sprint_points.items():
            driver_points[2, pos] = points

        tables["driver_points"] = driver_points

        # Constructor points tables for qualifying, race
        constructor_points = np.zeros((2, 21))  # [type, position]

        # Fill constructor qualifying points (index 0)
        for pos, points in self.config.constructor_qualifying_points.items():
            constructor_points[0, pos] = points

        # Fill constructor race points (index 1)
        for pos, points in self.config.constructor_race_points.items():
            constructor_points[1, pos] = points

        tables["constructor_points"] = constructor_points

        # Improvement points array
        max_improve = max(self.config.improvement_points.keys())
        improvement_points = np.zeros(max_improve + 1)
        for pos, points in self.config.improvement_points.items():
            improvement_points[pos] = points

        tables["improvement_points"] = improvement_points

        # Convert teammate thresholds to sorted array of [threshold, points] pairs
        teammate_thresholds = np.array(
            sorted(self.config.teammate_points.items(), key=lambda x: x[0])
        )
        tables["teammate_thresholds"] = teammate_thresholds

        # Completion thresholds and points
        tables["completion_thresholds"] = np.array(
            sorted(self.config.completion_thresholds)
        )
        tables["stage_points"] = self.config.completion_stage_points
        tables["overtake_multiplier"] = self.config.overtake_multiplier

        return tables

    def calculate_driver(self, data: DriverWeekendData) -> dict:
        """
        Calculate points for a single driver.

        Parameters
        ----------
        data : DriverWeekendData
            Driver's race weekend data

        Returns
        -------
        dict
            Points breakdown by component
        """
        result = {}

        # Base points (qualifying, race)
        result["qualifying"] = self.tables["driver_points"][
            0, data.positions.qualifying
        ]
        result["race"] = self.tables["driver_points"][1, data.positions.race]

        # Sprint points if applicable
        if (
            data.format == RaceFormat.SPRINT
            and data.positions.sprint_finish is not None
        ):
            result["sprint"] = self.tables["driver_points"][
                2, data.positions.sprint_finish
            ]
        else:
            result["sprint"] = 0.0

        # Overtake points
        positions_gained = max(0, data.positions.qualifying - data.positions.race)
        result["overtake"] = positions_gained * self.tables["overtake_multiplier"]

        # Improvement points
        positions_ahead = max(0, round(data.rolling_average) - data.positions.race)
        if positions_ahead > 0:
            # Clip to maximum improvement points
            idx = min(positions_ahead, len(self.tables["improvement_points"]) - 1)
            result["improvement"] = self.tables["improvement_points"][idx]
        else:
            result["improvement"] = 0.0

        # Teammate points
        teammate_diff = max(0, data.teammate_position - data.positions.race)
        if teammate_diff > 0:
            # Find applicable threshold using searchsorted with exclusive=True
            thresholds = self.tables["teammate_thresholds"]

            if len(thresholds) > 0:
                # Find the largest threshold below or equal to our value
                idx = 0
                for i, threshold in enumerate(thresholds):
                    if teammate_diff > threshold[0] and i < len(thresholds) - 1:
                        continue
                    elif teammate_diff > threshold[0]:
                        idx = i
                        break
                    else:
                        idx = i
                        break

                result["teammate"] = thresholds[idx, 1]
            else:
                result["teammate"] = 0.0
        else:
            result["teammate"] = 0.0

        # Completion points
        completed_stages = np.sum(
            data.completion_percentage >= self.tables["completion_thresholds"]
        )
        result["completion"] = completed_stages * self.tables["stage_points"]

        return result

    def calculate_constructor(self, data: ConstructorWeekendData) -> dict:
        """
        Calculate points for a constructor.

        Parameters
        ----------
        data : ConstructorWeekendData
            Constructor's race weekend data

        Returns
        -------
        dict
            Points breakdown by component
        """
        result = {
            "qualifying": 0.0,
            "race": 0.0,
        }

        # Qualifying points (sum of both drivers)
        result["qualifying"] += self.tables["constructor_points"][
            0, data.positions.driver1_qualifying
        ]
        result["qualifying"] += self.tables["constructor_points"][
            0, data.positions.driver2_qualifying
        ]

        # Race points (sum of both drivers)
        result["race"] += self.tables["constructor_points"][
            1, data.positions.driver1_race
        ]
        result["race"] += self.tables["constructor_points"][
            1, data.positions.driver2_race
        ]

        return result

    def calculate_driver_batch(self, data_array: np.ndarray) -> np.ndarray:
        """
        Calculate points for multiple driver scenarios efficiently.

        Parameters
        ----------
        data_array : np.ndarray
            Structured array of driver scenarios

        Returns
        -------
        np.ndarray
            Array of total points for each scenario
        """
        n_scenarios = len(data_array)
        points = np.zeros(n_scenarios)

        for i in range(n_scenarios):
            # Get data for this scenario
            format_val = data_array[i]["format"]
            qual_pos = data_array[i]["qualifying"]
            race_pos = data_array[i]["race"]
            sprint_pos = data_array[i]["sprint"]
            completion = data_array[i]["completion"]
            rolling_avg = data_array[i]["rolling_avg"]
            teammate_pos = data_array[i]["teammate"]

            # Base points (qualifying, race)
            points[i] = (
                self.tables["driver_points"][0, qual_pos]  # Qualifying
                + self.tables["driver_points"][1, race_pos]  # Race
            )

            # Sprint points if applicable
            if format_val == RaceFormat.SPRINT.value and sprint_pos > 0:
                points[i] += self.tables["driver_points"][2, sprint_pos]

            # Overtake points
            positions_gained = max(0, qual_pos - race_pos)
            points[i] += positions_gained * self.tables["overtake_multiplier"]

            # Improvement points
            positions_ahead = max(0, round(rolling_avg) - race_pos)
            if positions_ahead > 0:
                # Clip to maximum improvement points
                idx = min(positions_ahead, len(self.tables["improvement_points"]) - 1)
                points[i] += self.tables["improvement_points"][idx]

            # Teammate points
            teammate_diff = max(0, teammate_pos - race_pos)
            if teammate_diff > 0:
                # Find applicable threshold using searchsorted
                idx = np.searchsorted(
                    self.tables["teammate_thresholds"][:, 0],
                    teammate_diff,
                    side="right",
                )
                if idx > 0:
                    # Use the threshold that's <= our value
                    idx = idx - 1
                    points[i] += self.tables["teammate_thresholds"][idx, 1]
                else:
                    # Driver is ahead of teammate but below first threshold
                    # Still award points for the first threshold
                    points[i] += self.tables["teammate_thresholds"][0, 1]

            # Completion points
            completed_stages = np.sum(
                completion >= self.tables["completion_thresholds"]
            )
            points[i] += completed_stages * self.tables["stage_points"]

        return points

    def calculate_constructor_batch(self, data_array: np.ndarray) -> np.ndarray:
        """
        Calculate points for multiple constructor scenarios efficiently.

        Parameters
        ----------
        data_array : np.ndarray
            Structured array of constructor scenarios

        Returns
        -------
        np.ndarray
            Array of total points for each scenario
        """
        n_scenarios = len(data_array)
        points = np.zeros(n_scenarios)

        for i in range(n_scenarios):
            # Get data for this scenario
            qual1 = data_array[i]["qualifying1"]
            qual2 = data_array[i]["qualifying2"]
            race1 = data_array[i]["race1"]
            race2 = data_array[i]["race2"]

            # Qualifying points (sum of both drivers)
            points[i] += (
                self.tables["constructor_points"][0, qual1]
                + self.tables["constructor_points"][0, qual2]
            )

            # Race points (sum of both drivers)
            points[i] += (
                self.tables["constructor_points"][1, race1]
                + self.tables["constructor_points"][1, race2]
            )

        return points
