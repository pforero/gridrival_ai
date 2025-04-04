"""
Configuration classes for F1 fantasy scoring rules.

This module provides the configuration interface for managing and validating
scoring rules in the GridRival F1 fantasy system. The ScoringConfig class
handles loading, validation, and persistence of scoring parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import jsonschema
import numpy as np

from gridrival_ai.scoring.constants import (
    CONFIG_SCHEMA,
    DEFAULT_COMPLETION_STAGE_POINTS,
    DEFAULT_COMPLETION_THRESHOLDS,
    DEFAULT_CONSTRUCTOR_QUALIFYING_POINTS,
    DEFAULT_CONSTRUCTOR_RACE_POINTS,
    DEFAULT_IMPROVEMENT_POINTS,
    DEFAULT_OVERTAKE_MULTIPLIER,
    DEFAULT_QUALIFYING_POINTS,
    DEFAULT_RACE_POINTS,
    DEFAULT_SPRINT_POINTS,
    DEFAULT_TEAMMATE_POINTS,
    MAX_MULTIPLIER,
    MAX_POINTS,
    MAX_POSITION,
    MIN_MULTIPLIER,
    MIN_POINTS,
)
from gridrival_ai.scoring.exceptions import ConfigurationError


@dataclass
class ScoringConfig:
    """Configuration for F1 fantasy scoring rules.

    This class manages the scoring configuration for the GridRival F1 fantasy system,
    including position-based points, completion stages, and special bonuses.

    Parameters
    ----------
    qualifying_points : Dict[int, float], optional
        Points awarded for driver qualifying positions (1-20)
    race_points : Dict[int, float], optional
        Points awarded for driver race positions (1-20)
    sprint_points : Dict[int, float], optional
        Points awarded for driver sprint positions (1-8)
    constructor_qualifying_points : Dict[int, float], optional
        Points awarded for constructor qualifying positions (1-20)
    constructor_race_points : Dict[int, float], optional
        Points awarded for constructor race positions (1-20)
    completion_stage_points : float, optional
        Points awarded per completion stage (25%, 50%, 75%, 90%)
    completion_thresholds : list[float], optional
        Thresholds for completion stages (default: [0.25, 0.5, 0.75, 0.9])
    overtake_multiplier : float, optional
        Points per position gained
    improvement_points : Dict[int, float], optional
        Points for positions gained vs 8-race average
    teammate_points : Dict[int, float], optional
        Points for beating teammate by position margin

    Notes
    -----
    All position-based points use 1-based indexing to match F1 positions.
    Constructor points are calculated for each driver and summed.
    Configurations can be loaded from JSON files using `from_json`.

    Examples
    --------
    >>> # Create a default config
    >>> config = ScoringConfig.default()
    >>>
    >>> # Create a config with custom qualifying points
    >>> config = ScoringConfig(qualifying_points={1: 25, 2: 18, 3: 15})
    >>>
    >>> # Load from JSON file
    >>> config = ScoringConfig.from_json("config.json")
    >>>
    >>> # Save to JSON file
    >>> config.to_json("new_config.json")
    >>>
    >>> # Create modified version
    >>> modified = config.with_modifications(overtake_multiplier=2.0)
    """

    # Driver scoring
    qualifying_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_QUALIFYING_POINTS
    )
    race_points: Dict[int, float] = field(default_factory=lambda: DEFAULT_RACE_POINTS)
    sprint_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_SPRINT_POINTS
    )

    # Constructor scoring
    constructor_qualifying_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_CONSTRUCTOR_QUALIFYING_POINTS
    )
    constructor_race_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_CONSTRUCTOR_RACE_POINTS
    )

    # Additional scoring components
    completion_stage_points: float = DEFAULT_COMPLETION_STAGE_POINTS
    completion_thresholds: list[float] = field(
        default_factory=lambda: DEFAULT_COMPLETION_THRESHOLDS
    )
    overtake_multiplier: float = DEFAULT_OVERTAKE_MULTIPLIER
    improvement_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_IMPROVEMENT_POINTS
    )
    teammate_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_TEAMMATE_POINTS
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self._validate_driver_points()
        self._validate_constructor_points()
        self._validate_multipliers()
        self._validate_additional_points()
        self._validate_thresholds()

    def _validate_driver_points(self) -> None:
        """Validate driver position-based point mappings."""
        # Validate qualifying points
        self._validate_positions(
            self.qualifying_points, "qualifying_points", MAX_POSITION
        )

        # Validate race points
        self._validate_positions(self.race_points, "race_points", MAX_POSITION)

        # Validate sprint points
        self._validate_positions(self.sprint_points, "sprint_points", MAX_POSITION)

    def _validate_constructor_points(self) -> None:
        """Validate constructor position-based point mappings."""
        # Validate qualifying points
        self._validate_positions(
            self.constructor_qualifying_points,
            "constructor_qualifying_points",
            MAX_POSITION,
        )

        # Validate race points
        self._validate_positions(
            self.constructor_race_points, "constructor_race_points", MAX_POSITION
        )

    def _validate_positions(
        self,
        points: Dict[int, float],
        name: str,
        max_pos: int,
        required: bool = True,
    ) -> None:
        """Validate position-based point mappings.

        Parameters
        ----------
        points : Dict[int, float]
            Points mapping to validate
        name : str
            Name of mapping for error messages
        max_pos : int
            Maximum allowed position
        required : bool, optional
            Whether full position coverage is required, by default True

        Raises
        ------
        ConfigurationError
            If validation fails
        """
        if not points:
            raise ConfigurationError(f"Empty {name} configuration")

        invalid_pos = [pos for pos in points if not 1 <= pos <= max_pos]
        if invalid_pos:
            raise ConfigurationError(
                f"Invalid positions in {name}: {invalid_pos}"
                f" (must be between 1 and {max_pos})"
            )

        if required and len(points) != max_pos:
            raise ConfigurationError(
                f"{name} must have points for all positions 1-{max_pos}"
            )

        invalid_points = [pts for pts in points.values() if pts < MIN_POINTS]
        if invalid_points:
            raise ConfigurationError(
                f"Invalid point values in {name}: {invalid_points}"
                f" (must be >= {MIN_POINTS})"
            )

    def _validate_multipliers(self) -> None:
        """Validate multiplier values."""
        if not MIN_MULTIPLIER <= self.overtake_multiplier <= MAX_MULTIPLIER:
            raise ConfigurationError(
                f"overtake_multiplier must be between {MIN_MULTIPLIER}"
                f" and {MAX_MULTIPLIER}"
            )

    def _validate_additional_points(self) -> None:
        """Validate additional point values."""
        if not MIN_POINTS <= self.completion_stage_points <= MAX_POINTS:
            raise ConfigurationError(
                f"completion_stage_points must be between {MIN_POINTS} and {MAX_POINTS}"
            )

    def _validate_thresholds(self) -> None:
        """Validate completion thresholds."""
        if not self.completion_thresholds:
            raise ConfigurationError("completion_thresholds cannot be empty")

        # Check thresholds are ordered
        if not all(
            self.completion_thresholds[i] < self.completion_thresholds[i + 1]
            for i in range(len(self.completion_thresholds) - 1)
        ):
            raise ConfigurationError("completion_thresholds must be in ascending order")

        # Check threshold values
        if any(t <= 0.0 or t >= 1.0 for t in self.completion_thresholds):
            raise ConfigurationError(
                "completion_thresholds must be between 0.0 and 1.0 (exclusive)"
            )

    @classmethod
    def default(cls) -> ScoringConfig:
        """Create a config with default scoring values.

        Returns
        -------
        ScoringConfig
            Configuration with default values

        Examples
        --------
        >>> config = ScoringConfig.default()
        """
        return cls()

    @classmethod
    def from_json(cls, path: str | Path) -> ScoringConfig:
        """Create configuration from JSON file.

        Parameters
        ----------
        path : str | Path
            Path to JSON configuration file

        Returns
        -------
        ScoringConfig
            Loaded configuration

        Raises
        ------
        ConfigurationError
            If JSON is invalid or fails validation

        Examples
        --------
        >>> config = ScoringConfig.from_json("scoring_config.json")
        """
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}") from e

        try:
            jsonschema.validate(instance=data, schema=CONFIG_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            raise ConfigurationError(f"Invalid config format: {e}") from e

        # Convert position keys from strings to integers
        data = cls._convert_position_keys(data)

        return cls(**data)

    @staticmethod
    def _convert_position_keys(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert position keys from strings to integers.

        Parameters
        ----------
        data : Dict[str, Any]
            Configuration data with string keys

        Returns
        -------
        Dict[str, Any]
            Configuration data with integer keys for position mappings
        """
        position_fields = [
            "qualifying_points",
            "race_points",
            "sprint_points",
            "constructor_qualifying_points",
            "constructor_race_points",
            "improvement_points",
            "teammate_points",
        ]

        result = data.copy()
        for pos in position_fields:
            if pos in result:
                result[pos] = {int(k): v for k, v in result[pos].items()}

        # Handle completion thresholds if present as a list of strings
        if "completion_thresholds" in result and isinstance(
            result["completion_thresholds"], list
        ):
            result["completion_thresholds"] = [
                float(t) for t in result["completion_thresholds"]
            ]

        return result

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Parameters
        ----------
        path : str | Path
            Path to save configuration

        Raises
        ------
        ConfigurationError
            If saving fails

        Examples
        --------
        >>> config = ScoringConfig.default()
        >>> config.to_json("my_config.json")
        """
        try:
            with open(path, "w") as f:
                json.dump(self._to_json_dict(), f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save config: {e}") from e

    def _to_json_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable configuration
        """
        return {
            "qualifying_points": {str(k): v for k, v in self.qualifying_points.items()},
            "race_points": {str(k): v for k, v in self.race_points.items()},
            "sprint_points": {str(k): v for k, v in self.sprint_points.items()},
            "constructor_qualifying_points": {
                str(k): v for k, v in self.constructor_qualifying_points.items()
            },
            "constructor_race_points": {
                str(k): v for k, v in self.constructor_race_points.items()
            },
            "improvement_points": {
                str(k): v for k, v in self.improvement_points.items()
            },
            "teammate_points": {str(k): v for k, v in self.teammate_points.items()},
            "completion_stage_points": self.completion_stage_points,
            "completion_thresholds": [float(t) for t in self.completion_thresholds],
            "overtake_multiplier": self.overtake_multiplier,
        }

    def with_modifications(self, **kwargs) -> ScoringConfig:
        """Create a new config with specific modifications.

        Parameters
        ----------
        **kwargs
            Configuration parameters to modify

        Returns
        -------
        ScoringConfig
            New configuration with modifications applied

        Examples
        --------
        >>> original = ScoringConfig.default()
        >>> modified = original.with_modifications(
        ...     overtake_multiplier=2.0,
        ...     completion_stage_points=4.0
        ... )
        """
        # Create a JSON representation of the current config
        config_dict = self._to_json_dict()

        # Update with provided modifications
        for key, value in kwargs.items():
            # Handle position-based points that need string keys in JSON
            if key in [
                "qualifying_points",
                "race_points",
                "sprint_points",
                "constructor_qualifying_points",
                "constructor_race_points",
                "improvement_points",
                "teammate_points",
            ] and isinstance(value, dict):
                config_dict[key] = {str(k): v for k, v in value.items()}
            else:
                config_dict[key] = value

        # Convert back to a config, handling conversion of string keys back to integers
        converted = self._convert_position_keys(config_dict)
        return ScoringConfig(**converted)

    def to_tables(self) -> Dict[str, np.ndarray]:
        """Convert configuration to numpy arrays for efficient scoring.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of numpy arrays for scoring calculations

        Notes
        -----
        This method converts the configuration to a format optimized for
        the scoring engine, creating numpy arrays for fast lookups.
        """
        tables = {}

        # Create driver points table [type, position]
        driver_points = np.zeros((3, MAX_POSITION + 1))  # qual, race, sprint

        # Fill qualifying points (type index 0)
        for pos, points in self.qualifying_points.items():
            driver_points[0, pos] = points

        # Fill race points (type index 1)
        for pos, points in self.race_points.items():
            driver_points[1, pos] = points

        # Fill sprint points (type index 2)
        for pos, points in self.sprint_points.items():
            driver_points[2, pos] = points

        tables["driver_points"] = driver_points

        # Create constructor points table [type, position]
        constructor_points = np.zeros((2, MAX_POSITION + 1))  # qual, race only

        # Fill constructor qualifying points (type index 0)
        for pos, points in self.constructor_qualifying_points.items():
            constructor_points[0, pos] = points

        # Fill constructor race points (type index 1)
        for pos, points in self.constructor_race_points.items():
            constructor_points[1, pos] = points

        tables["constructor_points"] = constructor_points

        # Create improvement points array
        max_improve = max(self.improvement_points.keys())
        improvement_points = np.zeros(max_improve + 1)
        for pos, points in self.improvement_points.items():
            improvement_points[pos] = points

        tables["improvement_points"] = improvement_points

        # Create sorted teammate thresholds
        teammate_thresholds = np.array(
            sorted(self.teammate_points.items(), key=lambda x: x[0])
        )
        tables["teammate_thresholds"] = teammate_thresholds

        # Completion thresholds
        tables["completion_thresholds"] = np.array(self.completion_thresholds)

        # Scalar values
        tables["stage_points"] = self.completion_stage_points
        tables["overtake_multiplier"] = self.overtake_multiplier

        return tables
