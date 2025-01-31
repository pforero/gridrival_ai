"""Base classes for F1 fantasy scoring calculations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import jsonschema

from gridrival_ai.scoring.constants import (
    CONFIG_SCHEMA,
    DEFAULT_COMPLETION_STAGE_POINTS,
    DEFAULT_IMPROVEMENT_POINTS,
    DEFAULT_MINIMUM_POINTS,
    DEFAULT_OVERTAKE_MULTIPLIER,
    DEFAULT_QUALIFYING_POINTS,
    DEFAULT_RACE_POINTS,
    DEFAULT_SPRINT_POINTS,
    DEFAULT_TALENT_MULTIPLIER,
    DEFAULT_TEAMMATE_POINTS,
    MAX_MULTIPLIER,
    MAX_POINTS,
    MAX_POSITION,
    MAX_SPRINT_POSITION,
    MIN_MULTIPLIER,
    MIN_POINTS,
)
from gridrival_ai.scoring.exceptions import ConfigurationError


@dataclass
class ScoringConfig:
    """Configuration for F1 fantasy scoring rules.

    Parameters
    ----------
    qualifying_points : Dict[int, float]
        Points awarded for qualifying positions (1-20)
    race_points : Dict[int, float]
        Points awarded for race positions (1-20)
    sprint_points : Dict[int, float]
        Points awarded for sprint positions (1-8)
    completion_stage_points : float
        Points awarded per completion stage (25%, 50%, 75%, 90%)
    overtake_multiplier : float
        Points per position gained
    improvement_points : Dict[int, float]
        Points for positions gained vs 8-race average
    teammate_points : Dict[int, float]
        Points for beating teammate by position margin
    minimum_points : float
        Minimum points awarded per event (default: 650)
    talent_multiplier : float
        Multiplier for talent driver points (default: 2.0)

    Notes
    -----
    All position-based points use 1-based indexing to match F1 positions.
    Configurations can be loaded from JSON files using `from_json`.
    """

    qualifying_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_QUALIFYING_POINTS
    )
    race_points: Dict[int, float] = field(default_factory=lambda: DEFAULT_RACE_POINTS)
    sprint_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_SPRINT_POINTS
    )
    completion_stage_points: float = DEFAULT_COMPLETION_STAGE_POINTS
    overtake_multiplier: float = DEFAULT_OVERTAKE_MULTIPLIER
    improvement_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_IMPROVEMENT_POINTS
    )
    teammate_points: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_TEAMMATE_POINTS
    )
    minimum_points: float = DEFAULT_MINIMUM_POINTS
    talent_multiplier: float = DEFAULT_TALENT_MULTIPLIER

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self._validate_point_mappings()
        self._validate_multipliers()
        self._validate_point_values()

    def _validate_point_mappings(self) -> None:
        """Validate position-based point mappings."""
        # Validate qualifying points
        self._validate_positions(
            self.qualifying_points, "qualifying_points", MAX_POSITION
        )

        # Validate race points
        self._validate_positions(self.race_points, "race_points", MAX_POSITION)

        # Validate sprint points
        self._validate_positions(
            self.sprint_points, "sprint_points", MAX_SPRINT_POSITION
        )

        # Validate improvement points
        self._validate_positions(
            self.improvement_points,
            "improvement_points",
            MAX_POSITION,
            required=False,
        )

        # Validate teammate points
        self._validate_positions(
            self.teammate_points,
            "teammate_points",
            MAX_POSITION,
            required=False,
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

        if not MIN_MULTIPLIER <= self.talent_multiplier <= MAX_MULTIPLIER:
            raise ConfigurationError(
                f"talent_multiplier must be between {MIN_MULTIPLIER}"
                f" and {MAX_MULTIPLIER}"
            )

    def _validate_point_values(self) -> None:
        """Validate point values."""
        if not MIN_POINTS <= self.completion_stage_points <= MAX_POINTS:
            raise ConfigurationError(
                f"completion_stage_points must be between {MIN_POINTS}"
                f" and {MAX_POINTS}"
            )

        if not MIN_POINTS <= self.minimum_points <= MAX_POINTS:
            raise ConfigurationError(
                f"minimum_points must be between {MIN_POINTS}" f" and {MAX_POINTS}"
            )

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
            "improvement_points",
            "teammate_points",
        ]

        for pos in position_fields:
            if pos in data:
                data[pos] = {int(k): v for k, v in data[pos].items()}

        return data

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
        # Convert all numeric keys to strings for JSON compatibility
        return {
            "qualifying_points": {str(k): v for k, v in self.qualifying_points.items()},
            "race_points": {str(k): v for k, v in self.race_points.items()},
            "sprint_points": {str(k): v for k, v in self.sprint_points.items()},
            "improvement_points": {
                str(k): v for k, v in self.improvement_points.items()
            },
            "teammate_points": {str(k): v for k, v in self.teammate_points.items()},
            "completion_stage_points": self.completion_stage_points,
            "overtake_multiplier": self.overtake_multiplier,
            "minimum_points": self.minimum_points,
            "talent_multiplier": self.talent_multiplier,
        }
