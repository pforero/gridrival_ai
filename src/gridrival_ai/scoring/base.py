"""Base classes for F1 fantasy scoring calculations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from gridrival_ai.scoring.constants import (
    CONFIG_SCHEMA,
    DEFAULT_COMPLETION_STAGE_POINTS,
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
        Points awarded for driver qualifying positions (1-20)
    race_points : Dict[int, float]
        Points awarded for driver race positions (1-20)
    sprint_points : Dict[int, float]
        Points awarded for driver sprint positions (1-8)
    constructor_qualifying_points : Dict[int, float]
        Points awarded for constructor qualifying positions (1-20)
    constructor_race_points : Dict[int, float]
        Points awarded for constructor race positions (1-20)
    completion_stage_points : float
        Points awarded per completion stage (25%, 50%, 75%, 90%)
    overtake_multiplier : float
        Points per position gained
    improvement_points : Dict[int, float]
        Points for positions gained vs 8-race average
    teammate_points : Dict[int, float]
        Points for beating teammate by position margin

    Notes
    -----
    All position-based points use 1-based indexing to match F1 positions.
    Constructor points are calculated for each driver and summed.
    Configurations can be loaded from JSON files using `from_json`.
    """

    # Driver scoring
    qualifying_points: dict = field(default_factory=lambda: DEFAULT_QUALIFYING_POINTS)
    race_points: dict = field(default_factory=lambda: DEFAULT_RACE_POINTS)
    sprint_points: dict = field(default_factory=lambda: DEFAULT_SPRINT_POINTS)

    # Constructor scoring
    constructor_qualifying_points: dict = field(
        default_factory=lambda: DEFAULT_CONSTRUCTOR_QUALIFYING_POINTS
    )
    constructor_race_points: dict = field(
        default_factory=lambda: DEFAULT_CONSTRUCTOR_RACE_POINTS
    )

    # Additional scoring components
    completion_stage_points: float = DEFAULT_COMPLETION_STAGE_POINTS
    overtake_multiplier: float = DEFAULT_OVERTAKE_MULTIPLIER
    improvement_points: dict = field(default_factory=lambda: DEFAULT_IMPROVEMENT_POINTS)
    teammate_points: dict = field(default_factory=lambda: DEFAULT_TEAMMATE_POINTS)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self._validate_driver_points()
        self._validate_constructor_points()
        self._validate_multipliers()
        self._validate_additional_points()

    def _validate_driver_points(self) -> None:
        """Validate driver position-based point mappings."""
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
        points: dict,
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
                f"completion_stage_points must be between {MIN_POINTS}"
                f" and {MAX_POINTS}"
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
    def _convert_position_keys(data: dict) -> dict:
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

    def _to_json_dict(self) -> dict:
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
            "overtake_multiplier": self.overtake_multiplier,
        }
