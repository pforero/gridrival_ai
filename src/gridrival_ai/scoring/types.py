"""Data types for F1 fantasy scoring calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from gridrival_ai.scoring.exceptions import ValidationError


def validate_position(pos: int, name: str, max_pos: int = 20) -> None:
    """Validate race position is within valid range.

    Parameters
    ----------
    pos : int
        Position to validate
    name : str
        Name for error messages
    max_pos : int, optional
        Maximum valid position, by default 20

    Raises
    ------
    ValidationError
        If position is invalid
    """
    if not isinstance(pos, int):
        raise ValidationError(f"{name} position must be an integer")
    if not 1 <= pos <= max_pos:
        raise ValidationError(
            f"{name} position must be between 1 and {max_pos}, got {pos}"
        )


def validate_percentage(value: float, name: str) -> None:
    """Validate percentage is between 0 and 1.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Name for error messages

    Raises
    ------
    ValidationError
        If value is not a valid percentage
    """
    if not isinstance(value, float):
        raise ValidationError(f"{name} must be a float")
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be between 0.0 and 1.0, got {value}")


def validate_positive_float(value: float | int, name: str) -> None:
    """Validate float is positive.

    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Name for error messages

    Raises
    ------
    ValidationError
        If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a float")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_format_consistency(format: RaceFormat, positions: DriverPositions) -> None:
    """Validate race format matches position data.

    Parameters
    ----------
    format : RaceFormat
        Race weekend format
    positions : DriverPositions
        Position data

    Raises
    ------
    ValidationError
        If format and positions are inconsistent
    """
    if format == RaceFormat.SPRINT and positions.sprint_finish is None:
        raise ValidationError("Sprint position required for sprint race format")
    if format == RaceFormat.STANDARD and positions.sprint_finish is not None:
        raise ValidationError("Sprint position not allowed for standard race format")


class RaceFormat(Enum):
    """Available F1 race weekend formats."""

    STANDARD = auto()
    SPRINT = auto()


@dataclass(frozen=True)
class DriverPointsBreakdown:
    """Detailed breakdown of driver points by component.

    Parameters
    ----------
    qualifying : float
        Points from qualifying position
    race : float
        Points from race finish position
    sprint : float, optional
        Points from sprint race, by default 0.0
    overtake : float, optional
        Points from positions gained, by default 0.0
    improvement : float, optional
        Points from beating rolling average, by default 0.0
    teammate : float, optional
        Points from beating teammate, by default 0.0
    completion : float, optional
        Points from race completion stages, by default 0.0

    Attributes
    ----------
    total : float
        Total points across all components

    Examples
    --------
    >>> # Create a breakdown
    >>> breakdown = DriverPointsBreakdown(
    ...     qualifying=50.0,
    ...     race=97.0,
    ...     overtake=6.0,
    ...     improvement=4.0,
    ...     teammate=2.0,
    ...     completion=12.0
    ... )
    >>> breakdown.qualifying
    50.0
    >>> breakdown.total  # Sum of all components
    171.0
    """

    qualifying: float
    race: float
    sprint: float = 0.0
    overtake: float = 0.0
    improvement: float = 0.0
    teammate: float = 0.0
    completion: float = 0.0

    @property
    def total(self) -> float:
        """Calculate total points across all components.

        Returns
        -------
        float
            Total points
        """
        return (
            self.qualifying
            + self.race
            + self.sprint
            + self.overtake
            + self.improvement
            + self.teammate
            + self.completion
        )


@dataclass(frozen=True)
class DriverPositions:
    """Position data for a single driver.

    Parameters
    ----------
    qualifying : int
        Qualifying position (1-20)
    race : int
        Race finish position (1-20)
    sprint_finish : int | None, optional
        Sprint race position (1-8) for sprint weekends
    """

    qualifying: int
    race: int
    sprint_finish: int | None = None

    def __init__(self, qualifying: int, race: int, sprint_finish: int | None = None):
        validate_position(qualifying, "qualifying")
        validate_position(race, "race")
        if sprint_finish is not None:
            validate_position(sprint_finish, "sprint", max_pos=8)
        object.__setattr__(self, "qualifying", qualifying)
        object.__setattr__(self, "race", race)
        object.__setattr__(self, "sprint_finish", sprint_finish)


@dataclass(frozen=True)
class DriverWeekendData:
    """Complete data for a single driver's race weekend.

    Parameters
    ----------
    format : RaceFormat
        Race weekend format (standard/sprint)
    positions : DriverPositions
        All position data for scoring
    completion_percentage : float
        Percentage of race completed (0.0 to 1.0)
    rolling_average : float
        8-race rolling average finish position (> 0)
    teammate_position : int
        Teammate's race finish position (1-20)
    """

    format: RaceFormat
    positions: DriverPositions
    completion_percentage: float
    rolling_average: float
    teammate_position: int

    def __init__(
        self,
        format: RaceFormat,
        positions: DriverPositions,
        completion_percentage: float,
        rolling_average: float,
        teammate_position: int,
    ):
        validate_percentage(completion_percentage, "completion")
        validate_positive_float(rolling_average, "rolling_average")
        validate_position(teammate_position, "teammate")
        validate_format_consistency(format, positions)

        object.__setattr__(self, "format", format)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "completion_percentage", completion_percentage)
        object.__setattr__(self, "rolling_average", float(rolling_average))
        object.__setattr__(self, "teammate_position", teammate_position)


@dataclass(frozen=True)
class ConstructorPositions:
    """Position data for both drivers in a constructor.

    Parameters
    ----------
    driver1_qualifying : int
        First driver qualifying position (1-20)
    driver1_race : int
        First driver race position (1-20)
    driver2_qualifying : int
        Second driver qualifying position (1-20)
    driver2_race : int
        Second driver race position (1-20)
    """

    driver1_qualifying: int
    driver1_race: int
    driver2_qualifying: int
    driver2_race: int

    def __init__(
        self,
        driver1_qualifying: int,
        driver1_race: int,
        driver2_qualifying: int,
        driver2_race: int,
    ):
        validate_position(driver1_qualifying, "driver1 qualifying")
        validate_position(driver1_race, "driver1 race")
        validate_position(driver2_qualifying, "driver2 qualifying")
        validate_position(driver2_race, "driver2 race")

        object.__setattr__(self, "driver1_qualifying", driver1_qualifying)
        object.__setattr__(self, "driver1_race", driver1_race)
        object.__setattr__(self, "driver2_qualifying", driver2_qualifying)
        object.__setattr__(self, "driver2_race", driver2_race)


@dataclass(frozen=True)
class ConstructorWeekendData:
    """Complete data for a constructor's race weekend.

    Parameters
    ----------
    format : RaceFormat
        Race weekend format (standard/sprint)
    positions : ConstructorPositions
        Position data for both drivers
    """

    format: RaceFormat
    positions: ConstructorPositions

    def __init__(self, format: RaceFormat, positions: ConstructorPositions):
        object.__setattr__(self, "format", format)
        object.__setattr__(self, "positions", positions)
