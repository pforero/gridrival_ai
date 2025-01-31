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


def validate_positive_float(value: float, name: str) -> None:
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
    if not isinstance(value, float):
        raise ValidationError(f"{name} must be a float")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_format_consistency(format: RaceFormat, positions: Positions) -> None:
    """Validate race format matches position data.

    Parameters
    ----------
    format : RaceFormat
        Race weekend format
    positions : Positions
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
    """Available F1 race weekend formats.

    Attributes
    ----------
    STANDARD
        Traditional format: Practice, Qualifying, Race
    SPRINT
        Sprint format: Practice, Qualifying, Sprint, Race

    Notes
    -----
    Race scoring differs between formats.
    Sprint races award fewer points than standard races.
    """

    STANDARD = auto()
    SPRINT = auto()


@dataclass(frozen=True)
class Positions:
    """Position tracking for race weekend.

    Parameters
    ----------
    qualifying : int
        Qualifying position (1-20)
    race : int
        Race finish position (1-20)
    sprint_finish : int | None, optional
        Sprint race position (1-8) for sprint weekends

    Notes
    -----
    All positions use 1-based indexing.
    sprint_finish should only be provided for sprint race weekends.
    Positions are validated using validate_position().
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
class RaceWeekendData:
    """Complete data for a single race weekend.

    Parameters
    ----------
    format : RaceFormat
        Race weekend format (standard/sprint)
    positions : Positions
        All position data for scoring
    completion_percentage : float
        Percentage of race completed (0.0 to 1.0)
    rolling_average : float
        8-race rolling average finish position (> 0)
    teammate_position : int
        Teammate's race finish position (1-20)

    Notes
    -----
    All data is validated using validators.validate_race_weekend().
    See factory methods for creating common scenarios.
    """

    format: RaceFormat
    positions: Positions
    completion_percentage: float
    rolling_average: float
    teammate_position: int

    def __init__(
        self,
        format: RaceFormat,
        positions: Positions,
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
        object.__setattr__(self, "rolling_average", rolling_average)
        object.__setattr__(self, "teammate_position", teammate_position)

    @classmethod
    def perfect_standard_race(cls, rolling_avg: float = 2.0) -> RaceWeekendData:
        """Create data for perfect standard race (P1 qualifying and race).

        Parameters
        ----------
        rolling_avg : float, optional
            8-race rolling average, by default 2.0

        Returns
        -------
        RaceWeekendData
            Perfect race scenario
        """
        return cls(
            format=RaceFormat.STANDARD,
            positions=Positions(qualifying=1, race=1),
            completion_percentage=1.0,
            rolling_average=rolling_avg,
            teammate_position=2,
        )

    @classmethod
    def perfect_sprint_weekend(cls, rolling_avg: float = 2.0) -> RaceWeekendData:
        """Create data for perfect sprint weekend (P1 in all sessions).

        Parameters
        ----------
        rolling_avg : float, optional
            8-race rolling average, by default 2.0

        Returns
        -------
        RaceWeekendData
            Perfect sprint weekend scenario
        """
        return cls(
            format=RaceFormat.SPRINT,
            positions=Positions(qualifying=1, race=1, sprint_finish=1),
            completion_percentage=1.0,
            rolling_average=rolling_avg,
            teammate_position=2,
        )

    @classmethod
    def midfield_performance(
        cls, position: int = 10, format: RaceFormat = RaceFormat.STANDARD
    ) -> RaceWeekendData:
        """Create data for consistent midfield performance.

        Parameters
        ----------
        position : int, optional
            Target position, by default 10
        format : RaceFormat, optional
            Race format, by default STANDARD

        Returns
        -------
        RaceWeekendData
            Midfield performance scenario
        """
        return cls(
            format=format,
            positions=Positions(
                qualifying=position,
                race=position,
                sprint_finish=position if format == RaceFormat.SPRINT else None,
            ),
            completion_percentage=1.0,
            rolling_average=float(position),
            teammate_position=position + 1,
        )

    @classmethod
    def dnf_scenario(cls, qualifying_pos: int = 5) -> RaceWeekendData:
        """Create data for DNF from points-scoring position.

        Parameters
        ----------
        qualifying_pos : int, optional
            Qualifying position, by default 5

        Returns
        -------
        RaceWeekendData
            DNF scenario
        """
        return cls(
            format=RaceFormat.STANDARD,
            positions=Positions(qualifying=qualifying_pos, race=20),
            completion_percentage=0.5,  # DNF halfway
            rolling_average=float(qualifying_pos),
            teammate_position=qualifying_pos + 1,
        )
