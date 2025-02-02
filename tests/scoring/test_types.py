"""
Unit tests for F1 fantasy scoring data types.

Tests the core data structures used for calculating F1 fantasy points:
- Race formats
- Driver and constructor position tracking
- Race completion stages
- Race weekend data
"""

import pytest

from gridrival_ai.scoring.exceptions import ValidationError
from gridrival_ai.scoring.types import (
    ConstructorPositions,
    ConstructorWeekendData,
    DriverPositions,
    DriverWeekendData,
    RaceFormat,
    validate_format_consistency,
    validate_percentage,
    validate_position,
    validate_positive_float,
)


def test_race_format_values():
    """Test RaceFormat enum has expected values."""
    assert len(RaceFormat) == 2
    assert RaceFormat.STANDARD.name == "STANDARD"
    assert RaceFormat.SPRINT.name == "SPRINT"


def test_driver_positions_creation():
    """Test DriverPositions dataclass creation and attributes."""
    # Standard race weekend positions
    pos = DriverPositions(qualifying=3, race=2)
    assert pos.qualifying == 3
    assert pos.race == 2
    assert pos.sprint_finish is None

    # Sprint race weekend positions
    sprint_pos = DriverPositions(qualifying=4, race=3, sprint_finish=5)
    assert sprint_pos.qualifying == 4
    assert sprint_pos.race == 3
    assert sprint_pos.sprint_finish == 5


def test_driver_positions_immutability():
    """Test that DriverPositions instances are immutable."""
    pos = DriverPositions(qualifying=1, race=1)
    with pytest.raises(AttributeError):
        pos.qualifying = 2


def test_constructor_positions_creation():
    """Test ConstructorPositions dataclass creation and attributes."""
    pos = ConstructorPositions(
        driver1_qualifying=3,
        driver1_race=2,
        driver2_qualifying=5,
        driver2_race=4,
    )
    assert pos.driver1_qualifying == 3
    assert pos.driver1_race == 2
    assert pos.driver2_qualifying == 5
    assert pos.driver2_race == 4


def test_constructor_positions_immutability():
    """Test that ConstructorPositions instances are immutable."""
    pos = ConstructorPositions(
        driver1_qualifying=1,
        driver1_race=1,
        driver2_qualifying=2,
        driver2_race=2,
    )
    with pytest.raises(AttributeError):
        pos.driver1_qualifying = 3


def test_driver_weekend_data_creation():
    """Test DriverWeekendData dataclass creation and attributes."""
    positions = DriverPositions(qualifying=3, race=2)
    weekend = DriverWeekendData(
        format=RaceFormat.STANDARD,
        positions=positions,
        completion_percentage=1.0,
        rolling_average=3.5,
        teammate_position=4,
    )

    assert weekend.format == RaceFormat.STANDARD
    assert weekend.positions == positions
    assert weekend.completion_percentage == 1.0
    assert weekend.rolling_average == 3.5
    assert weekend.teammate_position == 4


def test_driver_weekend_data_immutability():
    """Test that DriverWeekendData instances are immutable."""
    positions = DriverPositions(qualifying=3, race=2)
    weekend = DriverWeekendData(
        format=RaceFormat.STANDARD,
        positions=positions,
        completion_percentage=1.0,
        rolling_average=3.5,
        teammate_position=4,
    )

    with pytest.raises(AttributeError):
        weekend.completion_percentage = 0.5


def test_constructor_weekend_data_creation():
    """Test ConstructorWeekendData dataclass creation and attributes."""
    positions = ConstructorPositions(
        driver1_qualifying=3,
        driver1_race=2,
        driver2_qualifying=5,
        driver2_race=4,
    )
    weekend = ConstructorWeekendData(
        format=RaceFormat.STANDARD,
        positions=positions,
    )

    assert weekend.format == RaceFormat.STANDARD
    assert weekend.positions == positions


def test_constructor_weekend_data_immutability():
    """Test that ConstructorWeekendData instances are immutable."""
    positions = ConstructorPositions(
        driver1_qualifying=3,
        driver1_race=2,
        driver2_qualifying=5,
        driver2_race=4,
    )
    weekend = ConstructorWeekendData(
        format=RaceFormat.STANDARD,
        positions=positions,
    )

    with pytest.raises(AttributeError):
        weekend.format = RaceFormat.SPRINT


def test_driver_weekend_data_with_sprint():
    """Test DriverWeekendData creation with sprint format."""
    positions = DriverPositions(qualifying=3, race=2, sprint_finish=4)
    weekend = DriverWeekendData(
        format=RaceFormat.SPRINT,
        positions=positions,
        completion_percentage=1.0,
        rolling_average=3.5,
        teammate_position=5,
    )

    assert weekend.format == RaceFormat.SPRINT
    assert weekend.positions.sprint_finish == 4


# Validation Tests
def test_validate_position():
    """Test position validation."""
    # Valid positions
    validate_position(1, "test")
    validate_position(20, "test")
    validate_position(8, "test", max_pos=8)

    # Invalid positions
    with pytest.raises(ValidationError, match="must be an integer"):
        validate_position(1.5, "test")

    with pytest.raises(ValidationError, match="must be between"):
        validate_position(0, "test")

    with pytest.raises(ValidationError, match="must be between"):
        validate_position(21, "test")

    with pytest.raises(ValidationError, match="must be between"):
        validate_position(9, "test", max_pos=8)


def test_validate_percentage():
    """Test percentage validation."""
    # Valid percentages
    validate_percentage(0.0, "test")
    validate_percentage(0.5, "test")
    validate_percentage(1.0, "test")

    # Invalid percentages
    with pytest.raises(ValidationError, match="must be a float"):
        validate_percentage(1, "test")

    with pytest.raises(ValidationError, match="must be between"):
        validate_percentage(-0.1, "test")

    with pytest.raises(ValidationError, match="must be between"):
        validate_percentage(1.1, "test")


def test_validate_positive_float():
    """Test positive float validation."""
    # Valid values
    validate_positive_float(0.1, "test")
    validate_positive_float(1.0, "test")
    validate_positive_float(100.0, "test")

    # Invalid values
    with pytest.raises(ValidationError, match="must be a float"):
        validate_positive_float(1, "test")

    with pytest.raises(ValidationError, match="must be positive"):
        validate_positive_float(0.0, "test")

    with pytest.raises(ValidationError, match="must be positive"):
        validate_positive_float(-1.0, "test")


def test_validate_format_consistency():
    """Test race format consistency validation."""
    # Valid combinations
    standard_pos = DriverPositions(qualifying=1, race=1)
    sprint_pos = DriverPositions(qualifying=1, race=1, sprint_finish=1)

    validate_format_consistency(RaceFormat.STANDARD, standard_pos)
    validate_format_consistency(RaceFormat.SPRINT, sprint_pos)

    # Invalid combinations
    with pytest.raises(ValidationError, match="Sprint position required"):
        validate_format_consistency(RaceFormat.SPRINT, standard_pos)

    with pytest.raises(ValidationError, match="Sprint position not allowed"):
        validate_format_consistency(RaceFormat.STANDARD, sprint_pos)
