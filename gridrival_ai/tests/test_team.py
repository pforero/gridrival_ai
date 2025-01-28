"""Tests for the Team class."""

import pytest

from gridrival_ai.team import Team


@pytest.fixture
def empty_team():
    """Create an empty team with 100.0 budget."""
    return Team(budget=100.0)


@pytest.fixture
def valid_team():
    """Create a valid team with two drivers and a constructor."""
    team = Team(budget=100.0)
    team.add_driver("Max Verstappen", 45.0)
    team.add_driver("Sergio Perez", 35.0)
    team.add_constructor("Red Bull Racing", 15.0)
    return team


def test_team_initialization():
    """Test team initialization with default values."""
    team = Team(budget=100.0)
    assert team.budget == 100.0
    assert team.drivers == []
    assert team.constructor is None
    assert team.total_value == 0.0


def test_add_driver_success(empty_team):
    """Test successfully adding a driver to the team."""
    assert empty_team.add_driver("Max Verstappen", 45.0)
    assert "Max Verstappen" in empty_team.drivers
    assert empty_team.budget == 55.0
    assert empty_team.total_value == 45.0


def test_add_driver_exceeds_max(empty_team):
    """Test adding more drivers than allowed."""
    empty_team.add_driver("Max Verstappen", 45.0)
    empty_team.add_driver("Sergio Perez", 35.0)
    
    with pytest.raises(ValueError, match="Maximum number of drivers already reached"):
        empty_team.add_driver("Lewis Hamilton", 40.0)


def test_add_driver_exceeds_budget(empty_team):
    """Test adding a driver that exceeds the budget."""
    with pytest.raises(ValueError, match="Contract value exceeds available budget"):
        empty_team.add_driver("Max Verstappen", 150.0)


def test_add_constructor_success(empty_team):
    """Test successfully adding a constructor to the team."""
    assert empty_team.add_constructor("Red Bull Racing", 35.0)
    assert empty_team.constructor == "Red Bull Racing"
    assert empty_team.budget == 65.0
    assert empty_team.total_value == 35.0


def test_add_constructor_already_exists(empty_team):
    """Test adding a constructor when one already exists."""
    empty_team.add_constructor("Red Bull Racing", 35.0)
    
    with pytest.raises(ValueError, match="Team already has a constructor"):
        empty_team.add_constructor("Ferrari", 30.0)


def test_is_valid_complete_team(valid_team):
    """Test validation of a complete and valid team."""
    is_valid, message = valid_team.is_valid()
    assert is_valid
    assert message == "Team is valid"


def test_is_valid_incomplete_team(empty_team):
    """Test validation of an incomplete team."""
    is_valid, message = empty_team.is_valid()
    assert not is_valid
    assert "needs exactly 2 drivers" in message

    empty_team.add_driver("Max Verstappen", 45.0)
    is_valid, message = empty_team.is_valid()
    assert not is_valid
    assert "needs exactly 2 drivers" in message


def test_str_representation(valid_team):
    """Test string representation of a team."""
    team_str = str(valid_team)
    assert "Max Verstappen" in team_str
    assert "Sergio Perez" in team_str
    assert "Red Bull Racing" in team_str
    assert "Total Value: $95.0M" in team_str
    assert "Remaining Budget: $5.0M" in team_str 