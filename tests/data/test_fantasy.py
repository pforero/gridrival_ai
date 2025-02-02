"""Tests for fantasy league data structures."""

import pytest

from gridrival_ai.data.fantasy import (
    FantasyLeagueData,
    RollingAverages,
    Salaries,
    TeamConstraints,
)


@pytest.fixture
def valid_driver_salaries():
    """Sample valid driver salaries."""
    return {
        "VER": 33.0,
        "HAM": 26.2,
        "LEC": 29.6,
    }


@pytest.fixture
def valid_constructor_salaries():
    """Sample valid constructor salaries."""
    return {
        "RBR": 30.0,
        "MER": 27.2,
        "FER": 24.4,
    }


@pytest.fixture
def valid_rolling_averages():
    """Sample valid rolling averages."""
    return {
        "VER": 1.5,
        "HAM": 3.2,
        "LEC": 2.8,
    }


class TestSalaries:
    """Test suite for Salaries class."""

    def test_valid_creation(self, valid_driver_salaries, valid_constructor_salaries):
        """Test creating Salaries with valid data."""
        salaries = Salaries(valid_driver_salaries, valid_constructor_salaries)
        assert salaries.drivers == valid_driver_salaries
        assert salaries.constructors == valid_constructor_salaries

    def test_invalid_driver_id(self, valid_constructor_salaries):
        """Test error on invalid driver ID."""
        invalid_drivers = {"INVALID": 33.0}
        with pytest.raises(ValueError, match="Invalid driver IDs"):
            Salaries(invalid_drivers, valid_constructor_salaries)

    def test_invalid_constructor_id(self, valid_driver_salaries):
        """Test error on invalid constructor ID."""
        invalid_constructors = {"INVALID": 30.0}
        with pytest.raises(ValueError, match="Invalid constructor IDs"):
            Salaries(valid_driver_salaries, invalid_constructors)

    def test_negative_salary(self, valid_driver_salaries, valid_constructor_salaries):
        """Test error on negative salary."""
        invalid_drivers = valid_driver_salaries.copy()
        invalid_drivers["VER"] = -1.0
        with pytest.raises(ValueError, match="Negative driver salaries"):
            Salaries(invalid_drivers, valid_constructor_salaries)


class TestRollingAverages:
    """Test suite for RollingAverages class."""

    def test_valid_creation(self, valid_rolling_averages):
        """Test creating RollingAverages with valid data."""
        averages = RollingAverages(valid_rolling_averages)
        assert averages.values == valid_rolling_averages

    def test_invalid_driver_id(self):
        """Test error on invalid driver ID."""
        invalid_averages = {"INVALID": 1.5}
        with pytest.raises(ValueError, match="Invalid driver IDs"):
            RollingAverages(invalid_averages)

    @pytest.mark.parametrize("invalid_value", [0.5, 20.5])
    def test_invalid_average(self, invalid_value):
        """Test error on average outside valid range."""
        invalid_averages = {"VER": invalid_value}
        with pytest.raises(ValueError, match="Invalid rolling averages"):
            RollingAverages(invalid_averages)


class TestTeamConstraints:
    """Test suite for TeamConstraints class."""

    def test_valid_creation(self):
        """Test creating TeamConstraints with valid data."""
        constraints = TeamConstraints(
            locked_in=frozenset({"HAM", "RBR"}),
            locked_out=frozenset({"VER", "FER"}),
        )
        assert "HAM" in constraints.locked_in
        assert "VER" in constraints.locked_out

    def test_empty_constraints(self):
        """Test creating TeamConstraints with no constraints."""
        constraints = TeamConstraints(
            locked_in=frozenset(),
            locked_out=frozenset(),
        )
        assert not constraints.locked_in
        assert not constraints.locked_out

    def test_invalid_ids(self):
        """Test error on invalid IDs."""
        with pytest.raises(ValueError, match="Invalid locked in IDs"):
            TeamConstraints(
                locked_in=frozenset({"INVALID"}),
                locked_out=frozenset(),
            )

    def test_overlapping_constraints(self):
        """Test error on overlapping constraints."""
        with pytest.raises(
            ValueError, match="Elements cannot be both locked in and out"
        ):
            TeamConstraints(
                locked_in=frozenset({"HAM"}),
                locked_out=frozenset({"HAM"}),
            )


class TestFantasyLeagueData:
    """Test suite for FantasyLeagueData class."""

    def test_valid_creation_from_dicts(
        self,
        valid_driver_salaries,
        valid_constructor_salaries,
        valid_rolling_averages,
    ):
        """Test creating FantasyLeagueData using from_dicts."""
        data = FantasyLeagueData.from_dicts(
            driver_salaries=valid_driver_salaries,
            constructor_salaries=valid_constructor_salaries,
            rolling_averages=valid_rolling_averages,
            locked_in={"HAM"},
            locked_out={"VER"},
        )
        assert data.salaries.drivers == valid_driver_salaries
        assert data.salaries.constructors == valid_constructor_salaries
        assert data.averages.values == valid_rolling_averages
        assert "HAM" in data.constraints.locked_in
        assert "VER" in data.constraints.locked_out

    def test_get_available_drivers(
        self,
        valid_driver_salaries,
        valid_constructor_salaries,
        valid_rolling_averages,
    ):
        """Test getting available drivers."""
        data = FantasyLeagueData.from_dicts(
            driver_salaries=valid_driver_salaries,
            constructor_salaries=valid_constructor_salaries,
            rolling_averages=valid_rolling_averages,
            locked_out={"VER"},
        )
        available = data.get_available_drivers()
        assert "HAM" in available
        assert "VER" not in available

    def test_get_available_constructors(
        self,
        valid_driver_salaries,
        valid_constructor_salaries,
        valid_rolling_averages,
    ):
        """Test getting available constructors."""
        data = FantasyLeagueData.from_dicts(
            driver_salaries=valid_driver_salaries,
            constructor_salaries=valid_constructor_salaries,
            rolling_averages=valid_rolling_averages,
            locked_out={"RBR"},
        )
        available = data.get_available_constructors()
        assert "MER" in available
        assert "RBR" not in available

    def test_default_empty_constraints(
        self,
        valid_driver_salaries,
        valid_constructor_salaries,
        valid_rolling_averages,
    ):
        """Test that constraints default to empty."""
        data = FantasyLeagueData.from_dicts(
            driver_salaries=valid_driver_salaries,
            constructor_salaries=valid_constructor_salaries,
            rolling_averages=valid_rolling_averages,
        )
        assert not data.constraints.locked_in
        assert not data.constraints.locked_out
