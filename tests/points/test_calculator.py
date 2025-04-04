"""
Tests for the main PointsCalculator module.

This module contains tests for the PointsCalculator class,
which orchestrates the calculation of expected fantasy points
for drivers and constructors.
"""

from unittest.mock import MagicMock

import pytest

from gridrival_ai.points.calculator import PointsCalculator
from gridrival_ai.points.constructor import ConstructorPointsCalculator
from gridrival_ai.points.driver import DriverPointsCalculator
from gridrival_ai.probabilities.distributions import (
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat


@pytest.fixture
def mock_race_distribution():
    """Create a mock race distribution."""
    mock_race = MagicMock(spec=RaceDistribution)

    # Mock session distributions
    mock_race.race = MagicMock(spec=SessionDistribution)
    mock_race.qualifying = MagicMock(spec=SessionDistribution)
    mock_race.sprint = MagicMock(spec=SessionDistribution)

    return mock_race


@pytest.fixture
def mock_scorer():
    """Create a mock scorer."""
    return MagicMock(spec=ScoringCalculator)


@pytest.fixture
def mock_driver_stats():
    """Create sample driver rolling average statistics."""
    return {
        "VER": 1.5,
        "PER": 3.2,
        "HAM": 2.8,
        "RUS": 4.5,
        "LEC": 3.3,
        "SAI": 4.2,
    }


@pytest.fixture
def calculator(mock_race_distribution, mock_scorer, mock_driver_stats):
    """Create a PointsCalculator instance with mocks."""
    return PointsCalculator(
        scorer=mock_scorer,
        race_distribution=mock_race_distribution,
        driver_stats=mock_driver_stats,
    )


def test_calculator_initialization(
    mock_race_distribution, mock_scorer, mock_driver_stats
):
    """Test that the calculator initializes correctly with its dependencies."""
    calculator = PointsCalculator(
        mock_scorer, mock_race_distribution, mock_driver_stats
    )

    # Check that calculator has all required attributes
    assert hasattr(calculator, "scorer")
    assert hasattr(calculator, "race_distribution")
    assert hasattr(calculator, "driver_stats")
    assert hasattr(calculator, "driver_calculator")
    assert hasattr(calculator, "constructor_calculator")

    # Check that dependencies are correctly assigned
    assert calculator.scorer == mock_scorer
    assert calculator.driver_stats == mock_driver_stats
    assert calculator.race_distribution == mock_race_distribution

    # Check that component calculators are initialized
    assert isinstance(calculator.driver_calculator, DriverPointsCalculator)
    assert isinstance(calculator.constructor_calculator, ConstructorPointsCalculator)


def test_calculate_driver_points(calculator):
    """Test calculating driver points delegates to the driver calculator."""
    # Mock the driver calculator's calculate method
    driver_result = {
        "qualifying": 20.5,
        "race": 25.3,
        "overtake": 6.2,
        "teammate": 2.0,
        "completion": 12.0,
        "improvement": 4.5,
    }
    calculator.driver_calculator.calculate = MagicMock(return_value=driver_result)

    # Call the method under test
    result = calculator.calculate_driver_points("VER")

    # Check that the result matches the mock
    assert result == driver_result

    # Verify correct arguments were passed to driver calculator
    calculator.driver_calculator.calculate.assert_called_once_with(
        driver_id="VER",
        rolling_avg=1.5,  # From mock_driver_stats
        teammate_id="TSU",  # Should find teammate from CONSTRUCTORS
        race_format=RaceFormat.STANDARD,
    )


def test_calculate_driver_points_with_sprint(calculator):
    """Test calculating driver points for a sprint race weekend."""
    # Mock the driver calculator's calculate method
    driver_result = {
        "qualifying": 20.5,
        "race": 25.3,
        "sprint": 12.5,
        "overtake": 6.2,
        "teammate": 2.0,
        "completion": 12.0,
        "improvement": 4.5,
    }
    calculator.driver_calculator.calculate = MagicMock(return_value=driver_result)

    # Call the method under test with sprint format
    result = calculator.calculate_driver_points("VER", race_format=RaceFormat.SPRINT)

    # Check that the result matches the mock
    assert result == driver_result

    # Verify sprint format was passed to driver calculator
    calculator.driver_calculator.calculate.assert_called_once_with(
        driver_id="VER",
        rolling_avg=1.5,  # From mock_driver_stats
        teammate_id="TSU",  # Should find teammate from CONSTRUCTORS
        race_format=RaceFormat.SPRINT,
    )


def test_calculate_constructor_points(calculator):
    """Test calculating constructor points delegates to the constructor calculator."""
    # Mock the constructor calculator's calculate method
    constructor_result = {
        "qualifying": 35.8,
        "race": 48.7,
    }
    calculator.constructor_calculator.calculate = MagicMock(
        return_value=constructor_result
    )

    # Call the method under test
    result = calculator.calculate_constructor_points("RBR")

    # Check that the result matches the mock
    assert result == constructor_result

    # Verify correct arguments were passed to constructor calculator
    calculator.constructor_calculator.calculate.assert_called_once_with(
        constructor_id="RBR", race_format=RaceFormat.STANDARD
    )


def test_calculate_constructor_points_with_sprint(calculator):
    """Test calculating constructor points for a sprint race weekend."""
    # Mock the constructor calculator's calculate method
    constructor_result = {
        "qualifying": 35.8,
        "race": 48.7,
    }
    calculator.constructor_calculator.calculate = MagicMock(
        return_value=constructor_result
    )

    # Call the method under test with sprint format
    result = calculator.calculate_constructor_points(
        "RBR", race_format=RaceFormat.SPRINT
    )

    # Check that the result matches the mock
    assert result == constructor_result

    # Verify sprint format was passed to constructor calculator
    calculator.constructor_calculator.calculate.assert_called_once_with(
        constructor_id="RBR", race_format=RaceFormat.SPRINT
    )


def test_constructor_not_found(calculator):
    """Test handling when constructor is not found."""
    # Mock the constructor calculator to raise KeyError
    calculator.constructor_calculator.calculate = MagicMock(
        side_effect=KeyError("Constructor not found")
    )

    # Calculate points for unknown constructor should propagate the error
    with pytest.raises(KeyError, match="Constructor not found"):
        calculator.calculate_constructor_points("UNKNOWN")


def test_integration_with_component_calculators(calculator):
    """Test that the calculator integrates with its component calculators."""
    # Mock driver and constructor calculators to return sample results
    driver_result = {"qualifying": 20.5, "race": 25.3}
    constructor_result = {"qualifying": 35.8, "race": 48.7}

    calculator.driver_calculator.calculate = MagicMock(return_value=driver_result)
    calculator.constructor_calculator.calculate = MagicMock(
        return_value=constructor_result
    )

    # Call both calculation methods
    dr_points = calculator.calculate_driver_points("VER")
    con_points = calculator.calculate_constructor_points("RBR")

    # Verify results
    assert dr_points == driver_result
    assert con_points == constructor_result

    # Verify both calculators were called
    assert calculator.driver_calculator.calculate.called
    assert calculator.constructor_calculator.calculate.called
