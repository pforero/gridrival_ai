"""
Tests for the constructor points calculator module.

This module contains tests for the ConstructorPointsCalculator class,
validating the calculation of constructor fantasy points.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gridrival_ai.points.constructor import ConstructorPointsCalculator
from gridrival_ai.points.distributions import DistributionAdapter
from gridrival_ai.probabilities.core import PositionDistribution
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat


@pytest.fixture
def mock_adapter():
    """Create a mock distribution adapter."""
    adapter = MagicMock(spec=DistributionAdapter)

    # Set up the get_constructor_drivers method to return fixed drivers
    adapter.get_constructor_drivers.return_value = ("VER", "LAW")

    # Mock position distributions
    ver_qual = PositionDistribution({1: 0.7, 2: 0.2, 3: 0.1})
    law_qual = PositionDistribution({3: 0.3, 4: 0.4, 5: 0.3})
    ver_race = PositionDistribution({1: 0.6, 2: 0.3, 3: 0.1})
    law_race = PositionDistribution({3: 0.2, 4: 0.5, 5: 0.3})

    # Configure adapter to return position distributions
    def get_position_distribution(driver_id, session):
        if driver_id == "VER" and session == "qualifying":
            return ver_qual
        elif driver_id == "VER" and session == "race":
            return ver_race
        elif driver_id == "LAW" and session == "qualifying":
            return law_qual
        elif driver_id == "LAW" and session == "race":
            return law_race
        else:
            raise KeyError(f"No distribution for {driver_id} in {session}")

    adapter.get_position_distribution.side_effect = get_position_distribution

    return adapter


@pytest.fixture
def mock_scorer():
    """Create a mock scorer with predefined scoring tables."""
    scorer = MagicMock(spec=ScoringCalculator)

    # Constructor points tables (qualifying and race)
    # Using 0-indexed arrays where index 0 is reserved (not used)
    qual_points = np.zeros(21)
    qual_points[1:6] = [18, 15, 12, 10, 8]  # Points for P1-P5
    qual_points[6:11] = [6, 4, 2, 1, 0]  # Points for P6-P10

    race_points = np.zeros(21)
    race_points[1:6] = [25, 18, 15, 12, 10]  # Points for P1-P5
    race_points[6:11] = [8, 6, 4, 2, 1]  # Points for P6-P10

    # Set up tables in the scorer
    class Tables:
        def __init__(self):
            self.constructor_points = [qual_points, race_points]

    scorer.tables = Tables()

    return scorer


def test_constructor_calculator_initialization(mock_adapter, mock_scorer):
    """Test that the constructor points calculator initializes correctly."""
    calculator = ConstructorPointsCalculator(mock_adapter, mock_scorer)

    assert calculator.distributions == mock_adapter
    assert calculator.scorer == mock_scorer
    assert hasattr(calculator, "position_calculator")


def test_constructor_points_calculation(mock_adapter, mock_scorer):
    """Test the calculation of constructor points."""
    calculator = ConstructorPointsCalculator(mock_adapter, mock_scorer)

    # Calculate points for Red Bull
    points = calculator.calculate("RBR", RaceFormat.STANDARD)

    # Verify points structure
    assert isinstance(points, dict)
    assert "qualifying" in points
    assert "race" in points

    # Expected qualifying points: VER (P1: 18 * 0.7, P2: 15 * 0.2, P3: 12 * 0.1) +
    #                           PER (P3: 12 * 0.3, P4: 10 * 0.4, P5: 8 * 0.3)
    # = 0.7*18 + 0.2*15 + 0.1*12 + 0.3*12 + 0.4*10 + 0.3*8 = 12.6 + 3.0 + 1.2 + 3.6
    # + 4.0 + 2.4 = 26.8
    expected_qual_points = 26.8

    # Expected race points: VER (P1: 25 * 0.6, P2: 18 * 0.3, P3: 15 * 0.1) +
    #                      PER (P3: 15 * 0.2, P4: 12 * 0.5, P5: 10 * 0.3)
    # = 0.6*25 + 0.3*18 + 0.1*15 + 0.2*15 + 0.5*12 + 0.3*10 = 15.0 + 5.4 + 1.5 + 3.0
    # + 6.0 + 3.0 = 33.9
    expected_race_points = 33.9

    # Check that calculated points match expected values
    assert pytest.approx(points["qualifying"], 0.1) == expected_qual_points
    assert pytest.approx(points["race"], 0.1) == expected_race_points


def test_constructor_not_found(mock_adapter, mock_scorer):
    """Test handling when constructor is not found."""
    calculator = ConstructorPointsCalculator(mock_adapter, mock_scorer)

    # Configure mock to raise KeyError for unknown constructor
    mock_adapter.get_constructor_drivers.side_effect = KeyError("Constructor not found")

    # Calculate should return empty results with zero points
    points = calculator.calculate("UNKNOWN", RaceFormat.STANDARD)

    assert points["qualifying"] == 0.0
    assert points["race"] == 0.0


def test_missing_driver_distribution(mock_adapter, mock_scorer):
    """Test handling when one driver's distribution is missing."""
    calculator = ConstructorPointsCalculator(mock_adapter, mock_scorer)

    # Modify the mock to raise KeyError for one driver but not the other
    original_side_effect = mock_adapter.get_position_distribution.side_effect

    def modified_get_position(driver_id, session):
        if driver_id == "LAW":
            raise KeyError(f"No distribution for {driver_id}")
        return original_side_effect(driver_id, session)

    mock_adapter.get_position_distribution.side_effect = modified_get_position

    # Calculate should work with only one driver's points
    points = calculator.calculate("RBR", RaceFormat.STANDARD)

    # Should only include VER's points
    assert points["qualifying"] > 0.0
    assert points["race"] > 0.0

    # Expected qualifying points: Only VER (P1: 18 * 0.7, P2: 15 * 0.2, P3: 12 * 0.1)
    # = 0.7*18 + 0.2*15 + 0.1*12 = 12.6 + 3.0 + 1.2 = 16.8
    expected_qual_points = 16.8

    # Expected race points: Only VER (P1: 25 * 0.6, P2: 18 * 0.3, P3: 15 * 0.1)
    # = 0.6*25 + 0.3*18 + 0.1*15 = 15.0 + 5.4 + 1.5 = 21.9
    expected_race_points = 21.9

    assert pytest.approx(points["qualifying"], 0.1) == expected_qual_points
    assert pytest.approx(points["race"], 0.1) == expected_race_points


def test_different_race_formats(mock_adapter, mock_scorer):
    """Test that race format doesn't change constructor points calculation."""
    calculator = ConstructorPointsCalculator(mock_adapter, mock_scorer)

    # Calculate points for both race formats
    standard_points = calculator.calculate("RBR", RaceFormat.STANDARD)
    sprint_points = calculator.calculate("RBR", RaceFormat.SPRINT)

    # Points should be the same regardless of race format
    assert standard_points["qualifying"] == sprint_points["qualifying"]
    assert standard_points["race"] == sprint_points["race"]
    assert len(standard_points) == len(sprint_points)


def test_integration_with_position_calculator(mock_adapter, mock_scorer):
    """Test that the calculator correctly uses the position calculator."""
    calculator = ConstructorPointsCalculator(mock_adapter, mock_scorer)

    # Patch the position calculator's calculate method to verify it's called correctly
    with patch(
        "gridrival_ai.points.components.PositionPointsCalculator.calculate"
    ) as mock_calc:
        # Set up the mock to return a fixed value
        mock_calc.return_value = 25.0

        # Calculate constructor points
        points = calculator.calculate("RBR", RaceFormat.STANDARD)

        # Verify that position calculator was called for qualifying and race
        assert mock_calc.call_count == 4  # Twice for each driver (qual + race)

        # Verify points reflect the mocked return value
        assert points["qualifying"] == 50.0  # 25.0 * 2 drivers
        assert points["race"] == 50.0  # 25.0 * 2 drivers
