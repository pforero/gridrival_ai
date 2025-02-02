"""Tests for F1 fantasy team optimization.

Tests cover:
1. Basic optimization functionality
2. Budget constraints
3. Locked-in/out constraints
4. Talent driver selection
5. Error handling
6. Alternative solutions
"""

from unittest.mock import Mock

import pytest

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.team_optimizer import TeamOptimizer
from gridrival_ai.scoring.types import RaceFormat


@pytest.fixture
def mock_points_calculator():
    """Create mock points calculator with controlled outputs."""
    calculator = Mock()

    # Default points for drivers
    def get_driver_points(driver_id, format=None, is_talent=False):
        # Points roughly proportional to cost for testing
        points_map = {
            "VER": 100.0,  # Expensive, high points
            "LAW": 80.0,
            "RUS": 85.0,
            "ANT": 75.0,
            "LEC": 82.0,
            "HAM": 78.0,
            "NOR": 70.0,
            "PIA": 60.0,  # Cheap enough for talent
            "ALO": 72.0,
            "STR": 58.0,  # Cheap enough for talent
        }
        return points_map.get(driver_id, 50.0)

    calculator.calculate_driver_points = Mock(side_effect=get_driver_points)

    # Default points for constructors
    def get_constructor_points(constructor_id):
        points_map = {
            "RBR": 150.0,
            "MER": 130.0,
            "FER": 125.0,
            "MCL": 110.0,
            "AST": 100.0,
        }
        return points_map.get(constructor_id, 80.0)

    calculator.calculate_constructor_points = Mock(side_effect=get_constructor_points)

    return calculator


@pytest.fixture
def sample_league_data():
    """Create sample league data with realistic values."""
    driver_salaries = {
        "VER": 33.0,  # Too expensive for talent
        "LAW": 19.0,
        "RUS": 21.0,
        "ANT": 16.0,  # Eligible for talent
        "LEC": 22.0,
        "HAM": 23.0,
        "NOR": 18.0,  # Just at talent threshold
        "PIA": 14.0,  # Eligible for talent
        "ALO": 17.0,  # Eligible for talent
        "STR": 15.0,  # Eligible for talent
    }

    constructor_salaries = {
        "RBR": 23.0,
        "MER": 20.0,
        "FER": 18.0,
        "MCL": 16.0,
        "AST": 14.0,
    }

    # Default empty constraints
    league_data = FantasyLeagueData.from_dicts(
        driver_salaries=driver_salaries,
        constructor_salaries=constructor_salaries,
        rolling_averages={d: 2.0 for d in driver_salaries},
        locked_in=set(),
        locked_out=set(),
    )

    return league_data


def test_basic_optimization(mock_points_calculator, sample_league_data):
    """Test basic optimization with no constraints."""
    optimizer = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    assert len(result.best_solution.drivers) == 5
    assert result.best_solution.total_cost <= 100.0
    assert result.best_solution.constructor in sample_league_data.salaries.constructors
    assert result.best_solution.talent_driver in result.best_solution.drivers

    # Talent driver should be eligible
    talent_salary = sample_league_data.salaries.drivers[
        result.best_solution.talent_driver
    ]
    assert talent_salary <= 18.0


def test_budget_constraint(mock_points_calculator, sample_league_data):
    """Test optimization with tight budget constraint."""
    # Set budget that should allow only cheapest valid teams
    optimizer = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
        budget=95.0,  # Just enough for cheapest valid team
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    assert result.best_solution.total_cost <= 95.0

    # Test impossible budget
    optimizer_impossible = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
        budget=50.0,  # Too low for any valid team
    )

    result_impossible = optimizer_impossible.optimize()

    assert result_impossible.best_solution is None
    assert "No valid team composition found" in result_impossible.error_message


def test_locked_in_constraints(mock_points_calculator, sample_league_data):
    """Test optimization with locked-in drivers."""
    # Lock in one expensive and one cheap driver
    locked_league_data = FantasyLeagueData.from_dicts(
        driver_salaries=sample_league_data.salaries.drivers,
        constructor_salaries=sample_league_data.salaries.constructors,
        rolling_averages={d: 2.0 for d in sample_league_data.salaries.drivers},
        locked_in={"LAW", "PIA"},  # One medium, one cheap
        locked_out=set(),
    )

    optimizer = TeamOptimizer(
        league_data=locked_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    assert "LAW" in result.best_solution.drivers
    assert "PIA" in result.best_solution.drivers


def test_locked_out_constraints(mock_points_calculator, sample_league_data):
    """Test optimization with locked-out drivers."""
    # Lock out the highest scoring drivers
    locked_league_data = FantasyLeagueData.from_dicts(
        driver_salaries=sample_league_data.salaries.drivers,
        constructor_salaries=sample_league_data.salaries.constructors,
        rolling_averages={d: 2.0 for d in sample_league_data.salaries.drivers},
        locked_in=set(),
        locked_out={"VER", "LAW", "RUS"},
    )

    optimizer = TeamOptimizer(
        league_data=locked_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    assert "VER" not in result.best_solution.drivers
    assert "LAW" not in result.best_solution.drivers
    assert "RUS" not in result.best_solution.drivers


def test_talent_driver_constraints(mock_points_calculator, sample_league_data):
    """Test talent driver selection constraints."""
    # Test with all expensive drivers (no talent eligible)
    expensive_league_data = FantasyLeagueData.from_dicts(
        driver_salaries={
            "VER": 19.0,
            "LAW": 18.5,
            "RUS": 18.5,
            "LEC": 18.5,
            "HAM": 18.5,
            "NOR": 18.5,
        },
        constructor_salaries={"AST": 5.0},  # Very cheap constructor
        rolling_averages={
            "VER": 2.0,
            "LAW": 2.0,
            "RUS": 2.0,
            "LEC": 2.0,
            "HAM": 2.0,
            "NOR": 2.0,
        },
        locked_in=set(),
        locked_out=set(),
    )

    optimizer = TeamOptimizer(
        league_data=expensive_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    assert result.best_solution.talent_driver == ""  # No talent driver
    assert len(result.best_solution.drivers) == 5  # Still valid team

    # Test with mix of expensive and cheap drivers
    optimizer_mixed = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result_mixed = optimizer_mixed.optimize()

    assert result_mixed.best_solution is not None
    assert (
        result_mixed.best_solution.talent_driver != ""
    )  # Should use talent driver when available
    talent_salary = sample_league_data.salaries.drivers[
        result_mixed.best_solution.talent_driver
    ]
    assert talent_salary <= 18.0  # Must be eligible


def test_sprint_format(mock_points_calculator, sample_league_data):
    """Test optimization with sprint race format."""
    optimizer = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.SPRINT,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    # Points calculation should have used sprint format for all drivers
    for driver_id in result.best_solution.drivers:
        mock_points_calculator.calculate_driver_points.assert_any_call(
            driver_id,
            format=RaceFormat.SPRINT,
            is_talent=False,  # No talent driver in sprint format
        )


def test_alternative_solutions(mock_points_calculator, sample_league_data):
    """Test finding alternative optimal solutions."""

    # Modify points calculator to create ties
    def equal_points(driver_id, format=None, is_talent=False):
        return 70.0  # Same points for all drivers

    mock_points_calculator.calculate_driver_points = Mock(side_effect=equal_points)

    optimizer = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    assert len(result.alternative_solutions) > 0  # Should find multiple equal solutions


def test_error_handling(mock_points_calculator, sample_league_data):
    """Test error handling in optimization."""
    # Test with incompatible constraints
    locked_league_data = FantasyLeagueData.from_dicts(
        driver_salaries=sample_league_data.salaries.drivers,
        constructor_salaries=sample_league_data.salaries.constructors,
        rolling_averages={d: 2.0 for d in sample_league_data.salaries.drivers},
        locked_in={"VER", "LAW", "RUS", "ANT", "LEC", "HAM"},  # Too many drivers
        locked_out=set(),
    )

    optimizer = TeamOptimizer(
        league_data=locked_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result = optimizer.optimize()

    assert result.best_solution is None
    assert result.error_message is not None


def test_points_breakdown(mock_points_calculator, sample_league_data):
    """Test points breakdown in solution."""
    optimizer = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    breakdown = result.best_solution.points_breakdown

    # Should have points for each driver and constructor
    assert len(breakdown) == 6  # 5 drivers + 1 constructor

    # Talent driver should have double points
    talent_points = breakdown[result.best_solution.talent_driver]
    regular_points = mock_points_calculator.calculate_driver_points(
        result.best_solution.talent_driver
    )
    assert talent_points == regular_points * 2


def test_remaining_budget(mock_points_calculator, sample_league_data):
    """Test remaining budget calculation."""
    optimizer = TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_format=RaceFormat.STANDARD,
        budget=100.0,
    )

    result = optimizer.optimize()

    assert result.best_solution is not None
    assert result.remaining_budget == 100.0 - result.best_solution.total_cost
    assert result.remaining_budget >= 0
