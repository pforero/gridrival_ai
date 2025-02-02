"""Tests for optimization types module."""

from gridrival_ai.optimization.types import (
    DriverScoring,
    OptimizationResult,
    TeamSolution,
)


def test_driver_scoring_creation():
    """Test DriverScoring dataclass creation and attributes."""
    points_dict = {"qualifying": 5.0, "race": 5.5}
    scoring = DriverScoring(
        regular_points=10.5, points_dict=points_dict, salary=15.0, can_be_talent=True
    )
    assert scoring.regular_points == 10.5
    assert scoring.points_dict == points_dict
    assert scoring.salary == 15.0
    assert scoring.can_be_talent is True


def test_team_solution_creation():
    """Test TeamSolution namedtuple creation and attributes."""
    drivers = frozenset(["VER", "HAM", "PER", "ALO", "NOR"])
    points_breakdown = {
        "VER": 25.0,
        "HAM": 18.0,
        "PER": 15.0,
        "ALO": 12.0,
        "NOR": 10.0,
        "RED": 30.0,
    }

    solution = TeamSolution(
        drivers=drivers,
        constructor="RED",
        talent_driver="NOR",
        expected_points=110.0,
        total_cost=95.5,
        points_breakdown=points_breakdown,
    )

    assert solution.drivers == drivers
    assert solution.constructor == "RED"
    assert solution.talent_driver == "NOR"
    assert solution.expected_points == 110.0
    assert solution.total_cost == 95.5
    assert solution.points_breakdown == points_breakdown


def test_optimization_result_creation():
    """Test OptimizationResult dataclass creation and attributes."""
    drivers = frozenset(["VER", "HAM", "PER", "ALO", "NOR"])
    points_breakdown = {"VER": 25.0, "HAM": 18.0, "PER": 15.0, "ALO": 12.0, "NOR": 10.0}

    solution = TeamSolution(
        drivers=drivers,
        constructor="RED",
        talent_driver="NOR",
        expected_points=110.0,
        total_cost=95.5,
        points_breakdown=points_breakdown,
    )

    result = OptimizationResult(
        best_solution=solution,
        alternative_solutions=[],
        remaining_budget=4.5,
        error_message=None,
    )

    assert result.best_solution == solution
    assert result.alternative_solutions == []
    assert result.remaining_budget == 4.5
    assert result.error_message is None


def test_optimization_result_with_no_solution():
    """Test OptimizationResult creation with no valid solution."""
    result = OptimizationResult(
        best_solution=None,
        alternative_solutions=[],
        remaining_budget=0.0,
        error_message="No valid solution found under budget constraints",
    )

    assert result.best_solution is None
    assert result.alternative_solutions == []
    assert result.remaining_budget == 0.0
    assert result.error_message == "No valid solution found under budget constraints"
