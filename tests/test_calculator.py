"""Tests for F1 fantasy scoring calculator."""

from typing import Any, Dict

import numpy as np
import pytest

from gridrival_ai.scoring._calculator import race_dtype
from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import Positions, RaceFormat, RaceWeekendData


@pytest.fixture
def default_config() -> ScoringConfig:
    """Create default scoring configuration."""
    return ScoringConfig()


@pytest.fixture
def scorer(default_config: ScoringConfig) -> Scorer:
    """Create Scorer with default configuration."""
    return Scorer(default_config)


@pytest.fixture
def standard_race_data() -> RaceWeekendData:
    """Create standard race data with P1 positions."""
    return RaceWeekendData(
        format=RaceFormat.STANDARD,
        positions=Positions(qualifying=1, race=1, sprint_finish=None),
        completion_percentage=1.0,
        rolling_average=2.0,
        teammate_position=2,
    )


@pytest.fixture
def sprint_race_data() -> RaceWeekendData:
    """Create sprint race data with P1 positions."""
    return RaceWeekendData(
        format=RaceFormat.SPRINT,
        positions=Positions(qualifying=1, race=1, sprint_finish=1),
        completion_percentage=1.0,
        rolling_average=2.0,
        teammate_position=2,
    )


def create_batch_data(scenarios: list[Dict[str, Any]]) -> np.ndarray:
    """Create batch data array from list of scenarios."""
    return np.array(
        [
            (
                s.get("format", RaceFormat.STANDARD.value),
                s.get("qualifying", 1),
                s.get("race", 1),
                s.get("sprint", -1),
                s.get("completion", 1.0),
                s.get("rolling_avg", 2.0),
                s.get("teammate", 2),
            )
            for s in scenarios
        ],
        dtype=race_dtype,
    )


def test_perfect_race(scorer: Scorer, standard_race_data: RaceWeekendData):
    """Test scoring for perfect race (all P1)."""
    points = scorer.calculate(standard_race_data)

    # Calculate expected points:
    # P1 Qualifying (50) + P1 Race (100) + Full Completion (12) +
    # Beating Teammate (2) + Improvement vs Average (2)
    expected = 50 + 100 + 12 + 2 + 2

    assert points == pytest.approx(expected)


def test_sprint_race(scorer: Scorer, sprint_race_data: RaceWeekendData):
    """Test scoring for sprint race format."""
    points = scorer.calculate(sprint_race_data)

    # Add P1 Sprint (8) to perfect race points
    expected = 50 + 100 + 12 + 2 + 2 + 8

    assert points == pytest.approx(expected)


def test_overtaking_points(scorer: Scorer):
    """Test points for overtaking."""
    data = RaceWeekendData(
        format=RaceFormat.STANDARD,
        positions=Positions(qualifying=10, race=5, sprint_finish=None),
        completion_percentage=1.0,
        rolling_average=10.0,
        teammate_position=6,
    )

    points = scorer.calculate(data)

    # Check overtaking points (5 positions * 3 points)
    overtake_component = 15
    assert points >= overtake_component


def test_talent_driver(scorer: Scorer, standard_race_data: RaceWeekendData):
    """Test talent driver multiplier."""
    normal_points = scorer.calculate(standard_race_data, is_talent=False)
    talent_points = scorer.calculate(standard_race_data, is_talent=True)

    assert talent_points == pytest.approx(normal_points * 2.0)


def test_batch_calculation(scorer: Scorer):
    """Test batch point calculation."""
    scenarios = [
        # Perfect race
        {"qualifying": 1, "race": 1},
        # Overtaking
        {"qualifying": 10, "race": 5},
        # Sprint race
        {"format": RaceFormat.SPRINT.value, "qualifying": 1, "race": 1, "sprint": 1},
    ]

    batch_data = create_batch_data(scenarios)
    points = scorer.calculate_batch(batch_data)

    assert len(points) == len(scenarios)
    assert all(p > 0 for p in points)
    # Sprint race should score higher than standard
    assert points[2] > points[0]


def test_completion_stages(scorer: Scorer):
    """Test points for different completion stages."""
    stages = [0.24, 0.25, 0.49, 0.50, 0.74, 0.75, 0.89, 0.90, 1.0]
    expected_points = [0, 3, 3, 6, 6, 9, 9, 12, 12]

    for completion, expected in zip(stages, expected_points):
        data = RaceWeekendData(
            format=RaceFormat.STANDARD,
            positions=Positions(qualifying=20, race=20, sprint_finish=None),
            completion_percentage=completion,
            rolling_average=20.0,
            teammate_position=20,
        )
        points = scorer.calculate(data)
        # Only compare completion points component
        completion_component = points - (
            scorer.tables.base_points[0, 20]
            + scorer.tables.base_points[1, 20]  # Qualifying  # Race
        )
        assert completion_component == pytest.approx(expected)


def test_teammate_points(scorer: Scorer):
    """Test points for beating teammate by different margins."""
    margins = {
        1: 2,  # 1-3 positions: 2 points
        5: 5,  # 4-7 positions: 5 points
        10: 8,  # 8-12 positions: 8 points
        15: 12,  # >12 positions: 12 points
    }

    for positions_ahead, expected_points in margins.items():
        data = RaceWeekendData(
            format=RaceFormat.STANDARD,
            positions=Positions(qualifying=1, race=1, sprint_finish=None),
            completion_percentage=0.0,  # No completion points
            rolling_average=1.0,  # No improvement points
            teammate_position=1 + positions_ahead,
        )
        points = scorer.calculate(data)
        # Subtract base points to isolate teammate component
        teammate_component = points - (
            scorer.tables.base_points[0, 1]
            + scorer.tables.base_points[1, 1]  # Qualifying  # Race
        )
        assert teammate_component == pytest.approx(expected_points)


def test_improvement_points(scorer: Scorer):
    """Test points for improving vs rolling average."""
    improvements = {
        1: 2,  # +1 position: 2 points
        3: 6,  # +3 positions: 6 points
        5: 12,  # +5 positions: 12 points
        7: 20,  # +7 positions: 20 points
        9: 30,  # +9 positions: 30 points
    }

    for positions_ahead, expected_points in improvements.items():
        data = RaceWeekendData(
            format=RaceFormat.STANDARD,
            positions=Positions(qualifying=1, race=1, sprint_finish=None),
            completion_percentage=0.0,
            rolling_average=1.0 + positions_ahead,
            teammate_position=1,
        )
        points = scorer.calculate(data)
        # Subtract base points to isolate improvement component
        improvement_component = points - (
            scorer.tables.base_points[0, 1]
            + scorer.tables.base_points[1, 1]  # Qualifying  # Race
        )
        assert improvement_component == pytest.approx(expected_points)


def test_array_conversion(scorer: Scorer, standard_race_data: RaceWeekendData):
    """Test conversion between RaceWeekendData and structured array."""
    arr = scorer.convert_to_array(standard_race_data)

    assert arr["format"] == RaceFormat.STANDARD.value
    assert arr["qualifying"] == standard_race_data.positions.qualifying
    assert arr["race"] == standard_race_data.positions.race
    assert arr["sprint"] == -1  # No sprint in standard race
    assert arr["completion"] == standard_race_data.completion_percentage
    assert arr["rolling_avg"] == standard_race_data.rolling_average
    assert arr["teammate"] == standard_race_data.teammate_position
