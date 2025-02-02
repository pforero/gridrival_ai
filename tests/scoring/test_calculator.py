"""Tests for F1 fantasy scoring calculator."""

from typing import Any, Dict

import numpy as np
import pytest

from gridrival_ai.scoring._calculator import constructor_dtype, race_dtype
from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import (
    ConstructorPositions,
    ConstructorWeekendData,
    DriverPositions,
    DriverWeekendData,
    RaceFormat,
)


@pytest.fixture
def scoring_config():
    """Create a sample scoring configuration for testing."""
    # Create points for all positions (1-20)
    qualifying_points = {i: max(0, 10 - (i - 1)) for i in range(1, 21)}
    race_points = {i: max(0, 25 - (i - 1) * 3) for i in range(1, 21)}
    sprint_points = {i: max(0, 8 - (i - 1)) for i in range(1, 9)}
    constructor_qualifying_points = {i: max(0, 5 - (i - 1)) for i in range(1, 21)}
    constructor_race_points = {i: max(0, 12 - (i - 1)) for i in range(1, 21)}

    return ScoringConfig(
        qualifying_points=qualifying_points,
        race_points=race_points,
        sprint_points=sprint_points,
        constructor_qualifying_points=constructor_qualifying_points,
        constructor_race_points=constructor_race_points,
        completion_stage_points=5.0,
        overtake_multiplier=2.0,
        improvement_points={1: 2, 2: 4, 3: 6},
        teammate_points={3: 2, 7: 5, 12: 8},
    )


@pytest.fixture
def default_config() -> ScoringConfig:
    """Create default scoring configuration."""
    return ScoringConfig()


@pytest.fixture
def scorer(default_config: ScoringConfig) -> Scorer:
    """Create Scorer with default configuration."""
    return Scorer(default_config)


@pytest.fixture
def standard_driver_data() -> DriverWeekendData:
    """Create standard race data with P1 positions."""
    return DriverWeekendData(
        format=RaceFormat.STANDARD,
        positions=DriverPositions(qualifying=1, race=1),
        completion_percentage=1.0,
        rolling_average=2.0,
        teammate_position=2,
    )


@pytest.fixture
def sprint_driver_data() -> DriverWeekendData:
    """Create sprint race data with P1 positions."""
    return DriverWeekendData(
        format=RaceFormat.SPRINT,
        positions=DriverPositions(qualifying=1, race=1, sprint_finish=1),
        completion_percentage=1.0,
        rolling_average=2.0,
        teammate_position=2,
    )


@pytest.fixture
def standard_constructor_data() -> ConstructorWeekendData:
    """Create standard race data for constructor with P1/P2 positions."""
    return ConstructorWeekendData(
        format=RaceFormat.STANDARD,
        positions=ConstructorPositions(
            driver1_qualifying=1,
            driver1_race=1,
            driver2_qualifying=2,
            driver2_race=2,
        ),
    )


@pytest.fixture
def sprint_constructor_data() -> ConstructorWeekendData:
    """Create sprint race data for constructor with P1/P2 positions."""
    return ConstructorWeekendData(
        format=RaceFormat.SPRINT,
        positions=ConstructorPositions(
            driver1_qualifying=1,
            driver1_race=1,
            driver2_qualifying=2,
            driver2_race=2,
        ),
    )


def create_driver_batch_data(scenarios: list[Dict[str, Any]]) -> np.ndarray:
    """Create batch data array from list of driver scenarios."""
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


def create_constructor_batch_data(scenarios: list[Dict[str, Any]]) -> np.ndarray:
    """Create batch data array from list of constructor scenarios."""
    return np.array(
        [
            (
                s.get("format", RaceFormat.STANDARD.value),
                s.get("qualifying1", 1),
                s.get("qualifying2", 2),
                s.get("race1", 1),
                s.get("race2", 2),
            )
            for s in scenarios
        ],
        dtype=constructor_dtype,
    )


def test_perfect_driver_race(scorer: Scorer, standard_driver_data: DriverWeekendData):
    """Test scoring for perfect driver race (all P1)."""
    points = scorer.calculate_driver(standard_driver_data)

    # Calculate expected points:
    # P1 Qualifying (50) + P1 Race (100) + Full Completion (12) +
    # Beating Teammate (2) + Improvement vs Average (2)
    expected = 50 + 100 + 12 + 2 + 2

    assert points == pytest.approx(expected)


def test_sprint_driver_race(scorer: Scorer, sprint_driver_data: DriverWeekendData):
    """Test scoring for sprint race format."""
    points = scorer.calculate_driver(sprint_driver_data)

    # Add P1 Sprint (8) to perfect race points
    expected = 50 + 100 + 12 + 2 + 2 + 8

    assert points == pytest.approx(expected)


def test_perfect_constructor_race(
    scorer: Scorer, standard_constructor_data: ConstructorWeekendData
):
    """Test scoring for perfect constructor race (P1/P2)."""
    points = scorer.calculate_constructor(standard_constructor_data)

    # Calculate expected points:
    # Driver 1: P1 Qualifying (30) + P1 Race (60)
    # Driver 2: P2 Qualifying (29) + P2 Race (58)
    expected = (30 + 60) + (29 + 58)

    assert points == pytest.approx(expected)


def test_sprint_constructor_race(
    scorer: Scorer, sprint_constructor_data: ConstructorWeekendData
):
    """Test scoring for constructor in sprint format."""
    points = scorer.calculate_constructor(sprint_constructor_data)

    # Points should be same as standard race since constructors don't get sprint points
    # Driver 1: P1 Qualifying (30) + P1 Race (60)
    # Driver 2: P2 Qualifying (29) + P2 Race (58)
    expected = (30 + 60) + (29 + 58)

    assert points == pytest.approx(expected)


def test_driver_overtaking_points(scorer: Scorer):
    """Test points for driver overtaking."""
    data = DriverWeekendData(
        format=RaceFormat.STANDARD,
        positions=DriverPositions(qualifying=10, race=5),
        completion_percentage=1.0,
        rolling_average=10.0,
        teammate_position=6,
    )

    points = scorer.calculate_driver(data)

    # Check overtaking points (5 positions * 2 points)
    overtake_component = 10
    assert points >= overtake_component


def test_driver_batch_calculation(scorer: Scorer):
    """Test batch point calculation for drivers."""
    scenarios = [
        # Perfect race
        {"qualifying": 1, "race": 1},
        # Overtaking
        {"qualifying": 10, "race": 5},
        # Sprint race
        {"format": RaceFormat.SPRINT.value, "qualifying": 1, "race": 1, "sprint": 1},
    ]

    batch_data = create_driver_batch_data(scenarios)
    points = scorer.calculate_driver_batch(batch_data)

    assert len(points) == len(scenarios)
    assert all(p > 0 for p in points)
    # Sprint race should score higher than standard
    assert points[2] > points[0]


def test_constructor_batch_calculation(scorer: Scorer):
    """Test batch point calculation for constructors."""
    scenarios = [
        # Perfect race (P1/P2)
        {
            "qualifying1": 1,
            "qualifying2": 2,
            "race1": 1,
            "race2": 2,
        },
        # Mixed positions
        {
            "qualifying1": 5,
            "qualifying2": 6,
            "race1": 3,
            "race2": 4,
        },
    ]

    batch_data = create_constructor_batch_data(scenarios)
    points = scorer.calculate_constructor_batch(batch_data)

    assert len(points) == len(scenarios)
    assert all(p > 0 for p in points)
    # Perfect race should score higher
    assert points[0] > points[1]


def test_completion_stages(scorer: Scorer):
    """Test points for different completion stages."""
    stages = [0.24, 0.25, 0.49, 0.50, 0.74, 0.75, 0.89, 0.90, 1.0]
    expected_points = [0, 3, 3, 6, 6, 9, 9, 12, 12]

    for completion, expected in zip(stages, expected_points):
        data = DriverWeekendData(
            format=RaceFormat.STANDARD,
            positions=DriverPositions(qualifying=20, race=20),
            completion_percentage=completion,
            rolling_average=20.0,
            teammate_position=20,
        )
        points = scorer.calculate_driver(data)
        # Only compare completion points component
        completion_component = points - (
            scorer.tables.driver_points[0, 20]  # Qualifying
            + scorer.tables.driver_points[1, 20]  # Race
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
        data = DriverWeekendData(
            format=RaceFormat.STANDARD,
            positions=DriverPositions(qualifying=1, race=1),
            completion_percentage=0.0,  # No completion points
            rolling_average=1.0,  # No improvement points
            teammate_position=1 + positions_ahead,
        )
        points = scorer.calculate_driver(data)
        # Subtract base points to isolate teammate component
        teammate_component = points - (
            scorer.tables.driver_points[0, 1]  # Qualifying
            + scorer.tables.driver_points[1, 1]  # Race
        )
        assert teammate_component == pytest.approx(expected_points)


def test_improvement_points(scorer: Scorer):
    """Test points for improving vs rolling average."""
    improvements = {
        1: 2,  # +1 position: 2 points
        2: 4,  # +2 positions: 4 points
        3: 6,  # +3 positions: 6 points
    }

    for positions_ahead, expected_points in improvements.items():
        data = DriverWeekendData(
            format=RaceFormat.STANDARD,
            positions=DriverPositions(qualifying=1, race=1),
            completion_percentage=0.0,
            rolling_average=1.0 + positions_ahead,
            teammate_position=1,
        )
        points = scorer.calculate_driver(data)
        # Subtract base points to isolate improvement component
        improvement_component = points - (
            scorer.tables.driver_points[0, 1]  # Qualifying
            + scorer.tables.driver_points[1, 1]  # Race
        )
        assert improvement_component == pytest.approx(expected_points)


def test_driver_array_conversion(
    scorer: Scorer, standard_driver_data: DriverWeekendData
):
    """Test conversion between DriverWeekendData and structured array."""
    arr = scorer.convert_to_driver_array(standard_driver_data)

    assert arr["format"] == RaceFormat.STANDARD.value
    assert arr["qualifying"] == standard_driver_data.positions.qualifying
    assert arr["race"] == standard_driver_data.positions.race
    assert arr["sprint"] == -1  # No sprint in standard race
    assert arr["completion"] == standard_driver_data.completion_percentage
    assert arr["rolling_avg"] == standard_driver_data.rolling_average
    assert arr["teammate"] == standard_driver_data.teammate_position


def test_constructor_array_conversion(
    scorer: Scorer, standard_constructor_data: ConstructorWeekendData
):
    """Test conversion between ConstructorWeekendData and structured array."""
    arr = scorer.convert_to_constructor_array(standard_constructor_data)

    assert arr["format"] == RaceFormat.STANDARD.value
    assert arr["qualifying1"] == standard_constructor_data.positions.driver1_qualifying
    assert arr["qualifying2"] == standard_constructor_data.positions.driver2_qualifying
    assert arr["race1"] == standard_constructor_data.positions.driver1_race
    assert arr["race2"] == standard_constructor_data.positions.driver2_race
