"""
Tests for the main ScoringCalculator module.

This module contains tests for the ScoringCalculator class,
which orchestrates the calculation of expected fantasy points
for drivers and constructors.
"""

import pytest

from gridrival_ai.probabilities.distributions import PositionDistribution
from gridrival_ai.scoring.calculator import ScoringCalculator


@pytest.fixture
def calculator():
    """Create a ScoringCalculator instance."""
    return ScoringCalculator()


def test_calculate_driver_points(calculator):
    """Test calculating driver points."""
    # Test a perfect race (P1 in all sessions)
    points = calculator.calculate_driver_points(
        qualifying_pos=1,
        race_pos=1,
        rolling_avg=2.0,
        teammate_pos=2,
        sprint_pos=1,
        race_format="SPRINT",
        completion_pct=1.0,
    )

    # Verify components
    assert points.qualifying == 50  # P1 in qualifying
    assert points.race == 100  # P1 in race
    assert points.sprint == 20  # P1 in sprint
    assert points.overtake == 0  # No positions gained (P1 -> P1)
    assert points.improvement == 0  # 1 position ahead of average
    assert points.teammate == 2  # Beat teammate by 1 position
    assert points.completion == 12  # Full completion (4 stages * 3 points)

    # Verify total
    assert points.total == 50 + 100 + 20 + 0 + 0 + 2 + 12


def test_calculate_constructor_points(calculator):
    """Test calculating constructor points."""
    points = calculator.calculate_constructor_points(
        driver1_qualifying=1,
        driver1_race=1,
        driver2_qualifying=2,
        driver2_race=2,
    )

    # Verify structure
    assert isinstance(points, dict)
    assert "qualifying" in points
    assert "race" in points

    # Verify components
    assert points["qualifying"] == 30 + 29  # P1 (30) + P2 (29)
    assert points["race"] == 60 + 58  # P1 (60) + P2 (58)


def test_expected_driver_points(calculator):
    """Test calculating expected points from distributions."""
    # Create distributions
    qual_dist = PositionDistribution({1: 0.6, 2: 0.4})
    race_dist = PositionDistribution({1: 0.7, 2: 0.3})
    teammate_dist = PositionDistribution({1: 0.0, 2: 0.5, 3: 0.5})

    # Calculate expected points
    points = calculator.expected_driver_points(
        qual_dist=qual_dist,
        race_dist=race_dist,
        rolling_avg=3.0,
        teammate_dist=teammate_dist,
        completion_prob=1.0,
        race_format="STANDARD",
    )

    # Expected qualifying points: 0.6*50 + 0.4*48 = 30 + 19.2 = 49.2
    assert points.qualifying == pytest.approx(49.2)

    # Expected race points: 0.7*100 + 0.3*97 = 70 + 29.1 = 99.1
    assert points.race == pytest.approx(99.1)

    # Check other components
    assert points.improvement > 0
    assert points.teammate > 0
    assert points.completion == 12  # Full completion probability


def test_expected_constructor_points(calculator):
    """Test calculating expected constructor points from distributions."""
    # Create distributions
    d1_qual = PositionDistribution({1: 0.7, 2: 0.3})
    d1_race = PositionDistribution({1: 0.8, 2: 0.2})
    d2_qual = PositionDistribution({1: 0.0, 2: 0.6, 3: 0.4})
    d2_race = PositionDistribution({1: 0.0, 2: 0.5, 3: 0.5})

    # Calculate expected points
    points = calculator.expected_constructor_points(
        driver1_qual_dist=d1_qual,
        driver1_race_dist=d1_race,
        driver2_qual_dist=d2_qual,
        driver2_race_dist=d2_race,
    )

    # Expected qualifying: 0.7*30 + 0.3*29 + 0.6*29 + 0.4*28 = 21
    # + 8.7 + 17.4 + 11.2 = 58.3
    assert points["qualifying"] == pytest.approx(58.3)

    # Expected race: 0.8*60 + 0.2*58 + 0.5*58 + 0.5*56 = 48
    # + 11.6 + 29 + 28 = 116.6
    assert points["race"] == pytest.approx(116.6)


def test_sprint_race(calculator):
    """Test calculation with sprint race format."""
    # Standard race (no sprint)
    std_points = calculator.calculate_driver_points(
        qualifying_pos=1,
        race_pos=1,
        rolling_avg=2.0,
        teammate_pos=2,
        race_format="STANDARD",
        completion_pct=1.0,
    )

    # Sprint race
    sprint_points = calculator.calculate_driver_points(
        qualifying_pos=1,
        race_pos=1,
        sprint_pos=1,
        rolling_avg=2.0,
        teammate_pos=2,
        race_format="SPRINT",
        completion_pct=1.0,
    )

    # Sprint should have more points
    assert sprint_points.total > std_points.total
    assert sprint_points.sprint == 20  # P1 in sprint


def test_partial_completion(calculator):
    """Test scoring with partial race completion."""
    # Test different completion percentages
    stages = [0.0, 0.2, 0.3, 0.6, 0.8, 1.0]
    expected_completion_points = [0, 0, 3, 6, 9, 12]

    for completion, expected in zip(stages, expected_completion_points):
        points = calculator.calculate_driver_points(
            qualifying_pos=1,
            race_pos=1,
            rolling_avg=1.0,
            teammate_pos=2,
            completion_pct=completion,
        )
        assert points.completion == expected


def test_expected_completion_points(calculator):
    """Test expected completion points with different probabilities."""
    # Test with different completion probabilities
    probs = [0.0, 0.5, 1.0]

    for prob in probs:
        points = calculator.expected_driver_points(
            qual_dist=PositionDistribution({1: 1.0}),
            race_dist=PositionDistribution({1: 1.0}),
            rolling_avg=1.0,
            teammate_dist=PositionDistribution({1: 0.0, 2: 1.0}),
            completion_prob=prob,
        )

        # Full completion is 12 points
        # With prob=0.5, expected value should be around 6-7 points
        # (not exactly 6 because of stage distribution)
        if prob == 0.0:
            assert points.completion < 6
        elif prob == 0.5:
            assert 6 <= points.completion <= 9
        else:  # prob == 1.0
            assert points.completion == 12
