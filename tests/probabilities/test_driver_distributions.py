"""Tests for driver position probability distributions."""

import pytest

from gridrival_ai.probabilities.driver_distributions import (
    DistributionError,
    DriverDistribution,
)


@pytest.fixture
def valid_race_probs():
    """Fixture providing valid race probabilities for 20 positions."""
    # Simple geometric distribution
    probs = {i: 0.5**i for i in range(1, 21)}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def test_basic_race_distribution(valid_race_probs):
    """Test basic initialization with only race probabilities."""
    dist = DriverDistribution(race=valid_race_probs)
    assert dist.race == valid_race_probs
    assert dist.qualifying == valid_race_probs
    assert dist.sprint == valid_race_probs


def test_full_distribution(valid_race_probs):
    """Test initialization with all distributions specified."""
    # Shift probabilities for qualifying and sprint to create different distributions
    qual_probs = {k: valid_race_probs[min(k + 1, 20)] for k in range(1, 21)}
    sprint_probs = {k: valid_race_probs[min(k + 2, 20)] for k in range(1, 21)}

    # Normalize the shifted distributions
    qual_total = sum(qual_probs.values())
    sprint_total = sum(sprint_probs.values())
    qual_probs = {k: v / qual_total for k, v in qual_probs.items()}
    sprint_probs = {k: v / sprint_total for k, v in sprint_probs.items()}

    # Create joint distribution assuming some correlation
    joint_probs = {
        (q, r): qual_probs[q] * valid_race_probs[r] * 1.1
        for q in range(1, 21)
        for r in range(1, 21)
    }
    # Normalize joint probabilities
    total = sum(joint_probs.values())
    joint_probs = {k: v / total for k, v in joint_probs.items()}

    dist = DriverDistribution(
        race=valid_race_probs,
        qualifying=qual_probs,
        sprint=sprint_probs,
        joint_qual_race=joint_probs,
    )

    assert dist.race == valid_race_probs
    assert dist.qualifying == qual_probs
    assert dist.sprint == sprint_probs


def test_invalid_probabilities(valid_race_probs):
    """Test that invalid probabilities raise DistributionError."""
    # Test for invalid probability value
    invalid_probs = valid_race_probs.copy()
    invalid_probs[1] = 1.2  # Invalid probability > 1
    with pytest.raises(DistributionError, match="must be between 0 and 1"):
        DriverDistribution(race=invalid_probs)

    # Test for sum != 1
    invalid_probs = valid_race_probs.copy()
    invalid_probs = {k: v * 0.9 for k, v in invalid_probs.items()}  # Sum < 1
    with pytest.raises(DistributionError, match="must sum to 1.0"):
        DriverDistribution(race=invalid_probs)


def test_invalid_positions():
    """Test that invalid positions raise DistributionError."""
    with pytest.raises(DistributionError, match="Invalid positions"):
        DriverDistribution(race={0: 0.5, 1: 0.5})  # Position 0 invalid

    with pytest.raises(DistributionError, match="Invalid positions"):
        DriverDistribution(race={1: 0.5, 21: 0.5})  # Position > MAX_POSITION


def test_automatic_joint_distribution(valid_race_probs):
    """Test that joint distribution is created correctly when not provided."""
    dist = DriverDistribution(race=valid_race_probs)

    # Check if joint distribution follows independence assumption
    for q in range(1, 21):
        for r in range(1, 21):
            expected = valid_race_probs[q] * valid_race_probs[r]
            assert dist.joint_qual_race[(q, r)] == pytest.approx(expected)


def test_missing_positions(valid_race_probs):
    """Test that distributions must include all positions 1-20."""
    incomplete_probs = {k: v for k, v in valid_race_probs.items() if k <= 10}

    with pytest.raises(DistributionError):
        DriverDistribution(race=incomplete_probs)
