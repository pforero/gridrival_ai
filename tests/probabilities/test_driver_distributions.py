"""Tests for driver position probability distributions."""

import pytest

from gridrival_ai.probabilities.driver_distributions import DriverDistribution
from gridrival_ai.probabilities.types import (
    DistributionError,
    JointProbabilities,
    SessionProbabilities,
)


@pytest.fixture
def valid_race_probs():
    """Fixture providing valid race probabilities for 20 positions."""
    # Simple geometric distribution
    probs = {i: 0.5**i for i in range(1, 21)}
    total = sum(probs.values())
    return SessionProbabilities(probabilities={k: v / total for k, v in probs.items()})


def test_basic_race_distribution(valid_race_probs):
    """Test basic initialization with only race probabilities."""
    dist = DriverDistribution(race=valid_race_probs)
    assert dist.race == valid_race_probs
    assert dist.qualifying == valid_race_probs
    assert dist.sprint == valid_race_probs
    assert dist.completion_prob == 0.95


def test_full_distribution(valid_race_probs):
    """Test initialization with all distributions specified."""
    # Shift probabilities for qualifying and sprint to create different distributions
    qual_probs = {
        k: valid_race_probs.probabilities[min(k + 1, 20)] for k in range(1, 21)
    }
    sprint_probs = {
        k: valid_race_probs.probabilities[min(k + 2, 20)] for k in range(1, 21)
    }

    # Normalize the shifted distributions
    qual_total = sum(qual_probs.values())
    sprint_total = sum(sprint_probs.values())
    qual_probs = SessionProbabilities(
        probabilities={k: v / qual_total for k, v in qual_probs.items()}
    )
    sprint_probs = SessionProbabilities(
        probabilities={k: v / sprint_total for k, v in sprint_probs.items()}
    )

    # Create joint distribution assuming some correlation
    joint_probs = {
        (q, r): qual_probs.probabilities[q] * valid_race_probs.probabilities[r] * 1.1
        for q in range(1, 21)
        for r in range(1, 21)
    }
    # Normalize joint probabilities
    total = sum(joint_probs.values())
    joint_probs = JointProbabilities(
        session1="qualifying",
        session2="race",
        probabilities={k: v / total for k, v in joint_probs.items()},
    )

    dist = DriverDistribution(
        race=valid_race_probs,
        qualifying=qual_probs,
        sprint=sprint_probs,
        joint_qual_race=joint_probs,
    )

    assert dist.race == valid_race_probs
    assert dist.qualifying == qual_probs
    assert dist.sprint == sprint_probs
    assert dist.joint_qual_race == joint_probs


def test_invalid_probabilities(valid_race_probs):
    """Test that invalid probabilities raise DistributionError."""
    # Test for invalid completion probability
    with pytest.raises(
        DistributionError, match="completion_prob must be between 0 and 1"
    ):
        DriverDistribution(race=valid_race_probs, completion_prob=1.2)

    with pytest.raises(
        DistributionError, match="completion_prob must be between 0 and 1"
    ):
        DriverDistribution(race=valid_race_probs, completion_prob=-0.1)

    # Test for invalid probability value
    invalid_probs = valid_race_probs.probabilities.copy()
    invalid_probs[1] = 1.2  # Invalid probability > 1
    with pytest.raises(DistributionError, match="must be between 0 and 1"):
        DriverDistribution(race=SessionProbabilities(probabilities=invalid_probs))

    # Test for sum != 1
    invalid_probs = valid_race_probs.probabilities.copy()
    invalid_probs = {k: v * 0.9 for k, v in invalid_probs.items()}  # Sum < 1
    with pytest.raises(DistributionError, match="must sum to 1.0"):
        DriverDistribution(race=SessionProbabilities(probabilities=invalid_probs))


def test_invalid_positions():
    """Test that invalid positions raise DistributionError."""
    with pytest.raises(DistributionError, match="Invalid positions"):
        DriverDistribution(race=SessionProbabilities(probabilities={0: 0.5, 1: 0.5}))

    with pytest.raises(DistributionError, match="Invalid positions"):
        DriverDistribution(race=SessionProbabilities(probabilities={1: 0.5, 21: 0.5}))


def test_automatic_joint_distribution(valid_race_probs):
    """Test that joint distribution is created correctly when not provided."""
    dist = DriverDistribution(race=valid_race_probs)

    # Check if joint distribution follows independence assumption
    for q in range(1, 21):
        for r in range(1, 21):
            expected = (
                valid_race_probs.probabilities[q] * valid_race_probs.probabilities[r]
            )
            assert dist.joint_qual_race.probabilities[(q, r)] == pytest.approx(expected)


def test_missing_positions(valid_race_probs):
    """Test that distributions must include all positions 1-20."""
    incomplete_probs = {
        k: v for k, v in valid_race_probs.probabilities.items() if k <= 10
    }

    with pytest.raises(DistributionError):
        DriverDistribution(race=SessionProbabilities(probabilities=incomplete_probs))


def test_custom_completion_prob(valid_race_probs):
    """Test that custom completion probability is set correctly."""
    dist = DriverDistribution(race=valid_race_probs, completion_prob=0.8)
    assert dist.completion_prob == 0.8
