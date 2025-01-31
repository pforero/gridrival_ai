"""Tests for probability distribution types."""

import pytest

from gridrival_ai.probabilities.types import (
    DistributionError,
    JointProbabilities,
    SessionProbabilities,
)


def test_valid_session_probabilities():
    """Test valid session probability distribution."""
    probs = {1: 0.6, 2: 0.4}
    dist = SessionProbabilities(probabilities=probs)
    assert dist.probabilities == probs


def test_valid_joint_probabilities():
    """Test valid joint probability distribution."""
    probs = {(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
    dist = JointProbabilities(
        session1="qualifying", session2="race", probabilities=probs
    )
    assert dist.probabilities == probs
    assert dist.session1 == "qualifying"
    assert dist.session2 == "race"


def test_session_probabilities_validation():
    """Test session probability validation."""
    # Invalid sum
    with pytest.raises(DistributionError, match="must sum to 1.0"):
        SessionProbabilities(probabilities={1: 0.6, 2: 0.6})

    # Invalid probability value
    with pytest.raises(DistributionError, match="must be between 0 and 1"):
        SessionProbabilities(probabilities={1: 1.2, 2: -0.2})

    # Invalid position
    with pytest.raises(DistributionError, match="Invalid positions"):
        SessionProbabilities(probabilities={0: 0.5, 1: 0.5})


def test_joint_probabilities_validation():
    """Test joint probability validation."""
    # Invalid sum
    with pytest.raises(DistributionError, match="must sum to 1.0"):
        JointProbabilities(
            session1="qualifying",
            session2="race",
            probabilities={(1, 1): 0.6, (1, 2): 0.6},
        )

    # Invalid position pair
    with pytest.raises(DistributionError, match="Invalid positions"):
        JointProbabilities(
            session1="qualifying",
            session2="race",
            probabilities={(0, 1): 0.5, (1, 1): 0.5},
        )
