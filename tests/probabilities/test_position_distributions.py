"""Tests for position probability distributions container."""

import pytest

from gridrival_ai.probabilities.driver_distributions import DriverDistribution
from gridrival_ai.probabilities.position_distribution import PositionDistributions
from gridrival_ai.probabilities.types import SessionProbabilities


@pytest.fixture
def simple_probabilities():
    """Create simple probability distribution for testing."""
    # Create a simple uniform distribution
    probs = {i: 1 / 20 for i in range(1, 21)}
    return SessionProbabilities(probabilities=probs)


@pytest.fixture
def driver_distribution(simple_probabilities):
    """Create a simple driver distribution for testing."""
    return DriverDistribution(race=simple_probabilities)


@pytest.fixture
def position_distributions(driver_distribution):
    """Create position distributions with two drivers."""
    driver_dists = {
        "VER": driver_distribution,
        "PER": driver_distribution,
    }
    return PositionDistributions(driver_distributions=driver_dists)


def test_basic_initialization(position_distributions):
    """Test basic initialization and access."""
    assert len(position_distributions.driver_distributions) == 2
    assert "VER" in position_distributions.driver_distributions
    assert "PER" in position_distributions.driver_distributions


def test_get_session_probabilities(position_distributions, simple_probabilities):
    """Test getting probabilities for different sessions."""
    # Test qualifying probabilities
    qual_probs = position_distributions.get_session_probabilities("VER", "qualifying")
    assert qual_probs == simple_probabilities

    # Test race probabilities
    race_probs = position_distributions.get_session_probabilities("VER", "race")
    assert race_probs == simple_probabilities

    # Test sprint probabilities
    sprint_probs = position_distributions.get_session_probabilities("VER", "sprint")
    assert sprint_probs == simple_probabilities


def test_get_driver_session_correlation(position_distributions):
    """Test getting correlation between a driver's positions across sessions."""
    joint_probs = position_distributions.get_driver_session_correlation(
        "VER", "qualifying", "race"
    )

    # Test a few values for the independent case
    expected_prob = 1 / 20 * 1 / 20  # Independent uniform distributions
    assert joint_probs[(1, 1)] == pytest.approx(expected_prob)
    assert joint_probs[(5, 10)] == pytest.approx(expected_prob)


def test_invalid_session_name(position_distributions):
    """Test that invalid session names raise ValueError."""
    with pytest.raises(ValueError, match="Invalid session name"):
        position_distributions.get_session_probabilities("VER", "invalid_session")


def test_invalid_driver_id(position_distributions):
    """Test that invalid driver IDs raise KeyError."""
    with pytest.raises(KeyError):
        position_distributions.get_session_probabilities("HAM", "qualifying")


def test_get_completion_probability(simple_probabilities, position_distributions):
    """Test getting completion probability for a driver."""
    # Default completion probability should be 0.95
    assert position_distributions.get_completion_probability("VER") == 0.95

    # Test with custom completion probability
    driver_dist = DriverDistribution(race=simple_probabilities, completion_prob=0.8)
    custom_distributions = PositionDistributions({"VER": driver_dist})
    assert custom_distributions.get_completion_probability("VER") == 0.8

    # Test invalid driver ID
    with pytest.raises(KeyError):
        position_distributions.get_completion_probability("HAM")


def test_get_available_sessions(position_distributions):
    """Test getting available session names."""
    sessions = position_distributions.get_available_sessions()
    assert sessions == {"qualifying", "race", "sprint"}
