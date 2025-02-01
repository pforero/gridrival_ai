"""Tests for expected points calculator."""

import pytest

from gridrival_ai.optimization.expected_points import ExpectedPointsCalculator
from gridrival_ai.probabilities.driver_distributions import DriverDistribution
from gridrival_ai.probabilities.position_distribution import PositionDistributions
from gridrival_ai.probabilities.types import JointProbabilities, SessionProbabilities
from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.utils.driver_stats import DriverStats


@pytest.fixture
def scoring_config():
    """Create simple scoring configuration for testing."""
    return ScoringConfig()


@pytest.fixture
def scorer(scoring_config):
    """Create scorer with simple configuration."""
    return Scorer(scoring_config)


@pytest.fixture
def driver_stats():
    """Create driver stats for testing."""
    return DriverStats(
        rolling_averages={
            "VER": 1.5,  # Dominant
            "LAW": 3.0,  # Strong
            "ALO": 6.0,  # Midfield
            "STR": 8.0,  # Midfield
        }
    )


@pytest.fixture
def create_distribution():
    """Factory for creating position distributions."""

    def _create(
        qualifying: dict[int, float],
        race: dict[int, float],
        completion_prob: float = 1.0,
    ) -> DriverDistribution:
        """Create distribution with specified probabilities."""
        qual_probs = {i: 0.0 for i in range(1, 21)}
        qual_probs.update(qualifying)
        race_probs = {i: 0.0 for i in range(1, 21)}
        race_probs.update(race)

        # Create joint probabilities assuming independence
        joint_probs = {
            (q, r): qualifying.get(q, 0.0) * race.get(r, 0.0)
            for q in range(1, 21)
            for r in range(1, 21)
        }

        return DriverDistribution(
            qualifying=SessionProbabilities(qual_probs),
            race=SessionProbabilities(race_probs),
            joint_qual_race=JointProbabilities(
                session1="qualifying",
                session2="race",
                probabilities=joint_probs,
            ),
            completion_prob=completion_prob,
        )

    return _create


@pytest.fixture
def position_distributions(create_distribution):
    """Create position distributions for testing."""
    distributions = {
        # VER: Always P1 in qualifying and race
        "VER": create_distribution(
            qualifying={1: 1.0},
            race={1: 1.0},
            completion_prob=1.0,
        ),
        # LAW: P3 in qualifying, P2 in race
        "LAW": create_distribution(
            qualifying={3: 1.0},
            race={2: 1.0},
            completion_prob=0.95,
        ),
        # ALO: P2 in qualifying, P3 in race
        "ALO": create_distribution(
            qualifying={2: 1.0},
            race={3: 1.0},
            completion_prob=0.90,
        ),
        # STR: P4 in qualifying, P4 in race
        "STR": create_distribution(
            qualifying={4: 1.0},
            race={4: 1.0},
            completion_prob=0.90,
        ),
    }
    return PositionDistributions(driver_distributions=distributions)


@pytest.fixture
def calculator(position_distributions, scorer, driver_stats):
    """Create calculator instance for testing."""
    return ExpectedPointsCalculator(
        distributions=position_distributions,
        scorer=scorer,
        driver_stats=driver_stats,
    )


def test_base_points_calculation(calculator):
    """Test calculation of qualifying and race points."""
    points = calculator.calculate_driver_points("VER")
    assert points["qualifying"] == 50.0  # P1 in qualifying
    assert points["race"] == 100.0  # P1 in race
    assert "sprint" not in points  # Standard format


def test_overtake_points(calculator):
    """Test calculation of overtake points."""
    # LAW: P3 -> P2 gains one position
    points = calculator.calculate_driver_points("LAW")
    assert points["overtake"] == 3.0  # One position * 3 points

    # ALO: P2 -> P3 loses one position
    points = calculator.calculate_driver_points("ALO")
    assert points["overtake"] == 0.0  # No points for losing positions


def test_teammate_points(calculator):
    """Test calculation of teammate points."""
    # VER and LAW are teammates
    points_ver = calculator.calculate_driver_points("VER")
    points_law = calculator.calculate_driver_points("LAW")

    # VER finishes P1, LAW P2
    assert points_ver["teammate"] > 0.0  # Points for beating teammate
    assert points_law["teammate"] == 0.0  # No points when behind teammate


def test_completion_points(calculator):
    """Test calculation of completion points."""
    # VER: 100% completion probability
    points = calculator.calculate_driver_points("VER")
    assert points["completion"] == 12.0  # 4 stages * 3 points

    # LAW: 95% completion probability
    points = calculator.calculate_driver_points("LAW")
    assert 10.0 < points["completion"] < 12.0

    # ALO: 90% completion probability
    points = calculator.calculate_driver_points("ALO")
    assert 8.0 < points["completion"] < 10.0


def test_improvement_points(calculator):
    """Test calculation of improvement vs rolling average."""
    # VER: Rolling avg 1.5, finishes P1 (+0.5 positions)
    points = calculator.calculate_driver_points("VER")
    assert points["improvement"] > 0.0

    # LAW: Rolling avg 3.0, finishes P2 (+1 position)
    points = calculator.calculate_driver_points("LAW")
    assert points["improvement"] > 0.0
