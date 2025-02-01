"""Tests for expected points calculator."""

import pytest

from gridrival_ai.optimization.expected_points import ExpectedPointsCalculator
from gridrival_ai.probabilities.driver_distributions import DriverDistribution
from gridrival_ai.probabilities.position_distribution import PositionDistributions
from gridrival_ai.probabilities.types import JointProbabilities, SessionProbabilities
from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.calculator import Scorer
from gridrival_ai.scoring.types import RaceFormat
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
        sprint: dict[int, float],
        completion_prob: float = 1.0,
    ) -> DriverDistribution:
        """Create distribution with specified probabilities."""
        qual_probs = {i: 0.0 for i in range(1, 21)}
        qual_probs.update(qualifying)
        race_probs = {i: 0.0 for i in range(1, 21)}
        race_probs.update(race)
        sprint_probs = {i: 0.0 for i in range(1, 21)}
        sprint_probs.update(sprint)

        # Create joint probabilities assuming independence
        joint_probs = {
            (q, r): qualifying.get(q, 0.0) * race.get(r, 0.0)
            for q in range(1, 21)
            for r in range(1, 21)
        }

        return DriverDistribution(
            qualifying=SessionProbabilities(qual_probs),
            race=SessionProbabilities(race_probs),
            sprint=SessionProbabilities(sprint_probs),
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
        # VER: Always P1 in qualifying, race and sprint
        "VER": create_distribution(
            qualifying={1: 1.0},
            race={1: 1.0},
            sprint={1: 1.0},  # Same as race
            completion_prob=1.0,
        ),
        # LAW: P3 in qualifying, P2 in race/sprint
        "LAW": create_distribution(
            qualifying={3: 1.0},
            race={2: 1.0},
            sprint={2: 1.0},  # Same as race
            completion_prob=0.95,
        ),
        # ALO: 50% P2 and 50% P3 in qualifying, 50% P3 and 50% P4 in race/sprint
        "ALO": create_distribution(
            qualifying={2: 0.5, 3: 0.5},
            race={3: 0.5, 4: 0.5},
            sprint={3: 0.5, 4: 0.5},  # Same as race
            completion_prob=0.90,
        ),
        # STR: 50% P4 and 50% P5 in qualifying, 50% P4 and 50% P5 in race
        # Sprint: 50% P9 and 50% P10 (no points)
        "STR": create_distribution(
            qualifying={4: 0.5, 5: 0.5},
            race={4: 0.5, 5: 0.5},
            sprint={9: 0.5, 10: 0.5},  # Outside points positions
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
    # VER: Deterministic P1 in both sessions
    points = calculator.calculate_driver_points("VER", format=RaceFormat.STANDARD)
    assert points["qualifying"] == 50.0  # P1 in qualifying
    assert points["race"] == 100.0  # P1 in race
    assert "sprint" not in points  # Standard format

    # Test sprint format
    points = calculator.calculate_driver_points("VER", format=RaceFormat.SPRINT)
    assert points["qualifying"] == 50.0  # P1 in qualifying
    assert points["race"] == 100.0  # P1 in race
    assert points["sprint"] == 8.0  # P1 in sprint (8 points)

    # LAW: P3 qualifying, P2 race/sprint
    points = calculator.calculate_driver_points("LAW", format=RaceFormat.SPRINT)
    assert points["qualifying"] == 46.0  # P3 in qualifying
    assert points["race"] == 97.0  # P2 in race
    assert points["sprint"] == 7.0  # P2 in sprint (7 points)

    # ALO: Mixed probabilities
    points = calculator.calculate_driver_points("ALO", format=RaceFormat.SPRINT)
    # Qualifying: 0.5 * P2(48) + 0.5 * P3(46) = 47.0
    assert points["qualifying"] == 47.0
    # Race: 0.5 * P3(94) + 0.5 * P4(91) = 92.5
    assert points["race"] == 92.5
    # Sprint: 0.5 * P3(6) + 0.5 * P4(5) = 5.5
    assert points["sprint"] == 5.5

    # STR: Mixed probabilities
    points = calculator.calculate_driver_points("STR", format=RaceFormat.SPRINT)
    # Qualifying: 0.5 * P4(44) + 0.5 * P5(42) = 43.0
    assert points["qualifying"] == 43.0
    # Race: 0.5 * P4(91) + 0.5 * P5(88) = 89.5
    assert points["race"] == 89.5
    # Sprint: P9/P10 = 0 points (only top 8 score)
    assert points["sprint"] == 0.0


def test_overtake_points(calculator):
    """Test calculation of overtake points."""
    # VER: P1 -> P1 (no positions gained)
    points = calculator.calculate_driver_points("VER")
    assert points["overtake"] == 0.0

    # LAW: P3 -> P2 (gains one position)
    points = calculator.calculate_driver_points("LAW")
    assert points["overtake"] == 3.0  # One position * 3 points

    # ALO: Mixed probabilities
    points = calculator.calculate_driver_points("ALO")
    # From P2 (50%): to P3 (-1) or P4 (-2) = 0 points
    # From P3 (50%): to P3 (0) or P4 (-1) = 0 points
    assert points["overtake"] == 0.0

    # STR: Mixed probabilities
    points = calculator.calculate_driver_points("STR")
    # From P4 (50%): to P4 (0) or P5 (-1) = 0 points
    # From P5 (50%): to P4 (+1) or P5 (0)
    # Expected: 0.5 * 0.5 * 3 = 0.75 points (only P5->P4 gains points)
    assert points["overtake"] == 0.75


def test_teammate_points(calculator):
    """Test calculation of teammate points.

    Tests the teammate points calculation using the joint distribution that
    enforces the constraint that teammates cannot finish in the same position.
    """
    # VER vs LAW
    points_ver = calculator.calculate_driver_points("VER")
    points_law = calculator.calculate_driver_points("LAW")
    # VER: Always P1, LAW: Always P2
    # Since they can't finish in same position, this is deterministic
    assert points_ver["teammate"] == 2.0  # 1-3 positions bracket
    assert points_law["teammate"] == 0.0  # Behind teammate

    # ALO vs STR
    points_alo = calculator.calculate_driver_points("ALO")
    points_str = calculator.calculate_driver_points("STR")
    # ALO: 50% P3, 50% P4
    # STR: 50% P4, 50% P5
    # After enforcing no-same-position constraint and renormalizing:
    # P(ALO=3, STR=4) = 0.5 * 0.5 / 0.75 = 1/3  -> +2 points
    # P(ALO=3, STR=5) = 0.5 * 0.5 / 0.75 = 1/3  -> +2 points
    # P(ALO=4, STR=5) = 0.5 * 0.5 / 0.75 = 1/3  -> +2 points
    # Expected points = 2.0
    assert points_alo["teammate"] == 2.0
    assert points_str["teammate"] == 0.0  # Behind teammate


def test_completion_points(calculator):
    """Test calculation of completion points.

    The test verifies the new completion points calculation that assumes DNFs
    occur uniformly across race laps. For a driver with completion probability p,
    the (1-p) probability of DNF is distributed uniformly across race distance.

    Using thresholds [0.25, 0.5, 0.75, 0.9]:
    - DNF before 0.25: 0 points (25% of DNF cases)
    - DNF between 0.25-0.5: 3 points (25% of DNF cases)
    - DNF between 0.5-0.75: 6 points (25% of DNF cases)
    - DNF between 0.75-0.9: 9 points (15% of DNF cases)
    - DNF between 0.9-1.0: 12 points (10% of DNF cases)
    - Complete race (no DNF): 12 points
    """
    # VER: 100% completion probability
    points = calculator.calculate_driver_points("VER")
    # All stages completed with certainty
    assert points["completion"] == 12.0  # 4 stages * 3 points

    # LAW: 95% completion probability
    points = calculator.calculate_driver_points("LAW")
    # 95% chance of completing race: 0.95 * 12 = 11.4
    # 5% chance of DNF distributed across thresholds:
    # - 0.25 * 0.05 * 0 = 0.0 (DNF before 0.25)
    # - 0.25 * 0.05 * 3 = 0.0375 (DNF between 0.25-0.5)
    # - 0.25 * 0.05 * 6 = 0.075 (DNF between 0.5-0.75)
    # - 0.15 * 0.05 * 9 = 0.0675 (DNF between 0.75-0.9)
    # - 0.10 * 0.05 * 12 = 0.06 (DNF between 0.9-1.0)
    # Total = 11.4 + 0.0 + 0.0375 + 0.075 + 0.0675 + 0.06 = 11.64
    expected_law = 11.64
    assert abs(points["completion"] - expected_law) < 0.01

    # ALO: 90% completion probability
    points = calculator.calculate_driver_points("ALO")
    # 90% chance of completing race: 0.9 * 12 = 10.8
    # 10% chance of DNF distributed across thresholds:
    # - 0.25 * 0.10 * 0 = 0.0 (DNF before 0.25)
    # - 0.25 * 0.10 * 3 = 0.075 (DNF between 0.25-0.5)
    # - 0.25 * 0.10 * 6 = 0.15 (DNF between 0.5-0.75)
    # - 0.15 * 0.10 * 9 = 0.135 (DNF between 0.75-0.9)
    # - 0.10 * 0.10 * 12 = 0.12 (DNF between 0.9-1.0)
    # Total = 10.8 + 0.0 + 0.075 + 0.15 + 0.135 + 0.12 = 11.28
    expected_alo = 11.28
    assert abs(points["completion"] - expected_alo) < 0.01

    # STR: 90% completion probability (same as ALO)
    points = calculator.calculate_driver_points("STR")
    assert abs(points["completion"] - expected_alo) < 0.01  # Should match ALO


def test_improvement_points(calculator):
    """Test calculation of improvement vs rolling average."""
    # VER: Rolling avg 1.5, always P1
    points = calculator.calculate_driver_points("VER")
    # Gains 0.5 positions = 2 points (1 position bracket)
    assert points["improvement"] == 2.0

    # LAW: Rolling avg 3.0, finishes P2
    points = calculator.calculate_driver_points("LAW")
    # Gains 1 position = 2 points (1 position bracket)
    assert points["improvement"] == 2.0

    # ALO: Rolling avg 6.0
    points = calculator.calculate_driver_points("ALO")
    # 50% P3 (+3 positions = 6 points)
    # 50% P4 (+2 positions = 4 points)
    # Expected: 0.5 * 6 + 0.5 * 4 = 5.0
    assert points["improvement"] == 5.0

    # STR: Rolling avg 8.0
    points = calculator.calculate_driver_points("STR")
    # 50% P4 (+4 positions = 9 points)
    # 50% P5 (+3 positions = 6 points)
    # Expected: 0.5 * 9 + 0.5 * 6 = 7.5
    assert points["improvement"] == 7.5
