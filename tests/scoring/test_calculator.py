"""Tests for the ScoringCalculator class."""

import pytest

from gridrival_ai.probabilities.distributions import (
    JointDistribution,
    PositionDistribution,
)
from gridrival_ai.scoring.calculator import (
    DriverPointsBreakdown,
    ScoringCalculator,
)


@pytest.fixture
def calculator():
    """Create a ScoringCalculator."""
    return ScoringCalculator()


class TestScoringCalculator:
    """Test suite for ScoringCalculator class."""

    def test_initialization(self):
        """Test initialization."""
        calculator = ScoringCalculator()
        assert calculator is not None

    def test_calculate_driver_points(self, calculator):
        """Test calculating points for a driver."""
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

        # Verify structure
        assert isinstance(points, DriverPointsBreakdown)

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

    def test_calculate_constructor_points(self, calculator):
        """Test calculating points for a constructor."""
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

    def test_expected_driver_points(self, calculator):
        """Test calculating expected points from distributions."""
        # Create distributions
        qual_dist = PositionDistribution({1: 0.6, 2: 0.4})
        race_dist = PositionDistribution({1: 0.7, 2: 0.3})
        teammate_dist = PositionDistribution({1: 0.0, 2: 0.5, 3: 0.5})

        # Calculate expected points
        points = calculator.expected_driver_points(
            qual_dist=qual_dist,
            race_dist=race_dist,
            rolling_avg=3,
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

    def test_expected_constructor_points(self, calculator):
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

    def test_sprint_race(self, calculator):
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

    def test_joint_distribution_overtakes(self, calculator):
        """Test overtake calculation with explicit joint distribution."""
        # Create marginal distributions
        qual_dist = PositionDistribution({1: 0.6, 2: 0.4})
        race_dist = PositionDistribution({1: 0.7, 2: 0.3})

        # Create joint distribution with positive correlation
        joint_dist = JointDistribution(
            {
                (1, 1): 0.5,  # Both P1
                (1, 2): 0.1,  # Qualify P1, finish P2
                (2, 1): 0.2,  # Qualify P2, finish P1
                (2, 2): 0.2,  # Both P2
            }
        )

        # Calculate with joint distribution
        points_with_joint = calculator.expected_driver_points(
            qual_dist=qual_dist,
            race_dist=race_dist,
            rolling_avg=2.0,
            teammate_dist=PositionDistribution({1: 0.0, 2: 0.0, 3: 1.0}),
            joint_qual_race=joint_dist,
        )

        # Expected overtake points: only (2,1) contributes = 0.2 * 3 * 1 = 0.6
        assert points_with_joint.overtake == pytest.approx(0.6)

        # Calculate without joint (assuming independence)
        points_independent = calculator.expected_driver_points(
            qual_dist=qual_dist,
            race_dist=race_dist,
            rolling_avg=2.0,
            teammate_dist=PositionDistribution({1: 0.0, 2: 0.0, 3: 1.0}),
        )

        # Independent overtake points should be different
        # P(Q1,R2) = 0.6*0.3 = 0.18, P(Q2,R1) = 0.4*0.7 = 0.28
        # Expected: 0.28 * 3 * 1 = 0.84
        assert points_independent.overtake == pytest.approx(0.84)
        assert points_with_joint.overtake != points_independent.overtake

    def test_partial_completion(self, calculator):
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

    def test_expected_completion_points(self, calculator):
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

    def test_driver_points_breakdown(self):
        """Test DriverPointsBreakdown class."""
        # Create a breakdown
        breakdown = DriverPointsBreakdown(
            qualifying=50,
            race=100,
            sprint=20,
            overtake=6,
            improvement=4,
            teammate=2,
            completion=12,
        )

        # Check individual components
        assert breakdown.qualifying == 50
        assert breakdown.race == 100
        assert breakdown.sprint == 20
        assert breakdown.overtake == 6
        assert breakdown.improvement == 4
        assert breakdown.teammate == 2
        assert breakdown.completion == 12

        # Check total
        assert breakdown.total == 194
