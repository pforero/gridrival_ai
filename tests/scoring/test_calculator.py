"""Tests for the ScoringCalculator class."""

import pytest

from gridrival_ai.probabilities.distributions import (
    JointDistribution,
    PositionDistribution,
)
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.config import ScoringConfig
from gridrival_ai.scoring.types import (
    DriverPointsBreakdown,
    DriverPositions,
    RaceFormat,
)


@pytest.fixture
def simple_config():
    """Create a simple scoring configuration for testing."""
    # Create a simplified scoring config for easier testing
    return ScoringConfig(
        qualifying_points={
            1: 10,
            2: 8,
            3: 6,
            4: 5,
            5: 4,
            6: 3,
            7: 2,
            8: 1,
            **{i: 0 for i in range(9, 21)},
        },
        race_points={
            1: 25,
            2: 18,
            3: 15,
            4: 12,
            5: 10,
            6: 8,
            7: 6,
            8: 4,
            9: 2,
            10: 1,
            **{i: 0 for i in range(11, 21)},
        },
        sprint_points={1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1},
        constructor_qualifying_points={
            1: 5,
            2: 4,
            3: 3,
            4: 2,
            5: 1,
            **{i: 0 for i in range(6, 21)},
        },
        constructor_race_points={
            1: 10,
            2: 8,
            3: 6,
            4: 5,
            5: 4,
            6: 3,
            7: 2,
            8: 1,
            **{i: 0 for i in range(9, 21)},
        },
        improvement_points={1: 2, 2: 4, 3: 6, 4: 8, 5: 10},
        teammate_points={2: 2, 5: 5, 10: 8},
        completion_stage_points=3.0,
        completion_thresholds=[0.25, 0.5, 0.75, 0.9],
        overtake_multiplier=2.0,
    )


@pytest.fixture
def calculator(simple_config):
    """Create a ScoringCalculator with the simple config."""
    return ScoringCalculator(config=simple_config)


class TestScoringCalculator:
    """Test suite for ScoringCalculator class."""

    def test_initialization(self, simple_config):
        """Test initialization with config."""
        calculator = ScoringCalculator(config=simple_config)
        assert calculator.config == simple_config
        assert calculator.engine is not None

        # Test default initialization
        default_calculator = ScoringCalculator()
        assert default_calculator.config is not None
        assert default_calculator.engine is not None

    def test_calculate_driver_points(self, calculator):
        """Test calculating points for a driver."""
        # Test a perfect race (P1 in all sessions)
        points = calculator.calculate_driver_points(
            qualifying_pos=1,
            race_pos=1,
            rolling_avg=2.0,
            teammate_pos=2,
            sprint_pos=1,
            race_format=RaceFormat.SPRINT,
            completion_pct=1.0,
        )

        # Verify structure
        assert isinstance(points, DriverPointsBreakdown)

        # Verify components
        assert points.qualifying == 10  # P1 in qualifying
        assert points.race == 25  # P1 in race
        assert points.sprint == 8  # P1 in sprint
        assert points.overtake == 0  # No positions gained (P1 -> P1)
        assert points.improvement == 2  # 1 position ahead of average
        assert points.teammate == 2  # Beat teammate by 1 position
        assert points.completion == 12  # Full completion (4 stages * 3 points)

        # Verify total
        assert points.total == 59

    def test_calculate_constructor_points(self, calculator):
        """Test calculating points for a constructor."""
        points = calculator.calculate_constructor_points(
            driver1_qualifying=1,
            driver1_race=1,
            driver2_qualifying=2,
            driver2_race=2,
            race_format=RaceFormat.STANDARD,
        )

        # Verify structure
        assert isinstance(points, dict)
        assert "qualifying" in points
        assert "race" in points

        # Verify components
        assert points["qualifying"] == 9  # P1 (5) + P2 (4)
        assert points["race"] == 18  # P1 (10) + P2 (8)

    def test_expected_driver_points(self, calculator):
        """Test calculating expected points from distributions."""
        # Create distributions
        qual_dist = PositionDistribution({1: 0.6, 2: 0.4})
        race_dist = PositionDistribution({1: 0.7, 2: 0.3})
        teammate_dist = PositionDistribution(
            {1: 0.0, 2: 0.5, 3: 0.5}
        )  # Added position 1

        # Calculate expected points
        points = calculator.expected_driver_points(
            qual_dist=qual_dist,
            race_dist=race_dist,
            rolling_avg=2.5,
            teammate_dist=teammate_dist,
            completion_prob=1.0,
            race_format=RaceFormat.STANDARD,
        )

        # Expected qualifying points: 0.6*10 + 0.4*8 = 9.2
        assert points.qualifying == pytest.approx(9.2)

        # Expected race points: 0.7*25 + 0.3*18 = 17.5 + 5.4 = 22.9
        assert points.race == pytest.approx(22.9)

        # Check other components
        assert points.improvement > 0
        assert points.teammate > 0
        assert points.completion == 12  # Full completion probability

    def test_expected_constructor_points(self, calculator):
        """Test calculating expected constructor points from distributions."""
        # Create distributions
        d1_qual = PositionDistribution({1: 0.7, 2: 0.3})
        d1_race = PositionDistribution({1: 0.8, 2: 0.2})
        d2_qual = PositionDistribution({1: 0.0, 2: 0.6, 3: 0.4})  # Added position 1
        d2_race = PositionDistribution({1: 0.0, 2: 0.5, 3: 0.5})  # Added position 1

        # Calculate expected points
        points = calculator.expected_constructor_points(
            driver1_qual_dist=d1_qual,
            driver1_race_dist=d1_race,
            driver2_qual_dist=d2_qual,
            driver2_race_dist=d2_race,
        )

        # Expected qualifying: 0.7*5 + 0.3*4 + 0.6*4 + 0.4*3 = 3.5 + 1.2 + 2.4 + 1.2 = 8.3  # noqa: E501
        assert points["qualifying"] == pytest.approx(8.3)

        # Expected race: 0.8*10 + 0.2*8 + 0.5*8 + 0.5*6 = 8.0 + 1.6 + 4.0 + 3.0 = 16.6
        assert points["race"] == pytest.approx(16.6)

    def test_calculate_individual_components(self, calculator):
        """Test calculating individual scoring components."""
        # Test qualifying points
        assert calculator.calculate_qualifying_points(1) == 10
        assert calculator.calculate_qualifying_points(5) == 4

        # Test race points
        assert calculator.calculate_race_points(1) == 25
        assert calculator.calculate_race_points(10) == 1

        # Test sprint points
        assert calculator.calculate_sprint_points(1) == 8
        assert calculator.calculate_sprint_points(9) == 0  # No points beyond P8

        # Test overtake points
        assert (
            calculator.calculate_overtake_points(10, 5) == 10
        )  # 5 positions * 2 points
        assert (
            calculator.calculate_overtake_points(5, 10) == 0
        )  # No points for losing positions

        # Test improvement points
        assert calculator.calculate_improvement_points(2, 5.0) == 6  # 3 positions ahead
        assert calculator.calculate_improvement_points(10, 5.0) == 0  # No improvement

        # Test teammate points
        assert calculator.calculate_teammate_points(1, 3) == 2  # 2 positions ahead
        assert calculator.calculate_teammate_points(1, 8) == 8  # 7 positions ahead
        assert calculator.calculate_teammate_points(3, 1) == 0  # Behind teammate

        # Test completion points
        assert calculator.calculate_completion_points(1.0) == 12  # Full completion
        assert calculator.calculate_completion_points(0.6) == 6  # 2 stages (25%, 50%)
        assert calculator.calculate_completion_points(0.0) == 0  # No completion

    def test_sprint_race(self, calculator):
        """Test calculation with sprint race format."""
        # Standard race (no sprint)
        std_points = calculator.calculate_driver_points(
            qualifying_pos=1,
            race_pos=1,
            rolling_avg=2.0,
            teammate_pos=2,
            race_format=RaceFormat.STANDARD,
            completion_pct=1.0,
        )

        # Sprint race
        sprint_points = calculator.calculate_driver_points(
            qualifying_pos=1,
            race_pos=1,
            sprint_pos=1,
            rolling_avg=2.0,
            teammate_pos=2,
            race_format=RaceFormat.SPRINT,
            completion_pct=1.0,
        )

        # Sprint should have more points
        assert sprint_points.total > std_points.total
        assert sprint_points.sprint == 8  # P1 in sprint

    def test_driver_position_validation(self):
        """Test validation of driver positions."""
        # Should raise error for invalid positions
        with pytest.raises(Exception):
            DriverPositions(qualifying=0, race=1)  # Invalid qualifying position

        with pytest.raises(Exception):
            DriverPositions(qualifying=1, race=21)  # Invalid race position

        with pytest.raises(Exception):
            DriverPositions(
                qualifying=1, race=1, sprint_finish=9
            )  # Invalid sprint position

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
            teammate_dist=PositionDistribution(
                {1: 0.0, 2: 0.0, 3: 1.0}
            ),  # Added position 1
            joint_qual_race=joint_dist,
        )

        # Expected overtake points: only (2,1) contributes = 0.2 * 2 * 1 = 0.4
        assert points_with_joint.overtake == pytest.approx(0.4)

        # Calculate without joint (assuming independence)
        points_independent = calculator.expected_driver_points(
            qual_dist=qual_dist,
            race_dist=race_dist,
            rolling_avg=2.0,
            teammate_dist=PositionDistribution(
                {1: 0.0, 2: 0.0, 3: 1.0}
            ),  # Added position 1
        )

        # Independent overtake points should be different
        # P(Q1,R2) = 0.6*0.3 = 0.18, P(Q2,R1) = 0.4*0.7 = 0.28
        # Expected: 0.28 * 2 * 1 = 0.56
        assert points_independent.overtake == pytest.approx(0.56)
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
                teammate_dist=PositionDistribution(
                    {1: 0.0, 2: 1.0}
                ),  # Added position 1
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
            qualifying=10,
            race=25,
            sprint=8,
            overtake=6,
            improvement=4,
            teammate=2,
            completion=12,
        )

        # Check individual components
        assert breakdown.qualifying == 10
        assert breakdown.race == 25
        assert breakdown.sprint == 8
        assert breakdown.overtake == 6
        assert breakdown.improvement == 4
        assert breakdown.teammate == 2
        assert breakdown.completion == 12

        # Check total
        assert breakdown.total == 67
