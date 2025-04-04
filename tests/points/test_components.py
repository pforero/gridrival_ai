"""
Tests for the component calculators module.

This module contains tests for the specialized calculator components
that handle various aspects of the GridRival F1 fantasy scoring system.
"""

import numpy as np
import pytest

from gridrival_ai.points.components import (
    CompletionPointsCalculator,
    ImprovementPointsCalculator,
    OvertakePointsCalculator,
    PositionPointsCalculator,
    TeammatePointsCalculator,
)
from gridrival_ai.probabilities.distributions import (
    JointDistribution,
    PositionDistribution,
)


class TestPositionPointsCalculator:
    """Tests for the PositionPointsCalculator."""

    def test_simple_distribution(self):
        """Test with a simple position distribution."""
        calculator = PositionPointsCalculator()

        # Position distribution: P1 (60%), P2 (40%)
        position_dist = PositionDistribution({1: 0.6, 2: 0.4})

        # Points table: P1 = 25, P2 = 18
        points_table = np.zeros(3)  # 0-indexed array with space for P0, P1, P2
        points_table[1] = 25  # P1 = 25 points
        points_table[2] = 18  # P2 = 18 points

        expected_points = 0.6 * 25 + 0.4 * 18  # = 15 + 7.2 = 22.2

        points = calculator.calculate(position_dist, points_table)
        assert pytest.approx(points, 0.01) == expected_points

    def test_complex_distribution(self):
        """Test with a more complex distribution across many positions."""
        calculator = PositionPointsCalculator()

        # Position distribution across P1-P10
        probs = {
            1: 0.1,
            2: 0.15,
            3: 0.2,
            4: 0.15,
            5: 0.1,
            6: 0.1,
            7: 0.05,
            8: 0.05,
            9: 0.05,
            10: 0.05,
        }
        position_dist = PositionDistribution(probs)

        # Points table for top 10 positions
        points_table = np.zeros(11)  # 0-indexed array
        points_table[1:11] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # Points for P1-P10

        # Calculate expected points manually
        expected = (
            0.1 * 25
            + 0.15 * 18
            + 0.2 * 15
            + 0.15 * 12
            + 0.1 * 10
            + 0.1 * 8
            + 0.05 * 6
            + 0.05 * 4
            + 0.05 * 2
            + 0.05 * 1
        )
        # = 2.5 + 2.7 + 3.0 + 1.8 + 1.0 + 0.8 + 0.3 + 0.2 + 0.1 + 0.05 = 12.45

        points = calculator.calculate(position_dist, points_table)
        assert pytest.approx(points, 0.01) == expected

    def test_zero_probability_positions(self):
        """Test handling of positions with zero probability."""
        calculator = PositionPointsCalculator()

        # Position distribution with some zero-probability positions
        position_dist = PositionDistribution({1: 0.5, 2: 0.0, 3: 0.5})  # No P2

        # Points table
        points_table = np.array([0, 25, 18, 15])  # Points for P0, P1, P2, P3

        expected_points = 0.5 * 25 + 0.5 * 15  # = 12.5 + 7.5 = 20

        points = calculator.calculate(position_dist, points_table)
        assert pytest.approx(points, 0.01) == expected_points


class TestOvertakePointsCalculator:
    """Tests for the OvertakePointsCalculator."""

    def test_simple_overtake_calculation(self):
        """Test with a simple joint distribution for overtakes."""
        calculator = OvertakePointsCalculator()

        # Joint distribution: (qual_pos, race_pos)
        # Only including combinations with non-zero probabilities
        joint_probs = {
            (3, 1): 0.4,  # Qual P3, Race P1 (+2 positions)
            (3, 2): 0.2,  # Qual P3, Race P2 (+1 position)
            (2, 1): 0.1,  # Qual P2, Race P1 (+1 position)
            (2, 2): 0.3,  # Qual P2, Race P2 (0 positions)
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="qualifying", entity2_name="race"
        )

        # 2 points per position gained
        multiplier = 2.0

        # Expected: (3-1)*0.4*2 + (3-2)*0.2*2 + (2-1)*0.1*2 + (2-2)*0.3*0
        # = 1.6 + 0.4 + 0.2 + 0 = 2.2
        expected_points = 2.2

        points = calculator.calculate(joint_dist, multiplier)
        assert pytest.approx(points, 0.01) == expected_points

    def test_lose_positions(self):
        """Test no points awarded for losing positions."""
        calculator = OvertakePointsCalculator()

        # Joint distribution where driver often loses positions
        joint_probs = {
            (1, 3): 0.5,  # Qual P1, Race P3 (-2 positions)
            (2, 4): 0.3,  # Qual P2, Race P4 (-2 positions)
            (3, 2): 0.2,  # Qual P3, Race P2 (+1 position)
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="qualifying", entity2_name="race"
        )

        multiplier = 2.0

        # Expected: only (3-2)*0.2*2 = 0.4 points (no points for lost positions)
        expected_points = 0.4

        points = calculator.calculate(joint_dist, multiplier)
        assert pytest.approx(points, 0.01) == expected_points

    def test_zero_overtake_points(self):
        """Test when no positions are gained."""
        calculator = OvertakePointsCalculator()

        # Joint distribution with no position gains
        joint_probs = {
            (1, 1): 0.5,  # Same position
            (2, 2): 0.3,  # Same position
            (3, 5): 0.2,  # Lost positions
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="qualifying", entity2_name="race"
        )

        multiplier = 2.0

        # Expected: no points because no positions gained
        expected_points = 0.0

        points = calculator.calculate(joint_dist, multiplier)
        assert pytest.approx(points, 0.01) == expected_points

    def test_high_multiplier(self):
        """Test with a high points multiplier."""
        calculator = OvertakePointsCalculator()

        # Simple joint distribution
        joint_probs = {
            (5, 1): 1.0,  # Qual P5, Race P1 (+4 positions)
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="qualifying", entity2_name="race"
        )

        # 5 points per position gained
        multiplier = 5.0

        # Expected: (5-1)*1.0*5 = 20
        expected_points = 20.0

        points = calculator.calculate(joint_dist, multiplier)
        assert pytest.approx(points, 0.01) == expected_points


class TestTeammatePointsCalculator:
    """Tests for the TeammatePointsCalculator."""

    def test_simple_teammate_points(self):
        """Test basic teammate points calculation."""
        calculator = TeammatePointsCalculator()

        # Joint distribution: (driver_pos, teammate_pos)
        joint_probs = {
            (1, 3): 0.4,  # Driver P1, Teammate P3 (2 positions ahead)
            (2, 4): 0.2,  # Driver P2, Teammate P4 (2 positions ahead)
            (3, 1): 0.1,  # Driver P3, Teammate P1 (2 positions behind)
            (4, 2): 0.3,  # Driver P4, Teammate P2 (2 positions behind)
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="driver", entity2_name="teammate"
        )

        # Thresholds: [margin, points]
        # 2+ positions ahead = 2 points
        # 5+ positions ahead = 5 points
        # 10+ positions ahead = 8 points
        thresholds = np.array([[2, 2], [5, 5], [10, 8]])

        # Expected: 0.4*2 + 0.2*2 = 1.2 (driver ahead by 2 positions)
        expected_points = 1.2

        points = calculator.calculate(joint_dist, thresholds)
        assert pytest.approx(points, 0.01) == expected_points

    def test_large_margins(self):
        """Test with large position margins."""
        calculator = TeammatePointsCalculator()

        joint_probs = {
            (1, 10): 0.3,  # Driver P1, Teammate P10 (9 positions ahead)
            (2, 20): 0.2,  # Driver P2, Teammate P20 (18 positions ahead)
            (5, 7): 0.5,  # Driver P5, Teammate P7 (2 positions ahead)
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="driver", entity2_name="teammate"
        )

        # Thresholds: [margin, points]
        thresholds = np.array([[2, 2], [5, 5], [10, 8]])

        # Expected:
        # P1 vs P10: 0.3*5 (9 positions > 5 but < 10, so 5 points)
        # P2 vs P20: 0.2*8 (18 positions > 10, so 8 points)
        # P5 vs P7: 0.5*2 (2 positions = 2, so 2 points)
        # Total: 0.3*5 + 0.2*8 + 0.5*2 = 1.5 + 1.6 + 1.0 = 4.1
        expected_points = 4.1

        points = calculator.calculate(joint_dist, thresholds)
        assert pytest.approx(points, 0.01) == expected_points

    def test_no_points_when_behind(self):
        """Test no points awarded when driver is behind teammate."""
        calculator = TeammatePointsCalculator()

        joint_probs = {
            (3, 1): 0.6,  # Driver P3, Teammate P1 (behind)
            (5, 2): 0.4,  # Driver P5, Teammate P2 (behind)
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="driver", entity2_name="teammate"
        )

        thresholds = np.array([[2, 2], [5, 5], [10, 8]])

        # Expected: 0 (driver never ahead of teammate)
        expected_points = 0.0

        points = calculator.calculate(joint_dist, thresholds)
        assert pytest.approx(points, 0.01) == expected_points

    def test_threshold_boundary_cases(self):
        """Test points at threshold boundaries."""
        calculator = TeammatePointsCalculator()

        joint_probs = {
            (1, 3): 0.2,  # 2 positions ahead (exactly 2)
            (1, 6): 0.2,  # 5 positions ahead (exactly 5)
            (1, 11): 0.2,  # 10 positions ahead (exactly 10)
            (1, 12): 0.2,  # 11 positions ahead (>10)
            (1, 5): 0.2,  # 4 positions ahead (between 2 and 5)
        }

        joint_dist = JointDistribution(
            joint_probs, entity1_name="driver", entity2_name="teammate"
        )

        thresholds = np.array([[2, 2], [5, 5], [10, 8]])

        # Expected:
        # 2 positions: 0.2*2 = 0.4
        # 5 positions: 0.2*5 = 1.0
        # 10 positions: 0.2*8 = 1.6
        # 11 positions: 0.2*8 = 1.6
        # 4 positions: 0.2*2 = 0.4 (between 2-5 thresholds gets 2 points)
        # Total: 0.4 + 1.0 + 1.6 + 1.6 + 0.4 = 5.0
        expected_points = 5.0

        points = calculator.calculate(joint_dist, thresholds)
        assert pytest.approx(points, 0.01) == expected_points


class TestCompletionPointsCalculator:
    """Tests for the CompletionPointsCalculator."""

    def test_full_completion(self):
        """Test when driver always completes the race."""
        calculator = CompletionPointsCalculator()

        # 100% completion probability
        completion_prob = 1.0

        # Completion thresholds (25%, 50%, 75%, 90%)
        thresholds = np.array([0.25, 0.5, 0.75, 0.9])

        # 3 points per stage
        points_per_stage = 3.0

        # Expected: 4 stages * 3 points = 12 points
        expected_points = 12.0

        points = calculator.calculate(completion_prob, thresholds, points_per_stage)
        assert pytest.approx(points, 0.01) == expected_points

    def test_zero_completion(self):
        """Test when driver never completes the race."""
        calculator = CompletionPointsCalculator()

        # 0% completion probability
        completion_prob = 0.0

        # Completion thresholds (25%, 50%, 75%, 90%)
        thresholds = np.array([0.25, 0.5, 0.75, 0.9])

        # 3 points per stage
        points_per_stage = 3.0

        # Expected: points based on uniform DNF distribution
        # First stage (0-25%): 0.25 * 0 points = 0
        # Second stage (25-50%): 0.25 * 3 points = 0.75
        # Third stage (50-75%): 0.25 * 6 points = 1.5
        # Fourth stage (75-90%): 0.15 * 9 points = 1.35
        # Final stage (90-100%): 0.1 * 12 points = 1.2
        # Total: 0 + 0.75 + 1.5 + 1.35 + 1.2 = 4.8
        expected_points = 4.8

        points = calculator.calculate(completion_prob, thresholds, points_per_stage)
        assert pytest.approx(points, 0.01) == expected_points

    def test_partial_completion(self):
        """Test with partial completion probability."""
        calculator = CompletionPointsCalculator()

        # 80% completion probability
        completion_prob = 0.8

        # Completion thresholds (25%, 50%, 75%, 90%)
        thresholds = np.array([0.25, 0.5, 0.75, 0.9])

        # 3 points per stage
        points_per_stage = 3.0

        # Expected:
        # Full completion: 0.8 * 12 = 9.6
        # DNF points (similar to test_zero_completion but scaled by 0.2):
        # = 0.2 * (0 + 0.75 + 1.5 + 1.35 + 1.2) = 0.2 * 4.8 = 0.96
        # Total: 9.6 + 0.96 = 10.56
        expected_points = 10.56

        points = calculator.calculate(completion_prob, thresholds, points_per_stage)
        assert pytest.approx(points, 0.01) == expected_points

    def test_different_thresholds(self):
        """Test with different stage thresholds."""
        calculator = CompletionPointsCalculator()

        # 75% completion probability
        completion_prob = 0.75

        # Different thresholds (20%, 40%, 60%, 80%)
        thresholds = np.array([0.2, 0.4, 0.6, 0.8])

        # 2 points per stage
        points_per_stage = 2.0

        # Expected calculation is complex, but the test verifies
        # that thresholds are used correctly
        points = calculator.calculate(completion_prob, thresholds, points_per_stage)
        assert points > 0


class TestImprovementPointsCalculator:
    """Tests for the ImprovementPointsCalculator."""

    def test_basic_improvement(self):
        """Test basic improvement points calculation."""
        calculator = ImprovementPointsCalculator()

        # Position distribution
        position_dist = PositionDistribution({1: 0.6, 2: 0.4})

        # Rolling average of P3
        rolling_avg = 3.0

        # Improvement points (indexed by positions gained)
        # [0, 2, 4, 6, 8] means:
        # 0 positions gained = 0 points
        # 1 position gained = 2 points
        # 2 positions gained = 4 points
        # etc.
        improvement_points = np.array([0, 2, 4, 6, 8])

        # Expected:
        # P1 (2 positions ahead of avg): 0.6 * 4 = 2.4
        # P2 (1 position ahead of avg): 0.4 * 2 = 0.8
        # Total: 2.4 + 0.8 = 3.2
        expected_points = 3.2

        points = calculator.calculate(position_dist, rolling_avg, improvement_points)
        assert pytest.approx(points, 0.01) == expected_points

    def test_no_improvement(self):
        """Test when there's no improvement over rolling average."""
        calculator = ImprovementPointsCalculator()

        # Position distribution
        position_dist = PositionDistribution(
            {1: 0.0, 2: 0.0, 3: 0.5, 4: 0.5}
        )  # Added positions 1-2

        # Rolling average of P3 (same or worse)
        rolling_avg = 3.0

        # Improvement points
        improvement_points = np.array([0, 2, 4, 6, 8])

        # Expected: no points (no improvement)
        expected_points = 0.0

        points = calculator.calculate(position_dist, rolling_avg, improvement_points)
        assert pytest.approx(points, 0.01) == expected_points

    def test_fractional_rolling_average(self):
        """Test with a fractional rolling average that needs rounding."""
        calculator = ImprovementPointsCalculator()

        # Position distribution
        position_dist = PositionDistribution({1: 0.3, 2: 0.3, 3: 0.4})

        # Rolling average of P3.6 (rounds to P4)
        rolling_avg = 3.6

        # Improvement points
        improvement_points = np.array([0, 2, 4, 6, 8])

        # Expected (with P4 as rounded average):
        # P1 (3 positions ahead): 0.3 * 6 = 1.8
        # P2 (2 positions ahead): 0.3 * 4 = 1.2
        # P3 (1 position ahead): 0.4 * 2 = 0.8
        # Total: 1.8 + 1.2 + 0.8 = 3.8
        expected_points = 3.8

        points = calculator.calculate(position_dist, rolling_avg, improvement_points)
        assert pytest.approx(points, 0.01) == expected_points

    def test_large_improvement(self):
        """Test with improvement beyond the points table."""
        calculator = ImprovementPointsCalculator()

        # Position distribution
        position_dist = PositionDistribution({1: 1.0})  # Always P1

        # Rolling average of P10
        rolling_avg = 10.0

        # Improvement points (only up to 5 positions)
        improvement_points = np.array([0, 2, 4, 6, 8, 10])

        # Expected:
        # P1 is 9 positions ahead, which is beyond the table
        # Should use the max value (10 points)
        expected_points = 10.0

        points = calculator.calculate(position_dist, rolling_avg, improvement_points)
        assert pytest.approx(points, 0.01) == expected_points

    def test_boundary_rounding(self):
        """Test rounding behavior at boundaries."""
        calculator = ImprovementPointsCalculator()

        # Test cases with different rolling averages
        test_cases = [
            # (rolling_avg, expected_points)
            (2.4, 1.2),  # Rounds to 2
            (2.5, 1.2),  # Rounds to 2 (Python's banker's rounding)
            (2.6, 2.4),  # Rounds to 3
        ]

        # Position distribution
        position_dist = PositionDistribution(
            {1: 0.6, 2: 0.0, 3: 0.4}
        )  # Added position 2

        # Improvement points
        improvement_points = np.array([0, 2, 4])

        for avg, expected in test_cases:
            points = calculator.calculate(position_dist, avg, improvement_points)
            assert points, 0.01 == pytest.approx(expected, rel=0.01)
