"""
Tests for driver points calculator module.

This module contains tests for the DriverPointsCalculator class that handles
the computation of expected fantasy points for F1 drivers.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gridrival_ai.points.components import (
    CompletionPointsCalculator,
    ImprovementPointsCalculator,
    OvertakePointsCalculator,
    PositionPointsCalculator,
    TeammatePointsCalculator,
)
from gridrival_ai.points.distributions import DistributionAdapter
from gridrival_ai.points.driver import DriverPointsCalculator
from gridrival_ai.probabilities.distributions import (
    JointDistribution,
    PositionDistribution,
)
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat


@pytest.fixture
def position_dist():
    """Create a sample position distribution."""
    return PositionDistribution({1: 0.6, 2: 0.4})


@pytest.fixture
def joint_dist():
    """Create a sample joint distribution."""
    probs = {(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
    return JointDistribution(probs, entity1_name="qual", entity2_name="race")


@pytest.fixture
def mock_scorer():
    """Create a mock scorer with tables."""
    scorer = MagicMock(spec=ScoringCalculator)

    # Create mock scoring tables
    class MockTables:
        driver_points = np.zeros((3, 21))
        improvement_points = np.array([0, 2, 4, 6, 9, 12])
        teammate_thresholds = np.array([[2, 2], [5, 5], [10, 8]])
        completion_thresholds = np.array([0.25, 0.5, 0.75, 0.9])
        stage_points = 3.0
        overtake_multiplier = 2.0

    # Fill driver points tables
    for i in range(1, 21):
        MockTables.driver_points[0, i] = max(0, 50 - (i - 1) * 2)  # Qualifying
        MockTables.driver_points[1, i] = max(0, 100 - (i - 1) * 5)  # Race
        if i <= 8:
            MockTables.driver_points[2, i] = max(0, 8 - (i - 1))  # Sprint

    scorer.tables = MockTables()
    return scorer


@pytest.fixture
def mock_dist_adapter(position_dist, joint_dist):
    """Create a mock distribution adapter."""
    adapter = MagicMock(spec=DistributionAdapter)

    # Set up method returns
    adapter.get_position_distribution.return_value = position_dist
    adapter.get_position_distribution_safe.return_value = position_dist
    adapter.get_joint_distribution_safe.return_value = joint_dist
    adapter.get_qualifying_race_distribution.return_value = joint_dist
    adapter.get_completion_probability.return_value = 0.95

    return adapter


@pytest.fixture
def driver_calculator(mock_scorer, mock_dist_adapter):
    """Create a driver points calculator with mocked dependencies."""
    return DriverPointsCalculator(mock_dist_adapter, mock_scorer)


class TestDriverPointsCalculator:
    """Test suite for the DriverPointsCalculator class."""

    def test_initialization(self, mock_scorer, mock_dist_adapter):
        """Test initialization of driver calculator."""
        calculator = DriverPointsCalculator(mock_dist_adapter, mock_scorer)

        # Verify attributes
        assert calculator.distributions is mock_dist_adapter
        assert calculator.scorer is mock_scorer

        # Verify component calculators
        assert isinstance(calculator.position_calculator, PositionPointsCalculator)
        assert isinstance(calculator.overtake_calculator, OvertakePointsCalculator)
        assert isinstance(calculator.teammate_calculator, TeammatePointsCalculator)
        assert isinstance(calculator.completion_calculator, CompletionPointsCalculator)
        assert isinstance(
            calculator.improvement_calculator, ImprovementPointsCalculator
        )

    def test_calculate_standard_race(
        self, driver_calculator, position_dist, joint_dist
    ):
        """Test calculation of points for standard race format."""
        # Mock component calculators
        with patch.object(PositionPointsCalculator, "calculate") as mock_position:
            with patch.object(OvertakePointsCalculator, "calculate") as mock_overtake:
                with patch.object(
                    TeammatePointsCalculator, "calculate"
                ) as mock_teammate:
                    with patch.object(
                        CompletionPointsCalculator, "calculate"
                    ) as mock_completion:
                        with patch.object(
                            ImprovementPointsCalculator, "calculate"
                        ) as mock_improvement:
                            # Set return values
                            mock_position.side_effect = [50.0, 95.0]  # Qual, race
                            mock_overtake.return_value = 5.0
                            mock_teammate.return_value = 2.0
                            mock_completion.return_value = 12.0
                            mock_improvement.return_value = 4.0

                            # Calculate points
                            points = driver_calculator.calculate(
                                driver_id="VER",
                                rolling_avg=1.5,
                                teammate_id="HAM",
                                race_format=RaceFormat.STANDARD,
                            )

                            # Verify components
                            assert points["qualifying"] == 50.0
                            assert points["race"] == 95.0
                            assert "sprint" not in points
                            assert points["overtake"] == 5.0
                            assert points["teammate"] == 2.0
                            assert points["completion"] == 12.0
                            assert points["improvement"] == 4.0

                            # Verify total
                            assert sum(points.values()) == 168.0

    def test_calculate_sprint_race(self, driver_calculator, position_dist, joint_dist):
        """Test calculation of points for sprint race format."""
        # Mock component calculators
        with patch.object(PositionPointsCalculator, "calculate") as mock_position:
            with patch.object(OvertakePointsCalculator, "calculate") as mock_overtake:
                with patch.object(
                    TeammatePointsCalculator, "calculate"
                ) as mock_teammate:
                    with patch.object(
                        CompletionPointsCalculator, "calculate"
                    ) as mock_completion:
                        with patch.object(
                            ImprovementPointsCalculator, "calculate"
                        ) as mock_improvement:
                            # Set return values
                            mock_position.side_effect = [
                                50.0,
                                95.0,
                                8.0,
                            ]  # Qual, race, sprint
                            mock_overtake.return_value = 5.0
                            mock_teammate.return_value = 2.0
                            mock_completion.return_value = 12.0
                            mock_improvement.return_value = 4.0

                            # Calculate points
                            points = driver_calculator.calculate(
                                driver_id="VER",
                                rolling_avg=1.5,
                                teammate_id="HAM",
                                race_format=RaceFormat.SPRINT,
                            )

                            # Verify components
                            assert points["qualifying"] == 50.0
                            assert points["race"] == 95.0
                            assert points["sprint"] == 8.0
                            assert points["overtake"] == 5.0
                            assert points["teammate"] == 2.0
                            assert points["completion"] == 12.0
                            assert points["improvement"] == 4.0

                            # Verify total
                            assert sum(points.values()) == 176.0

    def test_sprint_fallback(self, driver_calculator, mock_dist_adapter):
        """Test sprint points calculation with fallback to race distribution."""
        # Set up adapter to return None for sprint
        mock_dist_adapter.get_position_distribution_safe.return_value = None

        # Mock component calculators to focus on sprint handling
        with patch.object(PositionPointsCalculator, "calculate") as mock_position:
            # Set return values
            mock_position.side_effect = [50.0, 95.0, 8.0]  # Qual, race, race as sprint

            # Calculate points
            points = driver_calculator.calculate(
                driver_id="VER",
                rolling_avg=1.5,
                teammate_id="HAM",
                race_format=RaceFormat.SPRINT,
            )

            # Verify sprint calculated from race distribution
            assert "sprint" in points
            assert points["sprint"] == 8.0

            # Verify mock calls - third call should be for sprint using race table
            assert mock_position.call_count == 3
            # First call is for qualifying
            # Second call is for race
            # Third call should be for sprint with race distribution (table index 2)
            _, args, kwargs = mock_position.mock_calls[2]
            assert np.array_equal(
                args[1], driver_calculator.scorer.tables.driver_points[2]
            )

    def test_missing_teammate_distribution(self, driver_calculator, mock_dist_adapter):
        """Test points calculation with missing teammate distribution."""
        # Set up adapter to return None for teammate joint
        mock_dist_adapter.get_joint_distribution_safe.return_value = None

        # Calculate points
        points = driver_calculator.calculate(
            driver_id="VER",
            rolling_avg=1.5,
            teammate_id="HAM",
            race_format=RaceFormat.STANDARD,
        )

        # Verify teammate points are 0
        assert points["teammate"] == 0.0

    def test_calculator_api(self, driver_calculator, mock_dist_adapter):
        """Test driver calculator adheres to expected API."""
        # Calculate points
        points = driver_calculator.calculate(  # noqa: F841
            driver_id="VER",
            rolling_avg=1.5,
            teammate_id="HAM",
            race_format=RaceFormat.STANDARD,
        )

        # Verify expected API calls
        mock_dist_adapter.get_position_distribution.assert_any_call("VER", "qualifying")
        mock_dist_adapter.get_position_distribution.assert_any_call("VER", "race")
        mock_dist_adapter.get_qualifying_race_distribution.assert_called_with("VER")
        mock_dist_adapter.get_joint_distribution_safe.assert_called_with(
            "VER", "HAM", "race"
        )
        mock_dist_adapter.get_completion_probability.assert_called_with("VER")

    def test_missing_distribution_error(self, driver_calculator, mock_dist_adapter):
        """Test error handling with missing required distribution."""
        # Set up adapter to raise KeyError
        mock_dist_adapter.get_position_distribution.side_effect = KeyError(
            "No distribution found"
        )

        # Should raise KeyError
        with pytest.raises(KeyError):
            driver_calculator.calculate(
                driver_id="VER",
                rolling_avg=1.5,
                teammate_id="HAM",
                race_format=RaceFormat.STANDARD,
            )


if __name__ == "__main__":
    pytest.main(["-v"])
