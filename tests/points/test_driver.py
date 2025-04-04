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
from gridrival_ai.points.driver import DriverPointsCalculator
from gridrival_ai.probabilities.distributions import (
    JointDistribution,
    PositionDistribution,
    RaceDistribution,
    SessionDistribution,
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
def mock_race_distribution(position_dist, joint_dist):
    """Create a mock race distribution."""
    race_dist = MagicMock(spec=RaceDistribution)

    # Create and set up session distributions
    race_session = MagicMock(spec=SessionDistribution)
    qualifying_session = MagicMock(spec=SessionDistribution)
    sprint_session = MagicMock(spec=SessionDistribution)

    race_dist.race = race_session
    race_dist.qualifying = qualifying_session
    race_dist.sprint = sprint_session

    # Mock get_driver_distribution method
    race_dist.get_driver_distribution.return_value = position_dist

    # Mock get_qualifying_race_distribution method
    race_dist.get_qualifying_race_distribution.return_value = joint_dist

    # Mock get_completion_probability method
    race_dist.get_completion_probability.return_value = 0.95

    # Mock get_session method to return sessions
    def mock_get_session(session_type):
        if session_type == "race":
            return race_session
        elif session_type == "qualifying":
            return qualifying_session
        elif session_type == "sprint":
            return sprint_session
        else:
            raise ValueError(f"Invalid session type: {session_type}")

    race_dist.get_session.side_effect = mock_get_session

    # Set up race_session to return joint distribution
    race_session.get_joint_distribution.return_value = joint_dist

    return race_dist


@pytest.fixture
def driver_calculator(mock_scorer, mock_race_distribution):
    """Create a driver points calculator with mocked dependencies."""
    return DriverPointsCalculator(mock_race_distribution, mock_scorer)


class TestDriverPointsCalculator:
    """Test suite for the DriverPointsCalculator class."""

    def test_initialization(self, mock_scorer, mock_race_distribution):
        """Test initialization of driver calculator."""
        calculator = DriverPointsCalculator(mock_race_distribution, mock_scorer)

        # Verify attributes
        assert calculator.race_distribution is mock_race_distribution
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

    def test_sprint_error_fallback(self, driver_calculator, mock_race_distribution):
        """Test sprint points calculation with fallback to race distribution when error
        occurs."""
        # Set up race_distribution to raise error for sprint
        mock_race_distribution.get_driver_distribution.side_effect = [
            PositionDistribution({1: 0.6, 2: 0.4}),  # qualifying
            PositionDistribution({1: 0.7, 2: 0.3}),  # race
            KeyError("No sprint distribution"),  # sprint error
        ]

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

    def test_missing_teammate_joint_distribution(
        self, driver_calculator, mock_race_distribution
    ):
        """Test points calculation with missing teammate joint distribution."""
        # Set up race_session to raise error for get_joint_distribution
        race_session = mock_race_distribution.get_session("race")
        race_session.get_joint_distribution.side_effect = KeyError(
            "No joint distribution"
        )

        # Calculate points
        points = driver_calculator.calculate(
            driver_id="VER",
            rolling_avg=1.5,
            teammate_id="HAM",
            race_format=RaceFormat.STANDARD,
        )

        # Verify teammate points are 0
        assert points["teammate"] == 0.0

    def test_missing_completion_probability(
        self, driver_calculator, mock_race_distribution
    ):
        """Test handling of missing completion probability."""
        # Set up race_distribution to raise error for completion probability
        mock_race_distribution.get_completion_probability.side_effect = KeyError(
            "No completion prob"
        )

        # Calculate points
        points = driver_calculator.calculate(
            driver_id="VER",
            rolling_avg=1.5,
            teammate_id="HAM",
            race_format=RaceFormat.STANDARD,
        )

        # Should still have completion points (using default value)
        assert "completion" in points
        assert (
            points["completion"] > 0
        )  # Should have some points with default probability

    def test_calculator_api(self, driver_calculator, mock_race_distribution):
        """Test driver calculator adheres to expected API."""
        # Calculate points
        points = driver_calculator.calculate(  # noqa: F841
            driver_id="VER",
            rolling_avg=1.5,
            teammate_id="HAM",
            race_format=RaceFormat.STANDARD,
        )

        # Verify expected API calls
        mock_race_distribution.get_driver_distribution.assert_any_call(
            "VER", "qualifying"
        )
        mock_race_distribution.get_driver_distribution.assert_any_call("VER", "race")
        mock_race_distribution.get_qualifying_race_distribution.assert_called_with(
            "VER"
        )
        race_session = mock_race_distribution.get_session("race")
        race_session.get_joint_distribution.assert_called_with("VER", "HAM")
        mock_race_distribution.get_completion_probability.assert_called_with("VER")

    def test_missing_distribution_error(
        self, driver_calculator, mock_race_distribution
    ):
        """Test error handling with missing required distribution."""
        # Set up race_distribution to raise KeyError
        mock_race_distribution.get_driver_distribution.side_effect = KeyError(
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
