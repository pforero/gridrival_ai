"""Tests for the ScoringEngine class."""

import numpy as np
import pytest

from gridrival_ai.scoring.config import ScoringConfig
from gridrival_ai.scoring.engine import ScoringEngine
from gridrival_ai.scoring.types import (
    ConstructorPositions,
    ConstructorWeekendData,
    DriverPositions,
    DriverWeekendData,
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
def engine(simple_config):
    """Create a ScoringEngine with the simple config."""
    return ScoringEngine(config=simple_config)


class TestScoringEngine:
    """Test suite for ScoringEngine class."""

    def test_initialization(self, simple_config):
        """Test initialization with config."""
        engine = ScoringEngine(config=simple_config)
        assert engine.config == simple_config
        assert engine.tables is not None

        # Test table structure
        assert "driver_points" in engine.tables
        assert "constructor_points" in engine.tables
        assert "improvement_points" in engine.tables
        assert "teammate_thresholds" in engine.tables
        assert "completion_thresholds" in engine.tables
        assert "stage_points" in engine.tables
        assert "overtake_multiplier" in engine.tables

    def test_create_tables(self, engine, simple_config):
        """Test creation of lookup tables from config."""
        tables = engine.tables

        # Check driver points table shape
        assert tables["driver_points"].shape == (3, 21)  # [type, position]

        # Check specific values in driver tables
        assert tables["driver_points"][0, 1] == 10  # P1 qualifying
        assert tables["driver_points"][1, 1] == 25  # P1 race
        assert tables["driver_points"][2, 1] == 8  # P1 sprint

        # Check constructor points table shape
        assert tables["constructor_points"].shape == (2, 21)  # [type, position]

        # Check specific values in constructor tables
        assert tables["constructor_points"][0, 1] == 5  # P1 constructor qualifying
        assert tables["constructor_points"][1, 1] == 10  # P1 constructor race

        # Check improvement points
        assert len(tables["improvement_points"]) == 6  # [0, 1, 2, 3, 4, 5]
        assert tables["improvement_points"][1] == 2  # 1 position ahead
        assert tables["improvement_points"][5] == 10  # 5 positions ahead

        # Check teammate thresholds
        assert tables["teammate_thresholds"].shape == (3, 2)  # [threshold, points]
        assert tables["teammate_thresholds"][0, 0] == 2  # 2 positions threshold
        assert tables["teammate_thresholds"][0, 1] == 2  # 2 points

        # Check other scalar values
        assert tables["overtake_multiplier"] == 2.0

    def test_calculate_driver(self, engine):
        """Test driver points calculation."""
        # Create driver weekend data
        positions = DriverPositions(qualifying=1, race=1, sprint_finish=1)
        data = DriverWeekendData(
            format=RaceFormat.SPRINT,
            positions=positions,
            completion_percentage=1.0,
            rolling_average=2.0,
            teammate_position=2,
        )

        # Calculate points
        result = engine.calculate_driver(data)

        # Check result structure
        assert isinstance(result, dict)
        assert "qualifying" in result
        assert "race" in result
        assert "sprint" in result
        assert "overtake" in result
        assert "improvement" in result
        assert "teammate" in result
        assert "completion" in result

        # Check specific values
        assert result["qualifying"] == 10  # P1 in qualifying
        assert result["race"] == 25  # P1 in race
        assert result["sprint"] == 8  # P1 in sprint
        assert result["overtake"] == 0  # No positions gained
        assert result["improvement"] == 2  # 1 position ahead of average
        assert result["teammate"] == 2  # Beat teammate by 1 position
        assert result["completion"] == 12  # Full completion (4 stages)

        # Check total
        total = sum(result.values())
        assert total == 59

    def test_calculate_constructor(self, engine):
        """Test constructor points calculation."""
        # Create constructor weekend data
        positions = ConstructorPositions(
            driver1_qualifying=1,
            driver1_race=1,
            driver2_qualifying=2,
            driver2_race=2,
        )
        data = ConstructorWeekendData(
            format=RaceFormat.STANDARD,
            positions=positions,
        )

        # Calculate points
        result = engine.calculate_constructor(data)

        # Check result structure
        assert isinstance(result, dict)
        assert "qualifying" in result
        assert "race" in result

        # Check specific values
        assert result["qualifying"] == 9  # P1 (5) + P2 (4)
        assert result["race"] == 18  # P1 (10) + P2 (8)

        # Check total
        total = sum(result.values())
        assert total == 27

    def test_calculate_driver_batch(self, engine):
        """Test batch calculation for drivers."""
        # Create structured array for batch calculation
        data = np.zeros(
            3,
            dtype=[
                ("format", np.int32),
                ("qualifying", np.int32),
                ("race", np.int32),
                ("sprint", np.int32),
                ("completion", np.float64),
                ("rolling_avg", np.float64),
                ("teammate", np.int32),
            ],
        )

        # Fill data for three scenarios
        # 1. Perfect race (P1 everywhere)
        data[0] = (
            RaceFormat.SPRINT.value,  # format
            1,  # qualifying
            1,  # race
            1,  # sprint
            1.0,  # completion
            2.0,  # rolling_avg
            2,  # teammate
        )

        # 2. Overtaking scenario
        data[1] = (
            RaceFormat.STANDARD.value,  # format
            10,  # qualifying
            5,  # race
            -1,  # sprint (N/A)
            1.0,  # completion
            8.0,  # rolling_avg
            6,  # teammate
        )

        # 3. Partial completion
        data[2] = (
            RaceFormat.STANDARD.value,  # format
            3,  # qualifying
            2,  # race
            -1,  # sprint (N/A)
            0.6,  # completion (2 stages)
            5.0,  # rolling_avg
            1,  # teammate (behind)
        )

        # Calculate batch points
        points = engine.calculate_driver_batch(data)

        # Check result shape
        assert len(points) == 3

        # Check scenario 1: Perfect race
        # 10 (qualifying) + 25 (race) + 8 (sprint) + 0 (overtake) + 2 (improvement) + 2 (teammate) + 12 (completion) = 59
        assert points[0] == 59

        # Check scenario 2: Overtaking
        # 0 (qualifying P10) + 10 (race P5) + 0 (no sprint) + 10 (5 positions * 2) + 6 (3 positions ahead) + 2 (beat teammate) + 12 (completion) = 40
        assert points[1] == 40

        # Check scenario 3: Partial completion
        # 6 (qualifying P3) + 18 (race P2) + 0 (no sprint) + 2 (1 position * 2) + 6 (3 positions ahead) + 0 (behind teammate) + 6 (2 stages) = 38
        assert points[2] == 38

    def test_calculate_constructor_batch(self, engine):
        """Test batch calculation for constructors."""
        # Create structured array for batch calculation
        data = np.zeros(
            2,
            dtype=[
                ("format", np.int32),
                ("qualifying1", np.int32),
                ("qualifying2", np.int32),
                ("race1", np.int32),
                ("race2", np.int32),
            ],
        )

        # Fill data for two scenarios
        # 1. P1/P2 scenario
        data[0] = (
            RaceFormat.STANDARD.value,  # format
            1,  # driver1 qualifying
            2,  # driver2 qualifying
            1,  # driver1 race
            2,  # driver2 race
        )

        # 2. Mid-field scenario
        data[1] = (
            RaceFormat.SPRINT.value,  # format
            3,  # driver1 qualifying
            4,  # driver2 qualifying
            2,  # driver1 race
            5,  # driver2 race
        )

        # Calculate batch points
        points = engine.calculate_constructor_batch(data)

        # Check result shape
        assert len(points) == 2

        # Check scenario 1: P1/P2
        # Qualifying: 5 (P1) + 4 (P2) = 9
        # Race: 10 (P1) + 8 (P2) = 18
        # Total: 27
        assert points[0] == 27

        # Check scenario 2: Mid-field
        # Qualifying: 3 (P3) + 2 (P4) = 5
        # Race: 8 (P2) + 4 (P5) = 12
        # Total: 17
        assert points[1] == 17

    def test_overtake_points(self, engine):
        """Test calculation of overtake points."""
        # Create driver data with different overtaking scenarios
        scenarios = [
            (1, 1, 0),  # No positions gained
            (10, 5, 10),  # 5 positions gained
            (5, 10, 0),  # Positions lost (0 points)
            (20, 1, 38),  # Maximum gain
        ]

        for qual_pos, race_pos, expected in scenarios:
            positions = DriverPositions(qualifying=qual_pos, race=race_pos)
            data = DriverWeekendData(
                format=RaceFormat.STANDARD,
                positions=positions,
                completion_percentage=0.0,  # No completion points
                rolling_average=qual_pos,  # No improvement points
                teammate_position=race_pos,  # No teammate points
            )

            result = engine.calculate_driver(data)
            assert result["overtake"] == expected

    def test_improvement_points(self, engine):
        """Test calculation of improvement points."""
        # Create driver data with different improvement scenarios
        scenarios = [
            (5, 2, 6),  # 3 positions ahead
            (5, 5, 0),  # No improvement
            (5, 10, 0),  # Worse than average
            (5, 1, 8),  # 4 positions ahead
            (20, 10, 10),  # 10 positions ahead (max in our config)
        ]

        for avg_pos, race_pos, expected in scenarios:
            positions = DriverPositions(qualifying=1, race=race_pos)
            data = DriverWeekendData(
                format=RaceFormat.STANDARD,
                positions=positions,
                completion_percentage=0.0,  # No completion points
                rolling_average=float(avg_pos),
                teammate_position=race_pos,  # No teammate points
            )

            result = engine.calculate_driver(data)
            assert result["improvement"] == expected

    def test_teammate_points(self, engine):
        """Test calculation of teammate points."""
        # Create driver data with different teammate scenarios
        scenarios = [
            (1, 3, 2),  # 2 positions ahead
            (3, 1, 0),  # Behind teammate
            (1, 10, 8),  # 9 positions ahead
            (1, 20, 8),  # Maximum threshold in our config
        ]

        for driver_pos, teammate_pos, expected in scenarios:
            positions = DriverPositions(qualifying=1, race=driver_pos)
            data = DriverWeekendData(
                format=RaceFormat.STANDARD,
                positions=positions,
                completion_percentage=0.0,  # No completion points
                rolling_average=driver_pos,  # No improvement points
                teammate_position=teammate_pos,
            )

            result = engine.calculate_driver(data)
            assert result["teammate"] == expected

    def test_completion_points(self, engine):
        """Test calculation of completion points."""
        # Create driver data with different completion percentages
        scenarios = [
            (0.0, 0),  # No completion
            (0.2, 0),  # Below first threshold
            (0.3, 3),  # Passed one threshold
            (0.6, 6),  # Passed two thresholds
            (0.8, 9),  # Passed three thresholds
            (0.95, 12),  # Passed all thresholds
            (1.0, 12),  # Full completion
        ]

        for completion_pct, expected in scenarios:
            positions = DriverPositions(qualifying=1, race=1)
            data = DriverWeekendData(
                format=RaceFormat.STANDARD,
                positions=positions,
                completion_percentage=completion_pct,
                rolling_average=1.0,  # No improvement points
                teammate_position=1,  # No teammate points
            )

            result = engine.calculate_driver(data)
            assert result["completion"] == expected

    def test_sprint_race_format(self, engine):
        """Test calculation with sprint race format."""
        # Create driver data for standard and sprint formats
        standard_positions = DriverPositions(qualifying=1, race=1)
        standard_data = DriverWeekendData(
            format=RaceFormat.STANDARD,
            positions=standard_positions,
            completion_percentage=1.0,
            rolling_average=1.0,
            teammate_position=2,
        )

        sprint_positions = DriverPositions(qualifying=1, race=1, sprint_finish=1)
        sprint_data = DriverWeekendData(
            format=RaceFormat.SPRINT,
            positions=sprint_positions,
            completion_percentage=1.0,
            rolling_average=1.0,
            teammate_position=2,
        )

        # Calculate points for both formats
        standard_result = engine.calculate_driver(standard_data)
        sprint_result = engine.calculate_driver(sprint_data)

        # Sprint format should have sprint points
        assert "sprint" in sprint_result
        assert sprint_result["sprint"] == 8  # P1 in sprint

        # Sprint format should have more total points
        standard_total = sum(standard_result.values())
        sprint_total = sum(sprint_result.values())
        assert sprint_total > standard_total
        assert sprint_total - standard_total == 8  # Difference is sprint points

    def test_driver_batch_performance(self, engine):
        """Test batch calculation performance compared to individual calculations."""
        # Create a large number of scenarios
        n_scenarios = 1000

        # Create batch data
        batch_data = np.zeros(
            n_scenarios,
            dtype=[
                ("format", np.int32),
                ("qualifying", np.int32),
                ("race", np.int32),
                ("sprint", np.int32),
                ("completion", np.float64),
                ("rolling_avg", np.float64),
                ("teammate", np.int32),
            ],
        )

        # Fill with random data
        for i in range(n_scenarios):
            batch_data[i] = (
                RaceFormat.STANDARD.value,
                np.random.randint(1, 21),  # qualifying
                np.random.randint(1, 21),  # race
                -1,  # sprint (N/A for standard)
                1.0,  # completion
                5.0,  # rolling_avg
                np.random.randint(1, 21),  # teammate
            )

        # Time batch calculation
        import time

        start_batch = time.time()
        batch_points = engine.calculate_driver_batch(batch_data)
        batch_time = time.time() - start_batch

        # Verify batch calculation produces valid results
        assert len(batch_points) == n_scenarios
        assert np.all(batch_points >= 0)

        # This is a performance test, but we don't want to enforce specific timing
        # We just check that the batch calculation completes in a reasonable time
        assert batch_time < 1.0, f"Batch calculation took {batch_time} seconds"
