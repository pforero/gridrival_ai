"""Integration tests for the scoring module."""

import json
import tempfile
from pathlib import Path

import pytest

from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.config import ScoringConfig
from gridrival_ai.scoring.engine import ScoringEngine
from gridrival_ai.scoring.types import (
    DriverPositions,
    DriverWeekendData,
    RaceFormat,
)


@pytest.fixture
def custom_config_dict():
    """Create a custom scoring configuration dictionary for testing."""
    return {
        "qualifying_points": {str(i): 21 - i for i in range(1, 21)},
        "race_points": {str(i): 41 - (i * 2) for i in range(1, 21)},
        "sprint_points": {str(i): 9 - i for i in range(1, 9)},
        "constructor_qualifying_points": {str(i): 11 - (i // 2) for i in range(1, 21)},
        "constructor_race_points": {str(i): 21 - i for i in range(1, 21)},
        "improvement_points": {"1": 1, "2": 3, "3": 5, "4": 7, "5": 9},
        "teammate_points": {"3": 3, "6": 6, "10": 9},
        "completion_stage_points": 4.0,
        "overtake_multiplier": 1.5,
    }


@pytest.fixture
def custom_config_file(custom_config_dict):
    """Create a temporary file with custom configuration."""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    temp_path = Path(temp_file.name)

    # Close the file first (Windows compatibility)
    temp_file.close()

    # Write JSON data to the file
    with open(temp_path, "w") as f:
        json.dump(custom_config_dict, f)

    yield temp_path

    # Cleanup after test
    if temp_path.exists():
        temp_path.unlink()


class TestScoringIntegration:
    """Test integration between scoring components."""

    def test_config_to_engine_to_calculator(self, custom_config_file):
        """Test integration from config file to calculator."""
        # Load config from file
        config = ScoringConfig.from_json(custom_config_file)

        # Create engine and calculator
        engine = ScoringEngine(config)
        calculator = ScoringCalculator(config)

        # Ensure consistent tables in engine
        assert engine.tables["stage_points"] == 4.0
        assert engine.tables["overtake_multiplier"] == 1.5

        # Create test data
        positions = DriverPositions(qualifying=1, race=1)
        data = DriverWeekendData(
            format=RaceFormat.STANDARD,
            positions=positions,
            completion_percentage=1.0,
            rolling_average=1.0,
            teammate_position=2,
        )

        # Calculate points using both direct engine and calculator
        engine_result = engine.calculate_driver(data)
        calculator_result = calculator.calculate_driver_points(
            qualifying_pos=1,
            race_pos=1,
            rolling_avg=1.0,
            teammate_pos=2,
            completion_pct=1.0,
        )

        # Verify consistent results
        assert engine_result["qualifying"] == 20  # P1 in custom config
        assert engine_result["race"] == 39  # P1 in custom config
        assert engine_result["completion"] == 16  # 4 stages * 4 points

        # Calculator should produce same component values
        assert calculator_result.qualifying == engine_result["qualifying"]
        assert calculator_result.race == engine_result["race"]
        assert calculator_result.completion == engine_result["completion"]
        assert calculator_result.overtake == engine_result["overtake"]
        assert calculator_result.improvement == engine_result["improvement"]
        assert calculator_result.teammate == engine_result["teammate"]

        # Total should match sum of components
        assert calculator_result.total == sum(engine_result.values())

    def test_config_modifications(self):
        """Test calculator responds to config modifications."""
        # Create base config
        base_config = ScoringConfig(
            qualifying_points={i: 10 for i in range(1, 21)},
            race_points={i: 20 for i in range(1, 21)},
        )

        # Create calculator with base config
        calculator = ScoringCalculator(base_config)

        # Test with base config
        base_points = calculator.calculate_driver_points(
            qualifying_pos=1,
            race_pos=1,
            rolling_avg=1.0,
            teammate_pos=2,
        )

        # Modify config (double qualifying points)
        modified_config = ScoringConfig(
            qualifying_points={i: 20 for i in range(1, 21)},
            race_points={i: 20 for i in range(1, 21)},
        )

        # Create calculator with modified config
        modified_calculator = ScoringCalculator(modified_config)

        # Test with modified config
        modified_points = modified_calculator.calculate_driver_points(
            qualifying_pos=1,
            race_pos=1,
            rolling_avg=1.0,
            teammate_pos=2,
        )

        # Modified calculator should have higher qualifying points
        assert modified_points.qualifying == 20
        assert base_points.qualifying == 10

        # Total should differ by the qualifying points difference
        assert modified_points.total - base_points.total == 10

    def test_engine_reuse(self, custom_config_file):
        """Test reuse of engine with different configs."""
        # Load config from file
        config1 = ScoringConfig.from_json(custom_config_file)

        # Create second config with different values
        config2 = ScoringConfig(
            qualifying_points={i: 5 for i in range(1, 21)},
            race_points={i: 10 for i in range(1, 21)},
        )

        # Create engines with different configs
        engine1 = ScoringEngine(config1)
        engine2 = ScoringEngine(config2)

        # Create test data
        positions = DriverPositions(qualifying=1, race=1)
        data = DriverWeekendData(
            format=RaceFormat.STANDARD,
            positions=positions,
            completion_percentage=1.0,
            rolling_average=1.0,
            teammate_position=2,
        )

        # Calculate points using both engines
        result1 = engine1.calculate_driver(data)
        result2 = engine2.calculate_driver(data)

        # Results should reflect their different configs
        assert result1["qualifying"] == 20  # From config1
        assert result2["qualifying"] == 5  # From config2
        assert result1["race"] == 39  # From config1
        assert result2["race"] == 10  # From config2

    def test_calculator_component_methods(self, custom_config_file):
        """Test calculator component methods with custom config."""
        # Load custom config
        config = ScoringConfig.from_json(custom_config_file)
        calculator = ScoringCalculator(config)

        # Test individual component methods
        assert calculator.calculate_qualifying_points(1) == 20
        assert calculator.calculate_race_points(1) == 39
        assert calculator.calculate_sprint_points(1) == 8
        assert calculator.calculate_overtake_points(10, 5) == 7.5  # 5 positions * 1.5
        assert calculator.calculate_improvement_points(1, 5.0) == 7  # 4 positions ahead
        assert calculator.calculate_teammate_points(1, 10) == 9  # 9 positions ahead
        assert calculator.calculate_completion_points(1.0) == 16  # 4 stages * 4 points

    def test_full_race_weekend_calculation(self):
        """Test calculation for complete race weekend."""
        # Create default config
        config = ScoringConfig.default()
        calculator = ScoringCalculator(config)

        # Calculate points for specific driver scenario
        points = calculator.calculate_driver_points(
            qualifying_pos=2,  # P2 in qualifying
            race_pos=1,  # P1 in race
            rolling_avg=3.0,  # 3.0 rolling average
            teammate_pos=5,  # Teammate P5
            sprint_pos=2,  # P2 in sprint
            race_format=RaceFormat.SPRINT,
            completion_pct=1.0,  # Full completion
        )

        # Expected components
        assert points.qualifying == 48  # P2 in qualifying
        assert points.race == 100  # P1 in race
        assert points.sprint == 7  # P2 in sprint
        assert points.overtake == 3  # 1 position gained * 3.0
        assert points.improvement == 4  # 2 positions ahead
        assert points.teammate == 5  # 4 positions ahead
        assert points.completion == 12  # Full completion

        # Total should be sum of all components
        assert points.total == sum(
            [
                points.qualifying,
                points.race,
                points.sprint,
                points.overtake,
                points.improvement,
                points.teammate,
                points.completion,
            ]
        )
