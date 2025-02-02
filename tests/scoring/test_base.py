"""Tests for configuration loading and validation."""

import json
from pathlib import Path

import pytest

from gridrival_ai.scoring.base import ScoringConfig
from gridrival_ai.scoring.constants import (
    DEFAULT_COMPLETION_STAGE_POINTS,
    DEFAULT_CONSTRUCTOR_QUALIFYING_POINTS,
    DEFAULT_CONSTRUCTOR_RACE_POINTS,
    DEFAULT_OVERTAKE_MULTIPLIER,
    DEFAULT_QUALIFYING_POINTS,
    MAX_MULTIPLIER,
    MAX_POINTS,
)
from gridrival_ai.scoring.exceptions import ConfigurationError


@pytest.fixture
def valid_config_dict():
    """Create valid configuration dictionary."""
    return {
        "qualifying_points": {str(i): 52 - (i * 2) for i in range(1, 21)},
        "race_points": {str(i): 103 - (i * 3) for i in range(1, 21)},
        "sprint_points": {str(i): 9 - i for i in range(1, 9)},
        "constructor_qualifying_points": {str(i): 26 - i for i in range(1, 21)},
        "constructor_race_points": {str(i): 52 - (i * 2) for i in range(1, 21)},
        "completion_stage_points": 3.0,
        "overtake_multiplier": 3.0,
        "improvement_points": {"1": 2, "2": 4, "3": 6},
        "teammate_points": {"3": 2, "7": 5, "12": 8},
    }


@pytest.fixture
def config_file(tmp_path: Path, valid_config_dict):
    """Create temporary config file."""
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(valid_config_dict, f)
    return config_path


def test_default_initialization():
    """Test default configuration initialization."""
    config = ScoringConfig()
    assert config.qualifying_points == DEFAULT_QUALIFYING_POINTS
    assert config.constructor_qualifying_points == DEFAULT_CONSTRUCTOR_QUALIFYING_POINTS
    assert config.constructor_race_points == DEFAULT_CONSTRUCTOR_RACE_POINTS
    assert config.completion_stage_points == DEFAULT_COMPLETION_STAGE_POINTS
    assert config.overtake_multiplier == DEFAULT_OVERTAKE_MULTIPLIER


def test_json_loading(config_file):
    """Test loading configuration from JSON."""
    config = ScoringConfig.from_json(config_file)
    assert config.qualifying_points[1] == 50
    assert config.race_points[1] == 100
    assert config.sprint_points[1] == 8
    assert config.constructor_qualifying_points[1] == 25
    assert config.constructor_race_points[1] == 50
    assert config.completion_stage_points == 3.0


def test_invalid_json():
    """Test handling of invalid JSON file."""
    with pytest.raises(ConfigurationError, match="Failed to load config"):
        ScoringConfig.from_json("nonexistent.json")


def test_invalid_config_format(tmp_path):
    """Test validation of configuration format."""
    config_path = tmp_path / "invalid_config.json"

    # Create invalid qualifying points with position 0 (should be 1-20)
    invalid_config = {
        "qualifying_points": {"0": 50},  # Invalid position
        "race_points": {str(i): 103 - (i * 3) for i in range(1, 21)},
        "sprint_points": {str(i): 9 - i for i in range(1, 9)},
        "constructor_qualifying_points": {str(i): 26 - i for i in range(1, 21)},
        "constructor_race_points": {str(i): 52 - (i * 2) for i in range(1, 21)},
        "completion_stage_points": 3.0,
        "overtake_multiplier": 3.0,
        "improvement_points": {"1": 2, "2": 4, "3": 6},
        "teammate_points": {"3": 2, "7": 5, "12": 8},
    }

    with open(config_path, "w") as f:
        json.dump(invalid_config, f)

    with pytest.raises(
        ConfigurationError, match="Invalid positions in qualifying_points"
    ):
        ScoringConfig.from_json(config_path)


def test_json_roundtrip(config_file, tmp_path):
    """Test saving and loading configuration."""
    # Load original config
    config = ScoringConfig.from_json(config_file)

    # Save to new file
    output_path = tmp_path / "output_config.json"
    config.to_json(output_path)

    # Load saved config
    reloaded = ScoringConfig.from_json(output_path)

    # Compare all attributes
    assert config.qualifying_points == reloaded.qualifying_points
    assert config.race_points == reloaded.race_points
    assert config.sprint_points == reloaded.sprint_points
    assert (
        config.constructor_qualifying_points == reloaded.constructor_qualifying_points
    )
    assert config.constructor_race_points == reloaded.constructor_race_points
    assert config.completion_stage_points == reloaded.completion_stage_points
    assert config.overtake_multiplier == reloaded.overtake_multiplier
    assert config.improvement_points == reloaded.improvement_points
    assert config.teammate_points == reloaded.teammate_points


def test_validation_point_values():
    """Test validation of point values."""
    # Create full qualifying points but with a negative value
    qualifying_points = {i: 52 - (i * 2) for i in range(1, 21)}
    qualifying_points[1] = -1  # Set negative points

    with pytest.raises(ConfigurationError, match="Invalid point values"):
        ScoringConfig(qualifying_points=qualifying_points)

    # Test constructor points with negative value
    constructor_qualifying_points = {i: 26 - i for i in range(1, 21)}
    constructor_qualifying_points[1] = -1

    with pytest.raises(ConfigurationError, match="Invalid point values"):
        ScoringConfig(constructor_qualifying_points=constructor_qualifying_points)

    # Test points above maximum
    with pytest.raises(ConfigurationError, match="must be between"):
        ScoringConfig(completion_stage_points=MAX_POINTS + 1)


def test_validation_multipliers():
    """Test validation of multiplier values."""
    # Test multiplier below minimum
    with pytest.raises(ConfigurationError, match="must be between"):
        ScoringConfig(overtake_multiplier=0.5)

    # Test multiplier above maximum
    with pytest.raises(ConfigurationError, match="must be between"):
        ScoringConfig(overtake_multiplier=MAX_MULTIPLIER + 1)


def test_validation_required_positions():
    """Test validation of required position mappings."""
    # Missing qualifying positions
    with pytest.raises(ConfigurationError, match="must have points for all positions"):
        ScoringConfig(qualifying_points={1: 50})  # Missing positions 2-20

    # Missing race positions
    with pytest.raises(ConfigurationError, match="must have points for all positions"):
        ScoringConfig(race_points={1: 100})  # Missing positions 2-20

    # Missing constructor qualifying positions
    with pytest.raises(ConfigurationError, match="must have points for all positions"):
        ScoringConfig(constructor_qualifying_points={1: 25})  # Missing positions 2-20

    # Missing constructor race positions
    with pytest.raises(ConfigurationError, match="must have points for all positions"):
        ScoringConfig(constructor_race_points={1: 50})  # Missing positions 2-20


def test_validation_optional_positions():
    """Test validation of optional position mappings."""
    # Valid partial improvement points
    config = ScoringConfig(improvement_points={1: 2, 2: 4})  # Only two positions
    assert len(config.improvement_points) == 2

    # Valid partial teammate points
    config = ScoringConfig(teammate_points={3: 2, 7: 5})  # Only two thresholds
    assert len(config.teammate_points) == 2


def test_sprint_position_validation():
    """Test validation of sprint position limits."""
    # Invalid sprint position (above 8)
    with pytest.raises(ConfigurationError, match="must be between"):
        ScoringConfig(sprint_points={9: 1})

    # Valid sprint positions (1-8 only)
    config = ScoringConfig(sprint_points={i: 9 - i for i in range(1, 9)})
    assert len(config.sprint_points) == 8
    assert max(config.sprint_points.keys()) == 8


def test_save_error_handling(tmp_path):
    """Test error handling when saving configuration."""
    config = ScoringConfig()
    invalid_path = tmp_path / "nonexistent" / "config.json"

    with pytest.raises(ConfigurationError, match="Failed to save config"):
        config.to_json(invalid_path)
