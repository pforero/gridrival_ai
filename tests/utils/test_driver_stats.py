"""Tests for driver statistics."""

import pytest

from gridrival_ai.utils.driver_stats import DriverStats, validate_driver_id


def test_validate_driver_id():
    """Test driver ID validation."""
    # Valid driver ID
    validate_driver_id("VER")  # Should not raise

    # Unknown driver ID (should warn but not raise)
    with pytest.warns(
        UserWarning, match="Driver ID XYZ is not in the list of known drivers"
    ):
        validate_driver_id("XYZ")


def test_driver_stats_creation():
    """Test basic creation of driver statistics."""
    # Valid averages
    valid_averages = {
        "VER": 1.5,
        "HAM": 3.2,
        "ALO": 4.8,
        "LEC": 5.1,
    }
    stats = DriverStats(rolling_averages=valid_averages)
    assert stats.rolling_averages == valid_averages


def test_driver_stats_validation():
    """Test validation of rolling averages."""
    # Test average too low (< 1.0)
    with pytest.raises(ValueError, match="Invalid average for VER: 0.5"):
        DriverStats(rolling_averages={"VER": 0.5})

    # Test average too high (> 20.0)
    with pytest.raises(ValueError, match="Invalid average for HAM: 21.0"):
        DriverStats(rolling_averages={"HAM": 21.0})

    # Test multiple drivers with invalid averages
    invalid_averages = {
        "VER": 1.5,  # valid
        "HAM": 0.8,  # invalid (too low)
        "PER": 20.5,  # invalid (too high)
        "LEC": 5.1,  # valid
    }
    with pytest.raises(ValueError):
        DriverStats(rolling_averages=invalid_averages)


def test_driver_stats_edge_cases():
    """Test edge cases for driver statistics."""
    # Test boundary values (should be valid)
    edge_averages = {
        "VER": 1.0,  # minimum valid value
        "HAM": 20.0,  # maximum valid value
        "ALO": 10.5,  # middle value
    }
    stats = DriverStats(rolling_averages=edge_averages)
    assert stats.rolling_averages == edge_averages

    # Test empty dictionary (should be valid)
    stats = DriverStats(rolling_averages={})
    assert stats.rolling_averages == {}
