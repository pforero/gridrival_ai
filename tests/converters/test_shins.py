"""
Tests for the shins module.

This module contains tests for the ShinsConverter class, verifying its ability
to convert decimal odds to probabilities using Shin's method.
"""

import warnings

import numpy as np
import pytest

from gridrival_ai.probabilities.converters.shins import ShinsConverter


class TestShinsConverter:
    """Test suite for ShinsConverter class."""

    def test_initialization(self):
        """Test initialization of ShinsConverter."""
        converter = ShinsConverter()
        assert converter.max_iter == 1000
        assert converter.max_z == 0.2
        assert converter.last_z_value == 0.01

        # Test custom parameters
        custom_converter = ShinsConverter(max_iter=500, max_z=0.1)
        assert custom_converter.max_iter == 500
        assert custom_converter.max_z == 0.1
        assert custom_converter.last_z_value == 0.01

    def test_convert_valid_odds(self):
        """Test conversion of valid decimal odds to probabilities."""
        converter = ShinsConverter()
        odds = [3.0, 4.0, 5.0, 10.0]

        probabilities = converter.convert(odds)

        # Check result is numpy array
        assert isinstance(probabilities, np.ndarray)

        # Check probabilities sum to 1.0
        assert np.isclose(probabilities.sum(), 1.0)

        # Check probabilities are in descending order(higher odds = lower probabilities)
        assert np.all(np.diff(probabilities) <= 0)

        # Check probabilities are between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_convert_with_custom_target_sum(self):
        """Test conversion with a custom target sum."""
        converter = ShinsConverter()
        odds = [2.0, 3.0, 4.0]
        target_sum = 0.8

        probabilities = converter.convert(odds, target_sum=target_sum)

        # Check probabilities sum to target_sum
        assert np.isclose(probabilities.sum(), target_sum)

    def test_invalid_odds(self):
        """Test that invalid odds raise appropriate errors."""
        converter = ShinsConverter()

        # Test odds <= 1.0
        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 1.0, 3.0])

        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 0.5, 3.0])

    def test_last_z_value_persistence(self):
        """Test that last_z_value is updated and persists between calls."""
        converter = ShinsConverter()
        initial_z_value = converter.last_z_value

        # First conversion
        converter.convert([2.0, 3.0, 4.0])

        # Check that last_z_value was updated
        assert converter.last_z_value != initial_z_value

        # Second conversion
        converter.convert([3.0, 4.0, 5.0])

        # The second conversion should use the updated value as starting point
        # which should lead to a different optimization result
        assert converter.last_z_value != initial_z_value

    def test_z_bounds_respected(self):
        """Test that the optimized z value respects the bounds."""
        converter = ShinsConverter(max_z=0.1)
        odds = [2.0, 3.0, 4.0]

        converter.convert(odds)

        # Check that z value is within bounds
        assert 0 <= converter.last_z_value <= 0.1

        # Test with custom max_z
        custom_converter = ShinsConverter(max_z=0.05)
        custom_converter.convert(odds)

        # Check that z value is within custom bounds
        assert 0 <= custom_converter.last_z_value <= 0.05

    def test_with_failed_optimization(self, monkeypatch):
        """Test behavior when optimization fails."""
        # Create a mock result that always fails
        class MockResult:
            def __init__(self):
                self.success = False
                self.message = "Mock optimization failure"
                self.x = np.array([0.01])  # Some arbitrary value

        # Mock minimize to return our failed result
        def mock_minimize(*args, **kwargs):
            return MockResult()

        # Apply the mock
        monkeypatch.setattr(
            "gridrival_ai.probabilities.converters.shins.minimize", mock_minimize
        )

        # Set up test
        converter = ShinsConverter()
        odds = [2.0, 3.0, 4.0]

        # Test that warning is raised but conversion still proceeds
        with warnings.catch_warnings(record=True) as w:
            probabilities = converter.convert(odds)
            assert len(w) == 1
            assert "Optimization failed" in str(w[0].message)

        # Even with failed optimization, we should get valid probabilities
        assert np.isclose(probabilities.sum(), 1.0)

        # Should use the mocked z value
        assert converter.last_z_value == 0.01

    def test_shin_formula_correctness(self):
        """Test that the Shin formula is correctly implemented."""
        converter = ShinsConverter()
        odds = [2.0, 3.0, 4.0]
        raw_probs = np.array([1 / o for o in odds])

        # First run the conversion
        probabilities = converter.convert(odds)

        # Now manually apply Shin's formula with the optimized z value
        z = converter.last_z_value
        manual_probs = raw_probs * (1 - z) / (1 - z * raw_probs)

        # Normalize manually calculated probabilities
        manual_probs = manual_probs / manual_probs.sum()

        # Compare manually calculated probabilities with the output
        assert np.allclose(probabilities, manual_probs)

    def test_consistency(self):
        """Test that multiple calls with same input produce consistent results."""
        converter = ShinsConverter()
        odds = [2.5, 3.5, 5.0, 10.0]

        # Run conversion multiple times
        results = [converter.convert(odds) for _ in range(3)]

        # All results should be nearly identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i])

    def test_edge_cases(self):
        """Test conversion with edge case inputs."""
        converter = ShinsConverter()

        # Test with very large odds
        large_odds = [1000.0, 2000.0, 3000.0]
        large_probs = converter.convert(large_odds)
        assert np.isclose(large_probs.sum(), 1.0)

        # Test with very close odds
        close_odds = [2.01, 2.02, 2.03]
        close_probs = converter.convert(close_odds)
        assert np.isclose(close_probs.sum(), 1.0)

        # Should still maintain relative ordering
        assert np.all(np.diff(close_probs) <= 0)

    def test_theoretical_properties(self):
        """Test theoretical properties of Shin's method."""
        converter = ShinsConverter()
        odds = [2.0, 3.0, 4.0]

        # Run conversion
        probabilities = converter.convert(odds)
        raw_probs = np.array([1 / o for o in odds])

        # In Shin's model, with z > 0, true probabilities should be more "balanced"
        # than raw probabilities when z > 0

        # Check if maximum probability is reduced
        assert np.max(probabilities) <= np.max(raw_probs / raw_probs.sum())

        # Check if minimum probability is increased
        assert np.min(probabilities) >= np.min(raw_probs / raw_probs.sum())
