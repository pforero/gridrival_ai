"""
Tests for the odds_ratio module.

This module contains tests for the OddsRatioConverter class, verifying its ability
to convert decimal odds to probabilities using the odds ratio method.
"""

import warnings

import numpy as np
import pytest

from gridrival_ai.probabilities.converters.ratio import OddsRatioConverter


class TestOddsRatioConverter:
    """Test suite for OddsRatioConverter class."""

    def test_initialization(self):
        """Test initialization of OddsRatioConverter."""
        converter = OddsRatioConverter()
        assert converter.max_iter == 1000
        assert converter.last_or_value == 1.0

        # Test custom max_iter
        custom_converter = OddsRatioConverter(max_iter=500)
        assert custom_converter.max_iter == 500

    def test_convert_valid_odds(self):
        """Test conversion of valid decimal odds to probabilities."""
        converter = OddsRatioConverter()
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
        converter = OddsRatioConverter()
        odds = [2.0, 3.0, 4.0]
        target_sum = 0.8

        probabilities = converter.convert(odds, target_sum=target_sum)

        # Check probabilities sum to target_sum
        assert np.isclose(probabilities.sum(), target_sum)

    def test_invalid_odds(self):
        """Test that invalid odds raise appropriate errors."""
        converter = OddsRatioConverter()

        # Test odds <= 1.0
        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 1.0, 3.0])

        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 0.5, 3.0])

    def test_last_or_value_persistence(self):
        """Test that last_or_value is updated and persists between calls."""
        converter = OddsRatioConverter()
        initial_or_value = converter.last_or_value

        # First conversion
        converter.convert([2.0, 3.0, 4.0])

        # Check that last_or_value was updated
        assert converter.last_or_value != initial_or_value

        # Second conversion
        converter.convert([3.0, 4.0, 5.0])

        # The second conversion should use the updated value as starting point
        # which should lead to a different optimization result
        assert converter.last_or_value != initial_or_value

    def test_with_failed_optimization(self, monkeypatch):
        """Test behavior when optimization fails."""
        # Create a mock result that always fails
        class MockResult:
            def __init__(self):
                self.success = False
                self.message = "Mock optimization failure"
                self.x = np.array([1.5])  # Some arbitrary value

        # Mock minimize to return our failed result
        def mock_minimize(*args, **kwargs):
            return MockResult()

        # Apply the mock
        monkeypatch.setattr(
            "gridrival_ai.probabilities.converters.ratio.minimize", mock_minimize
        )

        # Set up test
        converter = OddsRatioConverter()
        odds = [2.0, 3.0, 4.0]

        # Test that warning is raised but conversion still proceeds
        with warnings.catch_warnings(record=True) as w:
            probabilities = converter.convert(odds)
            assert len(w) == 1
            assert "Optimization failed" in str(w[0].message)

        # Even with failed optimization, we should get valid probabilities
        assert np.isclose(probabilities.sum(), 1.0)

    def test_consistency(self):
        """Test that multiple calls with same input produce consistent results."""
        converter = OddsRatioConverter()
        odds = [2.5, 3.5, 5.0, 10.0]

        # Run conversion multiple times
        results = [converter.convert(odds) for _ in range(3)]

        # All results should be nearly identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i])
