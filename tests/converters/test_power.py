"""
Tests for the power module.

This module contains tests for the PowerConverter class, verifying its ability
to convert decimal odds to probabilities using the power method.
"""

import warnings

import numpy as np
import pytest

from gridrival_ai.probabilities.converters.power import PowerConverter


class TestPowerConverter:
    """Test suite for PowerConverter class."""

    def test_initialization(self):
        """Test initialization of PowerConverter."""
        converter = PowerConverter()
        assert converter.max_iter == 1000
        assert converter.last_k_value == 1.0

        # Test custom max_iter
        custom_converter = PowerConverter(max_iter=500)
        assert custom_converter.max_iter == 500

    def test_convert_valid_odds(self):
        """Test conversion of valid decimal odds to probabilities."""
        converter = PowerConverter()
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
        converter = PowerConverter()
        odds = [2.0, 3.0, 4.0]
        target_sum = 0.8

        probabilities = converter.convert(odds, target_sum=target_sum)

        # Check probabilities sum to target_sum
        assert np.isclose(probabilities.sum(), target_sum)

    def test_invalid_odds(self):
        """Test that invalid odds raise appropriate errors."""
        converter = PowerConverter()

        # Test odds <= 1.0
        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 1.0, 3.0])

        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 0.5, 3.0])

    def test_last_k_value_persistence(self):
        """Test that last_k_value is updated and persists between calls."""
        converter = PowerConverter()
        initial_k_value = converter.last_k_value

        # First conversion
        converter.convert([2.0, 3.0, 4.0])

        # Check that last_k_value was updated
        assert converter.last_k_value != initial_k_value

        # Second conversion
        converter.convert([3.0, 4.0, 5.0])

        # The second conversion should use the updated value as starting point
        # which may lead to a different optimization result
        assert converter.last_k_value != initial_k_value

    def test_optimization_methods(self, monkeypatch):
        """Test that multiple optimization methods are tried."""
        called_methods = []

        # Original minimize function
        original_minimize = PowerConverter.convert.__globals__["minimize"]

        # Mock minimize to track called methods
        def mock_minimize(*args, **kwargs):
            called_methods.append(kwargs.get("method", "unknown"))
            # Make first method fail
            if len(called_methods) == 1:
                raise Exception("First method failed")
            # Return result from original function for second method
            return original_minimize(*args, **kwargs)

        # Apply the mock
        monkeypatch.setattr(
            "gridrival_ai.probabilities.converters.power.minimize", mock_minimize
        )

        # Set up test
        converter = PowerConverter()
        odds = [2.0, 3.0, 4.0]

        # Run conversion
        probabilities = converter.convert(odds)

        # Check that multiple methods were called
        assert len(called_methods) > 1
        assert "Nelder-Mead" in called_methods
        assert "L-BFGS-B" in called_methods

        # Check that we still got valid probabilities
        assert np.isclose(probabilities.sum(), 1.0)

    def test_all_optimization_methods_fail(self, monkeypatch):
        """Test behavior when all optimization methods fail."""
        # Mock minimize to always fail
        def mock_minimize(*args, **kwargs):
            raise Exception("Optimization failed")

        # Apply the mock
        monkeypatch.setattr(
            "gridrival_ai.probabilities.converters.power.minimize", mock_minimize
        )

        # Set up test
        converter = PowerConverter()
        odds = [2.0, 3.0, 4.0]

        # Test that warning is raised but conversion still proceeds
        with warnings.catch_warnings(record=True) as w:
            probabilities = converter.convert(odds)
            assert len(w) == 1
            assert "All optimization methods failed" in str(w[0].message)

        # Even with failed optimization, we should get valid probabilities
        assert np.isclose(probabilities.sum(), 1.0)

        # Should use fallback k=1.0
        expected_raw_probs = np.array([1 / 2.0, 1 / 3.0, 1 / 4.0])
        expected_normalized = expected_raw_probs / expected_raw_probs.sum()
        assert np.allclose(probabilities, expected_normalized)

    def test_objective_function_behavior(self):
        """Test the behavior of the objective function."""
        converter = PowerConverter()
        odds = [2.0, 3.0, 4.0]

        # Access the objective function from the convert method
        # This is a bit of a hack for testing purposes
        raw_probs = np.array([1 / o for o in odds])

        def get_objective():
            # Need to recreate the objective function with the current raw_probs
            def objective(k: float) -> float:
                if k <= 0:
                    return float("inf")
                probs = raw_probs ** (1 / k)
                return abs(1.0 - probs.sum())

            return objective

        objective = get_objective()

        # Test that negative k values return infinity
        assert objective(-1.0) == float("inf")

        # Test that k=1.0 returns the absolute difference from 1.0
        expected_diff = abs(1.0 - raw_probs.sum())
        assert objective(1.0) == expected_diff

        # Find a k value that makes probabilities sum to 1.0
        # This should be close to the value found by the converter
        converter.convert(odds)
        optimal_k = converter.last_k_value

        # The objective value at the optimal k should be very small
        assert objective(optimal_k) < 1e-5

    def test_consistency(self):
        """Test that multiple calls with same input produce consistent results."""
        converter = PowerConverter()
        odds = [2.5, 3.5, 5.0, 10.0]

        # Run conversion multiple times
        results = [converter.convert(odds) for _ in range(3)]

        # All results should be nearly identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i])

    def test_edge_cases(self):
        """Test conversion with edge case inputs."""
        converter = PowerConverter()

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
