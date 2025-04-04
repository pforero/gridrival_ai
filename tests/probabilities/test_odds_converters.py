"""
Tests for the odds converters module.

This module contains tests for the odds converters base class,
implementations, and factory function.
"""

import math
from typing import List

import numpy as np
import pytest

from gridrival_ai.probabilities.distributions import PositionDistribution
from gridrival_ai.probabilities.odds_converters import (
    BasicConverter,
    OddsConverter,
    OddsRatioConverter,
    PowerConverter,
    ShinsConverter,
    get_odds_converter,
)


class TestBaseOddsConverter:
    """Tests for the OddsConverter abstract base class."""

    def test_cannot_instantiate_abstract_base(self):
        """Test that OddsConverter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OddsConverter()

    def test_subclass_must_implement_convert(self):
        """Test that subclasses must implement convert method."""

        class IncompleteConverter(OddsConverter):
            pass  # Missing convert method implementation

        with pytest.raises(TypeError):
            IncompleteConverter()

    def test_minimal_valid_subclass(self):
        """Test that a minimal valid subclass can be instantiated."""

        class MinimalConverter(OddsConverter):
            def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
                return np.ones(len(odds)) * (target_sum / len(odds))

        converter = MinimalConverter()
        result = converter.convert([2.0, 4.0], 1.0)
        assert isinstance(result, np.ndarray)
        assert math.isclose(result.sum(), 1.0)


class TestBasicConverter:
    """Tests for the BasicConverter implementation."""

    def test_initialization(self):
        """Test that BasicConverter can be instantiated."""
        converter = BasicConverter()
        assert isinstance(converter, OddsConverter)

    def test_convert_simple_case(self):
        """Test conversion with simple odds."""
        converter = BasicConverter()
        odds = [2.0, 4.0]
        result = converter.convert(odds)

        # Expected: [1/2, 1/4] normalized to sum to 1.0
        # = [0.5, 0.25] * (1.0/0.75) = [2/3, 1/3]
        expected = np.array([2 / 3, 1 / 3])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, expected)
        assert math.isclose(result.sum(), 1.0)

    def test_convert_win_odds(self):
        """Test conversion with typical F1 win odds."""
        converter = BasicConverter()
        odds = [2.5, 3.0, 6.0, 10.0, 15.0]
        result = converter.convert(odds)

        # Expected raw probs: [0.4, 0.33333, 0.16667, 0.1, 0.06667]
        # Sum = 1.06667
        # Normalized: raw_probs * (1.0/1.06667)
        expected = np.array([0.4, 1 / 3, 1 / 6, 0.1, 1 / 15]) / 1.06667

        assert np.allclose(result, expected, rtol=1e-5)
        assert math.isclose(result.sum(), 1.0)

    def test_convert_with_custom_target_sum(self):
        """Test conversion with a custom target sum (e.g., for top-3 market)."""
        converter = BasicConverter()
        odds = [1.5, 2.0, 3.0]
        result = converter.convert(odds, target_sum=3.0)

        # Expected raw probs: [0.66667, 0.5, 0.33333]
        # Sum = 1.5
        # Normalized to 3.0: raw_probs * (3.0/1.5)
        expected = np.array([0.66667, 0.5, 0.33333]) * 2.0

        assert np.allclose(result, expected, rtol=1e-5)
        assert math.isclose(result.sum(), 3.0)

    def test_rejects_invalid_odds(self):
        """Test that converter rejects odds <= 1.0."""
        converter = BasicConverter()

        # Test with a single invalid odd
        with pytest.raises(ValueError):
            converter.convert([2.0, 1.0, 3.0])

        # Test with all invalid odds
        with pytest.raises(ValueError):
            converter.convert([0.5, 0.9, 1.0])


class TestPowerConverter:
    """Tests for the PowerConverter implementation."""

    def test_initialization(self):
        """Test that PowerConverter can be instantiated with default parameters."""
        converter = PowerConverter()
        assert isinstance(converter, OddsConverter)
        assert converter.max_iter == 1000
        assert converter.last_k_value == 1.0

    def test_initialization_with_params(self):
        """Test that PowerConverter can be instantiated with custom parameters."""
        converter = PowerConverter(max_iter=2000)
        assert converter.max_iter == 2000

    def test_convert_simple_case(self):
        """Test conversion with simple, balanced odds."""
        converter = PowerConverter()
        odds = [2.0, 2.0, 2.0, 2.0]  # Equal odds
        result = converter.convert(odds)

        # For equal odds, all results should be equal and sum to 1.0
        expected = np.ones(4) / 4

        assert np.allclose(result, expected)
        assert math.isclose(result.sum(), 1.0)

    def test_convert_realistic_case(self):
        """Test conversion with realistic F1 odds."""
        converter = PowerConverter()
        odds = [2.0, 3.0, 5.0, 10.0, 20.0]
        result = converter.convert(odds)

        # Result should sum to 1.0 and be in descending order
        assert math.isclose(result.sum(), 1.0)
        assert np.all(np.diff(result) <= 0)

        # The k value should be adjusted to fit the target sum
        assert converter.last_k_value != 1.0

    def test_convert_with_custom_target_sum(self):
        """Test conversion with a custom target sum."""
        converter = PowerConverter()
        odds = [2.0, 3.0, 5.0]
        result = converter.convert(odds, target_sum=2.0)

        assert math.isclose(result.sum(), 2.0)

    def test_conversion_consistency(self):
        """Test that repeated conversions with same odds produce consistent results."""
        converter = PowerConverter()
        odds = [2.0, 3.0, 5.0, 10.0]

        result1 = converter.convert(odds)
        result2 = converter.convert(odds)

        assert np.allclose(result1, result2)

    def test_rejects_invalid_odds(self):
        """Test that converter rejects odds <= 1.0."""
        converter = PowerConverter()

        with pytest.raises(ValueError):
            converter.convert([2.0, 1.0, 3.0])


class TestOddsRatioConverter:
    """Tests for the OddsRatioConverter implementation."""

    def test_initialization(self):
        """Test that OddsRatioConverter can be instantiated with default parameters."""
        converter = OddsRatioConverter()
        assert isinstance(converter, OddsConverter)
        assert converter.max_iter == 1000
        assert converter.last_or_value == 1.0

    def test_initialization_with_params(self):
        """Test that OddsRatioConverter can be instantiated with custom parameters."""
        converter = OddsRatioConverter(max_iter=2000)
        assert converter.max_iter == 2000

    def test_convert_simple_case(self):
        """Test conversion with simple odds."""
        converter = OddsRatioConverter()
        odds = [2.0, 4.0]
        result = converter.convert(odds)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert math.isclose(result.sum(), 1.0)

        # First probability should be greater than second
        assert result[0] > result[1]

    def test_convert_realistic_case(self):
        """Test conversion with realistic F1 odds."""
        converter = OddsRatioConverter()
        odds = [2.0, 3.0, 5.0, 10.0, 20.0]
        result = converter.convert(odds)

        # Result should sum to 1.0 and be in descending order
        assert math.isclose(result.sum(), 1.0)
        assert np.all(np.diff(result) <= 0)

        # The OR value should be adjusted to fit the target sum
        assert converter.last_or_value != 1.0

    def test_convert_with_custom_target_sum(self):
        """Test conversion with a custom target sum."""
        converter = OddsRatioConverter()
        odds = [2.0, 3.0, 5.0]
        result = converter.convert(odds, target_sum=2.0)

        assert math.isclose(result.sum(), 2.0)

    def test_rejects_invalid_odds(self):
        """Test that converter rejects odds <= 1.0."""
        converter = OddsRatioConverter()

        with pytest.raises(ValueError):
            converter.convert([2.0, 1.0, 3.0])


class TestShinsConverter:
    """Tests for the ShinsConverter implementation."""

    def test_initialization(self):
        """Test that ShinsConverter can be instantiated with default parameters."""
        converter = ShinsConverter()
        assert isinstance(converter, OddsConverter)
        assert converter.max_z == 0.2
        assert converter.last_z_value == 0.01

    def test_initialization_with_params(self):
        """Test that ShinsConverter can be instantiated with custom parameters."""
        converter = ShinsConverter(max_z=0.1)
        assert converter.max_z == 0.1

    def test_convert_simple_case(self):
        """Test conversion with simple odds."""
        converter = ShinsConverter()
        odds = [2.0, 4.0]
        result = converter.convert(odds)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert math.isclose(result.sum(), 1.0)

        # First probability should be greater than second
        assert result[0] > result[1]

        # The z value should be set
        assert converter.last_z_value == 0.02

    def test_convert_realistic_case(self):
        """Test conversion with realistic F1 odds."""
        converter = ShinsConverter()
        odds = [2.0, 3.0, 5.0, 10.0, 20.0]
        result = converter.convert(odds)

        # Result should sum to 1.0 and be in descending order
        assert math.isclose(result.sum(), 1.0)
        assert np.all(np.diff(result) <= 0)

    def test_convert_with_custom_target_sum(self):
        """Test conversion with a custom target sum."""
        converter = ShinsConverter()
        odds = [2.0, 3.0, 5.0]
        result = converter.convert(odds, target_sum=2.0)

        assert math.isclose(result.sum(), 2.0)

    def test_rejects_invalid_odds(self):
        """Test that converter rejects odds <= 1.0."""
        converter = ShinsConverter()

        with pytest.raises(ValueError):
            converter.convert([2.0, 1.0, 3.0])


class TestGetOddsConverter:
    """Tests for the get_odds_converter factory function."""

    def test_get_basic_converter(self):
        """Test that get_odds_converter returns a BasicConverter for 'basic'."""
        converter = get_odds_converter("basic")
        assert isinstance(converter, BasicConverter)

    def test_get_power_converter(self):
        """Test that get_odds_converter returns a PowerConverter for 'power'."""
        converter = get_odds_converter("power")
        assert isinstance(converter, PowerConverter)

    def test_get_odds_ratio_converter(self):
        "Test that get_odds_converter returns an OddsRatioConverter for 'odds_ratio'."
        converter = get_odds_converter("odds_ratio")
        assert isinstance(converter, OddsRatioConverter)

    def test_get_shins_converter(self):
        """Test that get_odds_converter returns a ShinsConverter for 'shin'."""
        converter = get_odds_converter("shin")
        assert isinstance(converter, ShinsConverter)

    def test_get_converter_with_params(self):
        """Test that get_odds_converter passes parameters to the converter."""
        converter = get_odds_converter("power", max_iter=2000)
        assert isinstance(converter, PowerConverter)
        assert converter.max_iter == 2000

    def test_rejects_invalid_method(self):
        """Test that get_odds_converter rejects invalid method names."""
        with pytest.raises(ValueError):
            get_odds_converter("invalid_method")

    def test_default_converter(self):
        """Test that get_odds_converter returns BasicConverter by default."""
        converter = get_odds_converter()
        assert isinstance(converter, BasicConverter)


class TestConverterBehavioral:
    """Behavioral tests that compare converters against each other."""

    @pytest.fixture
    def converters(self):
        """Return a dictionary of all converters."""
        return {
            "basic": BasicConverter(),
            "power": PowerConverter(),
            "odds_ratio": OddsRatioConverter(),
            "shin": ShinsConverter(),
        }

    def test_all_converters_handle_equal_odds(self, converters):
        """Test that all converters handle equal odds identically."""
        odds = [2.0, 2.0, 2.0, 2.0]  # Equal odds
        expected = np.ones(4) / 4

        for name, converter in converters.items():
            result = converter.convert(odds)
            assert np.allclose(result, expected), f"{name} failed with equal odds"

    def test_all_converters_preserve_ordering(self, converters):
        """Test that all converters preserve the ordering of odds."""
        odds = [2.0, 3.0, 5.0, 10.0]  # Ascending odds

        for name, converter in converters.items():
            result = converter.convert(odds)
            # Probabilities should be in descending order (inverse of odds)
            assert np.all(np.diff(result) <= 0), f"{name} didn't preserve ordering"

    def test_all_converters_sum_to_target(self, converters):
        """Test that all converters sum to the target value."""
        odds = [2.5, 3.8, 6.2, 12.0, 25.0]
        targets = [1.0, 2.0, 3.0, 5.0]

        for name, converter in converters.items():
            for target in targets:
                result = converter.convert(odds, target_sum=target)
                assert math.isclose(
                    result.sum(), target
                ), f"{name} didn't sum to {target}"

    def test_compared_behavior(self, converters):
        """Compare the behavior of different converters."""
        odds = [1.5, 3.0, 6.0, 12.0, 26.0]

        results = {}
        for name, converter in converters.items():
            results[name] = converter.convert(odds)

        # All methods should preserve ordering (highest prob for lowest odds)
        for name, probs in results.items():
            assert np.all(
                np.diff(probs) <= 0
            ), f"{name} doesn't maintain probability ordering"

        # Check expected behavior of Shin's converter
        # Shin's method uses a fixed z value of 0.02 and is designed to balance
        # probabilities. Theoretical expectation: raw probs of favorites are reduced,
        # long-shots increased
        raw_probs = np.array([1 / o for o in odds]) / sum([1 / o for o in odds])
        shin_probs = results["shin"]

        # Relative to raw probabilities, Shin should reduce high probabilities
        # and increase low probabilities (the balancing effect)
        assert shin_probs[0] < raw_probs[0], "Shin should reduce favorite probability"
        assert (
            shin_probs[-1] > raw_probs[-1]
        ), "Shin should increase longshot probability"

        # Check that all converters produce valid probabilities
        for name, probs in results.items():
            assert np.all(probs >= 0), f"{name} produced negative probabilities"
            assert np.all(probs <= 1), f"{name} produced probabilities > 1"
            assert math.isclose(
                probs.sum(), 1.0
            ), f"{name} probabilities don't sum to 1.0"


if __name__ == "__main__":
    pytest.main()
