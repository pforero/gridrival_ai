"""Tests for odds conversion utilities."""

import numpy as np
import pytest

from gridrival_ai.probabilities.conversion import (
    BasicConverter,
    ConverterFactory,
    HarvilleConverter,
    OddsConverter,
    OddsRatioConverter,
    PowerConverter,
    ShinsConverter,
    odds_to_distributions,
    odds_to_grid,
    odds_to_position_distribution,
)
from gridrival_ai.probabilities.distributions import PositionDistribution


class TestConverterFactory:
    """Test suite for ConverterFactory."""

    def test_get_basic(self):
        """Test getting basic converter."""
        converter = ConverterFactory.get("basic")
        assert isinstance(converter, BasicConverter)

    def test_get_all_converters(self):
        """Test getting all converters."""
        methods = ["basic", "odds_ratio", "shin", "power", "harville"]
        expected_types = [
            BasicConverter,
            OddsRatioConverter,
            ShinsConverter,
            PowerConverter,
            HarvilleConverter,
        ]

        for method, expected_type in zip(methods, expected_types):
            converter = ConverterFactory.get(method)
            assert isinstance(converter, expected_type)

    def test_get_invalid_method(self):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown conversion method"):
            ConverterFactory.get("invalid_method")

    def test_register_custom_converter(self):
        """Test registering custom converter."""

        class CustomConverter(OddsConverter):
            def convert(self, odds, target_sum=1.0):
                return np.ones(len(odds)) / len(odds)

        # Register custom converter
        ConverterFactory.register("custom", CustomConverter)

        # Get and use custom converter
        converter = ConverterFactory.get("custom")
        odds = [2.0, 4.0]
        probs = converter.convert(odds)

        # Should return uniform distribution
        assert np.allclose(probs, [0.5, 0.5])

    def test_register_invalid_converter(self):
        """Test error on registering invalid converter."""

        class NotAConverter:
            pass

        with pytest.raises(TypeError, match="must be a subclass of OddsConverter"):
            ConverterFactory.register("invalid", NotAConverter)


class TestAdvancedConverters:
    """Test suite for advanced converters."""

    def test_odds_ratio_converter(self):
        """Test odds ratio converter."""
        converter = OddsRatioConverter()
        odds = [1.5, 3.0, 6.0]
        probs = converter.convert(odds)

        # Check that probabilities sum to 1.0
        assert np.sum(probs) == pytest.approx(1.0)

        # Check all probabilities are positive
        assert np.all(probs > 0)

        # Check order is preserved (lower odds = higher probability)
        assert probs[0] > probs[1] > probs[2]

    def test_shins_converter(self):
        """Test Shin's converter."""
        converter = ShinsConverter()
        odds = [1.5, 3.0, 6.0]
        probs = converter.convert(odds)

        # Check that probabilities sum to 1.0
        assert np.sum(probs) == pytest.approx(1.0)

        # Check all probabilities are positive
        assert np.all(probs > 0)

        # Check order is preserved (lower odds = higher probability)
        assert probs[0] > probs[1] > probs[2]

    def test_power_converter(self):
        """Test power converter."""
        converter = PowerConverter()
        odds = [1.5, 3.0, 6.0]
        probs = converter.convert(odds)

        # Check that probabilities sum to 1.0
        assert np.sum(probs) == pytest.approx(1.0)

        # Check all probabilities are positive
        assert np.all(probs > 0)

        # Check order is preserved (lower odds = higher probability)
        assert probs[0] > probs[1] > probs[2]


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_odds_to_position_distribution(self):
        """Test odds_to_position_distribution utility."""
        odds = [1.5, 3.0, 6.0]
        dist = odds_to_position_distribution(odds)

        # Check result type
        assert isinstance(dist, PositionDistribution)

        # Check values match BasicConverter
        converter = BasicConverter()
        expected = converter.to_position_distribution(odds)
        assert dist[1] == expected[1]
        assert dist[2] == expected[2]
        assert dist[3] == expected[3]

    def test_odds_to_position_distribution_method(self):
        """Test odds_to_position_distribution with different methods."""
        odds = [1.5, 3.0, 6.0]

        # Check basic method
        dist_basic = odds_to_position_distribution(odds, method="basic")

        # Check another method
        dist_shin = odds_to_position_distribution(odds, method="shin")

        # Result should be different but still valid
        assert dist_shin.is_valid
        assert dist_shin[1] != dist_basic[1]

    def test_odds_to_grid(self):
        """Test odds_to_grid utility."""
        odds = [1.5, 3.0]
        driver_ids = ["VER", "HAM"]
        grid = odds_to_grid(odds, driver_ids)

        # Check grid structure
        assert set(grid.keys()) == set(driver_ids)

        # Check probabilities
        assert sum(grid["VER"].values()) == pytest.approx(1.0)
        assert sum(grid["HAM"].values()) == pytest.approx(1.0)

    def test_odds_to_distributions(self):
        """Test odds_to_distributions utility."""
        odds = [1.5, 3.0]
        driver_ids = ["VER", "HAM"]
        dists = odds_to_distributions(odds, driver_ids)

        # Check result type
        assert isinstance(dists, dict)
        assert set(dists.keys()) == set(driver_ids)
        assert all(isinstance(dist, PositionDistribution) for dist in dists.values())

        # Check probabilities
        assert dists["VER"].is_valid
        assert dists["HAM"].is_valid
