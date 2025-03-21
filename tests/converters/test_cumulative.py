import math

import pytest

from gridrival_ai.probabilities.conversion import (
    ConverterFactory,
    CumulativeMarketConverter,
    HarvilleConverter,
)
from gridrival_ai.probabilities.core import PositionDistribution
from gridrival_ai.probabilities.factory import DistributionFactory


class TestCumulativeMarketConverter:
    """Test suite for CumulativeMarketConverter."""

    @pytest.fixture
    def valid_cumulative_probs(self):
        """Fixture with valid cumulative probabilities for testing."""
        return {
            1: {"VER": 0.4, "HAM": 0.3, "NOR": 0.2, "PIA": 0.1},  # Win probs
            3: {"VER": 0.7, "HAM": 0.6, "NOR": 0.5, "PIA": 0.2},  # Top-3 probs
            6: {"VER": 0.9, "HAM": 0.8, "NOR": 0.7, "PIA": 0.6},  # Top-6 probs
        }

    @pytest.fixture
    def win_only_probs(self):
        """Fixture with win-only probabilities."""
        return {
            1: {"VER": 0.4, "HAM": 0.3, "NOR": 0.2, "PIA": 0.1},  # Win probs
        }

    @pytest.fixture
    def converter(self):
        """Fixture providing a standard converter instance."""
        return CumulativeMarketConverter(
            max_position=20,
            baseline_method="exponential",
            baseline_params={"decay": 0.5},
            fallback_converter=HarvilleConverter(),
        )

    def test_basic_conversion(self, converter, valid_cumulative_probs):
        """Test basic conversion with valid inputs."""
        # Convert to distributions
        distributions = converter.convert_all(valid_cumulative_probs)

        # Check that we get distributions for all drivers
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "PIA"}

        # Check that all distributions are valid
        for driver, dist in distributions.items():
            assert isinstance(dist, PositionDistribution)
            assert dist.is_valid

        # Check that win probabilities match the input
        assert math.isclose(distributions["VER"][1], 0.4, abs_tol=1e-6)
        assert math.isclose(distributions["HAM"][1], 0.3, abs_tol=1e-6)
        assert math.isclose(distributions["NOR"][1], 0.2, abs_tol=1e-6)
        assert math.isclose(distributions["PIA"][1], 0.1, abs_tol=1e-6)

        # Check cumulative constraints
        for driver, dist in distributions.items():
            cum_probs = dist.cumulative()

            # Win probability (position 1)
            assert math.isclose(
                cum_probs[1], valid_cumulative_probs[1][driver], abs_tol=1e-6
            )

            # Top-3 probability (positions 1-3)
            top3_prob = sum(dist[pos] for pos in range(1, 4))
            assert math.isclose(
                top3_prob, valid_cumulative_probs[3][driver], abs_tol=1e-6
            )

            # Top-6 probability (positions 1-6)
            top6_prob = sum(dist[pos] for pos in range(1, 7))
            assert math.isclose(
                top6_prob, valid_cumulative_probs[6][driver], abs_tol=1e-6
            )

    def test_win_only_conversion(self, converter, win_only_probs):
        """Test conversion with win-only probabilities."""
        # Convert to distributions
        distributions = converter.convert_all(win_only_probs)

        # Check that we get distributions for all drivers
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "PIA"}

        # Check that all distributions are valid
        for driver, dist in distributions.items():
            assert isinstance(dist, PositionDistribution)
            assert dist.is_valid

        # Check that win probabilities match the input
        assert math.isclose(distributions["VER"][1], 0.4, abs_tol=1e-6)
        assert math.isclose(distributions["HAM"][1], 0.3, abs_tol=1e-6)
        assert math.isclose(distributions["NOR"][1], 0.2, abs_tol=1e-6)
        assert math.isclose(distributions["PIA"][1], 0.1, abs_tol=1e-6)

        # Check that all positions sum to 1.0 for each driver
        for driver, dist in distributions.items():
            pos_sum = sum(dist[p] for p in range(1, 21))
            assert math.isclose(pos_sum, 1.0, abs_tol=1e-6)

    def test_consistency_validation(self, converter, valid_cumulative_probs):
        """Test that distributions are consistent across positions."""
        distributions = converter.convert_all(valid_cumulative_probs)

        # For each position, the sum across all drivers should be close to 1.0
        for pos in range(1, 11):  # Check top 10 positions
            pos_sum = sum(dist[pos] for dist in distributions.values())
            # Allow some tolerance since we're not enforcing exact constraints
            assert 0.9 <= pos_sum <= 1.1, f"Position {pos} sum is {pos_sum}"

    def test_different_baseline_methods(self, valid_cumulative_probs):
        """Test different baseline weighting methods."""
        # Create converters with different baseline methods
        exponential_converter = CumulativeMarketConverter(
            baseline_method="exponential", fallback_converter=HarvilleConverter()
        )
        linear_converter = CumulativeMarketConverter(
            baseline_method="linear", fallback_converter=HarvilleConverter()
        )
        uniform_converter = CumulativeMarketConverter(
            baseline_method="uniform", fallback_converter=HarvilleConverter()
        )

        # Convert with each method
        exp_distributions = exponential_converter.convert_all(valid_cumulative_probs)
        lin_distributions = linear_converter.convert_all(valid_cumulative_probs)
        uni_distributions = uniform_converter.convert_all(valid_cumulative_probs)

        # Check that all produce valid distributions
        for dist_set in [exp_distributions, lin_distributions, uni_distributions]:
            for driver, dist in dist_set.items():
                assert isinstance(dist, PositionDistribution)
                assert dist.is_valid

        # Distributions should differ in some positions but match at constraints
        ver_exp = exp_distributions["VER"]
        ver_lin = lin_distributions["VER"]
        ver_uni = uni_distributions["VER"]

        # Win probability should match for all
        assert math.isclose(ver_exp[1], 0.4, abs_tol=1e-6)
        assert math.isclose(ver_lin[1], 0.4, abs_tol=1e-6)
        assert math.isclose(ver_uni[1], 0.4, abs_tol=1e-6)

        # Top-3 and Top-6 should also match
        assert math.isclose(sum(ver_exp[p] for p in range(1, 4)), 0.7, abs_tol=1e-6)
        assert math.isclose(sum(ver_lin[p] for p in range(1, 4)), 0.7, abs_tol=1e-6)
        assert math.isclose(sum(ver_uni[p] for p in range(1, 4)), 0.7, abs_tol=1e-6)

        assert math.isclose(sum(ver_exp[p] for p in range(1, 7)), 0.9, abs_tol=1e-6)
        assert math.isclose(sum(ver_lin[p] for p in range(1, 7)), 0.9, abs_tol=1e-6)
        assert math.isclose(sum(ver_uni[p] for p in range(1, 7)), 0.9, abs_tol=1e-6)

        # But intermediate positions should differ
        distribution_values = [
            [ver_exp[2], ver_lin[2], ver_uni[2]],  # Position 2
            [ver_exp[4], ver_lin[4], ver_uni[4]],  # Position 4
        ]

        for values in distribution_values:
            # Check that not all values are identical
            # some method produced different results)
            assert (
                len(set([round(v, 4) for v in values])) > 1
            ), f"All methods produced identical values: {values}"

    def test_error_handling(self, converter):
        """Test error handling with invalid inputs."""
        # Empty cumulative probs
        with pytest.raises(ValueError):
            converter.convert_all({})

        # Missing win probability
        invalid_probs = {
            3: {"VER": 0.7, "HAM": 0.6},  # Only Top-3 probs
            6: {"VER": 0.9, "HAM": 0.8},  # Only Top-6 probs
        }
        with pytest.raises(ValueError):
            converter.convert_all(invalid_probs)

        # Non-increasing probabilities
        invalid_probs = {
            1: {"VER": 0.4},
            3: {"VER": 0.3},  # Lower than position 1
        }
        # This should not raise but produce a warning about non-increasing probabilities
        distributions = converter.convert_all(invalid_probs)
        assert "VER" in distributions
        assert distributions["VER"].is_valid

    def test_integration_with_factory(self, valid_cumulative_probs):
        """Test integration with ConverterFactory."""
        # Register with ConverterFactory
        ConverterFactory.register("cumulative", CumulativeMarketConverter)

        # Get via factory
        converter = ConverterFactory.get(
            "cumulative", max_position=20, baseline_method="exponential"
        )

        # Test that it works
        distributions = converter.convert_all(valid_cumulative_probs)
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "PIA"}

        # Test with the utility method (mock implementation)
        DistributionFactory.from_cumulative_markets = staticmethod(
            lambda cumulative_probs, **kwargs: CumulativeMarketConverter(
                **kwargs
            ).convert_all(cumulative_probs)
        )

        distributions = DistributionFactory.from_cumulative_markets(
            valid_cumulative_probs
        )
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "PIA"}

    def test_fallback_converter_handling(self, win_only_probs):
        """Test handling of fallback converter."""
        # Create converter with no fallback
        converter_no_fallback = CumulativeMarketConverter(fallback_converter=None)

        # Should still work but use simple distribution
        distributions = converter_no_fallback.convert_all(win_only_probs)
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "PIA"}

        # Create converter with fallback
        converter_with_fallback = CumulativeMarketConverter(
            fallback_converter=HarvilleConverter()
        )

        # Should use Harville method
        distributions = converter_with_fallback.convert_all(win_only_probs)
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "PIA"}

        # Compare results - Harville should give different distributions
        ver_no_fallback = distributions["VER"]
        ver_with_fallback = distributions["VER"]

        # Win probability should still match input
        assert math.isclose(ver_no_fallback[1], 0.4, abs_tol=1e-6)
        assert math.isclose(ver_with_fallback[1], 0.4, abs_tol=1e-6)

    def test_real_world_conversion(self):
        """Test with realistic F1 betting odds."""
        # Raw decimal betting odds for multiple markets
        raw_betting_odds = {
            1: {  # Win market
                "VER": 2.2,  # 2.2 decimal odds = 45.5% implied probability
                "HAM": 4.0,  # 4.0 decimal odds = 25.0% implied probability
                "NOR": 7.0,  # 7.0 decimal odds = 14.3% implied probability
                "LEC": 13.0,  # 13.0 decimal odds = 7.7% implied probability
            },
            3: {  # Top-3 finish
                "VER": 1.2,  # 1.2 decimal odds = 83.3% implied probability
                "HAM": 1.5,  # 1.5 decimal odds = 66.7% implied probability
                "NOR": 1.8,  # 1.8 decimal odds = 55.6% implied probability
                "LEC": 2.5,  # 2.5 decimal odds = 40.0% implied probability
            },
            6: {  # Top-6 finish
                "VER": 1.05,  # 1.05 decimal odds = 95.2% implied probability
                "HAM": 1.15,  # 1.15 decimal odds = 87.0% implied probability
                "NOR": 1.2,  # 1.2 decimal odds = 83.3% implied probability
                "LEC": 1.33,  # 1.33 decimal odds = 75.2% implied probability
            },
        }

        # Convert to implied probabilities
        implied_probs = {}
        for market, positions in raw_betting_odds.items():
            implied_probs[market] = {}
            for driver, odds in positions.items():
                implied_probs[market][driver] = 1.0 / odds

        # Calculate overround for each market
        overrounds = {}
        for market, probabilities in implied_probs.items():
            overrounds[market] = sum(probabilities.values())

        # Remove margin proportionally
        demargin_probs = {}
        for market, probabilities in implied_probs.items():
            demargin_probs[market] = {}
            for driver, prob in probabilities.items():
                demargin_probs[market][driver] = prob / overrounds[market] * market

        # Create converter
        converter = CumulativeMarketConverter(fallback_converter=HarvilleConverter())

        # Convert
        distributions = converter.convert_all(demargin_probs)

        # Check results
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "LEC"}

        # All distributions should be valid
        for driver, dist in distributions.items():
            assert isinstance(dist, PositionDistribution)
            assert dist.is_valid

        # Win probabilities should match de-margined input
        for driver in distributions:
            assert math.isclose(
                distributions[driver][1], demargin_probs[1][driver], abs_tol=1e-6
            )

        # Cumulative probabilities should match constraints
        for market in [3, 6]:
            for driver in distributions:
                cum_prob = sum(
                    distributions[driver][pos] for pos in range(1, market + 1)
                )
                assert math.isclose(
                    cum_prob, demargin_probs[market][driver], abs_tol=1e-6
                )

    def test_missing_drivers(self, converter):
        """Test handling drivers missing from some markets."""
        # Driver RUS only has win and top-6 probabilities
        partial_probs = {
            1: {"VER": 0.4, "HAM": 0.3, "NOR": 0.2, "RUS": 0.1},
            3: {"VER": 0.7, "HAM": 0.6, "NOR": 0.5},  # RUS missing
            6: {"VER": 0.9, "HAM": 0.8, "NOR": 0.7, "RUS": 0.6},
        }

        # Should still work
        distributions = converter.convert_all(partial_probs)
        assert set(distributions.keys()) == {"VER", "HAM", "NOR", "RUS"}

        # VER, HAM, NOR should match all constraints
        for driver in ["VER", "HAM", "NOR"]:
            # Win probability
            assert math.isclose(
                distributions[driver][1], partial_probs[1][driver], abs_tol=1e-6
            )

            # Top-3 probability
            top3_prob = sum(distributions[driver][pos] for pos in range(1, 4))
            assert math.isclose(top3_prob, partial_probs[3][driver], abs_tol=1e-6)

            # Top-6 probability
            top6_prob = sum(distributions[driver][pos] for pos in range(1, 7))
            assert math.isclose(top6_prob, partial_probs[6][driver], abs_tol=1e-6)

        # RUS should match win and top-6 constraints
        assert math.isclose(
            distributions["RUS"][1], partial_probs[1]["RUS"], abs_tol=1e-6
        )

        top6_prob = sum(distributions["RUS"][pos] for pos in range(1, 7))
        assert math.isclose(top6_prob, partial_probs[6]["RUS"], abs_tol=1e-6)


if __name__ == "__main__":
    # Run tests manually
    pytest.main(["-xvs", __file__])
