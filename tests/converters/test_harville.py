"""
Tests for the harville module.

This module contains tests for the HarvilleConverter class, verifying its ability
to convert odds to position probability distributions using the Harville method.
"""

import numpy as np
import pytest

from gridrival_ai.probabilities.converters.harville import HarvilleConverter
from gridrival_ai.probabilities.core import PositionDistribution


class TestHarvilleConverter:
    """Test suite for HarvilleConverter."""

    def test_convert_basic(self):
        """Test basic conversion to win probabilities."""
        converter = HarvilleConverter()
        odds = [1.5, 3.0]
        probs = converter.convert(odds)

        # Check that probabilities sum to 1.0
        assert np.sum(probs) == pytest.approx(1.0)

        # Check values
        expected = np.array([1 / 1.5, 1 / 3.0]) / (1 / 1.5 + 1 / 3.0)
        assert np.allclose(probs, expected)

    def test_convert_to_grid_three_drivers(self):
        """Test conversion to grid with three drivers to catch accumulation issues."""
        converter = HarvilleConverter()
        odds = [1.5, 3.0, 6.0]
        grid = converter.convert_to_grid(odds, ["VER", "HAM", "NOR"])

        # Check grid structure
        assert set(grid.keys()) == {"VER", "HAM", "NOR"}
        assert set(grid["VER"].keys()) == {1, 2, 3}
        assert set(grid["HAM"].keys()) == {1, 2, 3}
        assert set(grid["NOR"].keys()) == {1, 2, 3}

        # Check probabilities sum to EXACTLY 1.0 for each driver (strict tolerance)
        for driver in grid:
            driver_sum = sum(grid[driver].values())
            assert driver_sum == pytest.approx(
                1.0, abs=1e-9
            ), f"Sum for {driver}: {driver_sum}"

        # Check probabilities sum to EXACTLY 1.0 for each position (strict tolerance)
        for pos in range(1, 4):
            pos_sum = sum(grid[driver][pos] for driver in grid if pos in grid[driver])
            assert pos_sum == pytest.approx(
                1.0, abs=1e-9
            ), f"Sum for position {pos}: {pos_sum}"

        # Check expected ordering patterns (lower odds → higher probability of better
        # positions)
        assert grid["VER"][1] > grid["HAM"][1] > grid["NOR"][1]  # Order in P1
        assert grid["NOR"][3] > grid["HAM"][3] > grid["VER"][3]  # Order in P3

        # Verify consistent transition probabilities
        # With exponential decay in position probabilities:
        # - First driver (lowest odds) should have P1 > P2 > P3
        # - Last driver (highest odds) should have P3 > P2 > P1
        assert grid["VER"][1] > grid["VER"][2] > grid["VER"][3]
        assert grid["NOR"][3] > grid["NOR"][2] > grid["NOR"][1]

    def test_convert_to_grid(self):
        """Test conversion to full grid."""
        converter = HarvilleConverter()
        odds = [1.5, 3.0]
        grid = converter.convert_to_grid(odds, ["VER", "HAM"])

        # Check grid structure
        assert set(grid.keys()) == {"VER", "HAM"}
        assert set(grid["VER"].keys()) == {1, 2}
        assert set(grid["HAM"].keys()) == {1, 2}

        # Check probabilities sum to 1.0 for each driver
        assert sum(grid["VER"].values()) == pytest.approx(1.0)
        assert sum(grid["HAM"].values()) == pytest.approx(1.0)

        # Check probabilities sum to 1.0 for each position
        assert grid["VER"][1] + grid["HAM"][1] == pytest.approx(1.0)
        assert grid["VER"][2] + grid["HAM"][2] == pytest.approx(1.0)

        # Check values based on expected pattern
        assert grid["VER"][1] > grid["VER"][2]  # VER more likely P1 than P2
        assert grid["HAM"][2] > grid["HAM"][1]  # HAM more likely P2 than P1

    def test_convert_to_grid_no_ids(self):
        """Test conversion to grid without driver IDs."""
        converter = HarvilleConverter()
        odds = [1.5, 3.0]
        grid = converter.convert_to_grid(odds)

        # Should use string numbers as IDs
        assert set(grid.keys()) == {"1", "2"}

    def test_grid_to_position_distributions(self):
        """Test converting grid to position distributions."""
        converter = HarvilleConverter()
        odds = [1.5, 3.0]
        grid = converter.convert_to_grid(odds, ["VER", "HAM"])
        dists = converter.grid_to_position_distributions(grid)

        # Check result type
        assert isinstance(dists, dict)
        assert all(isinstance(dist, PositionDistribution) for dist in dists.values())

        # Check values match grid
        assert dists["VER"][1] == grid["VER"][1]
        assert dists["VER"][2] == grid["VER"][2]
        assert dists["HAM"][1] == grid["HAM"][1]
        assert dists["HAM"][2] == grid["HAM"][2]

    def test_invalid_driver_ids(self):
        """Test error on mismatched driver IDs."""
        converter = HarvilleConverter()
        odds = [1.5, 3.0]
        with pytest.raises(ValueError, match="Length of driver_ids"):
            converter.convert_to_grid(odds, ["VER"])  # Too few IDs

    def test_invalid_odds(self):
        """Test that invalid odds raise appropriate errors."""
        converter = HarvilleConverter()

        # Test odds <= 1.0
        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 1.0, 3.0])

        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([2.0, 0.5, 3.0])

    def test_consistency_different_ids(self):
        """Test that results are consistent regardless of ID format."""
        converter = HarvilleConverter()
        odds = [1.5, 3.0, 6.0]

        # Try with string IDs
        grid1 = converter.convert_to_grid(odds, ["VER", "HAM", "NOR"])

        # Try with numeric IDs
        grid2 = converter.convert_to_grid(odds, [1, 2, 3])

        # Values should be identical regardless of ID type
        for i, driver in enumerate(["VER", "HAM", "NOR"]):
            for pos in range(1, 4):
                assert grid1[driver][pos] == grid2[i + 1][pos]

    def test_edge_case_identical_odds(self):
        """Test behavior with identical odds for all drivers."""
        converter = HarvilleConverter()
        odds = [2.0, 2.0, 2.0]
        grid = converter.convert_to_grid(odds, ["VER", "HAM", "NOR"])

        # All drivers should have identical probability distributions
        for pos in range(1, 4):
            assert grid["VER"][pos] == grid["HAM"][pos] == grid["NOR"][pos]

        # Each position should be equally likely for each driver
        assert grid["VER"][1] == pytest.approx(1 / 3)
        assert grid["VER"][2] == pytest.approx(1 / 3)
        assert grid["VER"][3] == pytest.approx(1 / 3)
