"""
Tests for grid normalizers.

This module contains tests for the grid normalization classes.
"""

import numpy as np
import pytest

from gridrival_ai.probabilities.distributions import (
    PositionDistribution,
    SessionDistribution,
)
from gridrival_ai.probabilities.normalizers.factory import get_grid_normalizer
from gridrival_ai.probabilities.normalizers.sinkhorn import SinkhornNormalizer


class TestSinkhornNormalizer:
    """Tests for the SinkhornNormalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Return a SinkhornNormalizer instance."""
        return SinkhornNormalizer()

    @pytest.fixture
    def simple_distributions(self):
        """Return a simple set of distributions for testing."""
        driver_distributions = {
            "VER": PositionDistribution({1: 0.7, 2: 0.3}),
            "HAM": PositionDistribution({1: 0.4, 2: 0.6}),
        }
        return SessionDistribution(driver_distributions, "race")

    @pytest.fixture
    def edge_case_distributions(self):
        """Return distributions with edge cases."""
        driver_distributions = {
            "VER": PositionDistribution({1: 0.6, 2: 0.3, 3: 0.1}),  # Has P3
            "HAM": PositionDistribution({1: 0.4, 2: 0.6}),  # No P3
        }
        return SessionDistribution(driver_distributions, "race")

    def test_normalization_simple(self, normalizer, simple_distributions):
        """Test normalization of simple distributions."""
        # P1 sum = 1.1, P2 sum = 0.9 before normalization
        p1_before = (
            simple_distributions.get_driver_distribution("VER")[1]
            + simple_distributions.get_driver_distribution("HAM")[1]
        )
        p2_before = (
            simple_distributions.get_driver_distribution("VER")[2]
            + simple_distributions.get_driver_distribution("HAM")[2]
        )
        assert not np.isclose(p1_before, 1.0)
        assert not np.isclose(p2_before, 1.0)

        normalized = normalizer.normalize(simple_distributions)

        # Check each position now sums to 1.0 across drivers
        p1_after = (
            normalized.get_driver_distribution("VER")[1]
            + normalized.get_driver_distribution("HAM")[1]
        )
        p2_after = (
            normalized.get_driver_distribution("VER")[2]
            + normalized.get_driver_distribution("HAM")[2]
        )
        assert np.isclose(p1_after, 1.0)
        assert np.isclose(p2_after, 1.0)

        # Check each driver's distribution still sums to 1.0 (or close to it)
        ver_sum = sum(normalized.get_driver_distribution("VER").position_probs.values())
        ham_sum = sum(normalized.get_driver_distribution("HAM").position_probs.values())
        assert np.isclose(ver_sum, 1.0)
        assert np.isclose(ham_sum, 1.0)


class TestGridNormalizerFactory:
    """Tests for the grid normalizer factory."""

    def test_get_default_normalizer(self):
        """Test getting the default normalizer."""
        normalizer = get_grid_normalizer()
        assert isinstance(normalizer, SinkhornNormalizer)

    def test_get_sinkhorn_normalizer(self):
        """Test getting the Sinkhorn normalizer explicitly."""
        normalizer = get_grid_normalizer("sinkhorn")
        assert isinstance(normalizer, SinkhornNormalizer)

    def test_get_normalizer_with_params(self):
        """Test getting a normalizer with custom parameters."""
        normalizer = get_grid_normalizer("sinkhorn", max_iter=50, tolerance=1e-8)
        assert isinstance(normalizer, SinkhornNormalizer)
        assert normalizer.max_iter == 50
        assert normalizer.tolerance == 1e-8

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError):
            get_grid_normalizer("invalid_method")
