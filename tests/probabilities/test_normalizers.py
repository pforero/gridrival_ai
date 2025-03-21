"""
Tests for grid normalizers.

This module contains tests for the grid normalization classes.
"""

import numpy as np
import pytest

from gridrival_ai.probabilities.normalizers.factory import get_grid_normalizer
from gridrival_ai.probabilities.normalizers.sinkhorn import SinkhornNormalizer


class TestSinkhornNormalizer:
    """Tests for the SinkhornNormalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Return a SinkhornNormalizer instance."""
        return SinkhornNormalizer()

    @pytest.fixture
    def simple_matrix(self):
        """Return a simple 2x2 matrix for testing."""
        return np.array([[0.7, 0.3], [0.4, 0.6]])

    @pytest.fixture
    def uneven_matrix(self):
        """Return a matrix with uneven row sums and column sums."""
        return np.array([[0.8, 0.4], [0.6, 0.2]])

    @pytest.fixture
    def normalized_matrix(self):
        """Return a matrix that's already normalized (rows and columns sum to 1.0)."""
        return np.array([[0.6, 0.4], [0.4, 0.6]])

    def test_normalization_simple(self, normalizer, simple_matrix):
        """Test normalization of a simple matrix."""
        # Column sums before normalization
        p1_before = simple_matrix[:, 0].sum()
        p2_before = simple_matrix[:, 1].sum()
        assert not np.isclose(p1_before, 1.0) or not np.isclose(p2_before, 1.0)

        normalized = normalizer.normalize(simple_matrix)

        # Check each column now sums to 1.0
        p1_after = normalized[:, 0].sum()
        p2_after = normalized[:, 1].sum()
        assert np.isclose(p1_after, 1.0)
        assert np.isclose(p2_after, 1.0)

        # Check each row also sums to 1.0
        row1_sum = normalized[0, :].sum()
        row2_sum = normalized[1, :].sum()
        assert np.isclose(row1_sum, 1.0)
        assert np.isclose(row2_sum, 1.0)

    def test_preserves_normalized_input(self, normalizer, normalized_matrix):
        """Test that an already normalized matrix is preserved by normalization."""
        # Verify the input matrix is already normalized
        assert np.allclose(normalized_matrix.sum(axis=0), 1.0)
        assert np.allclose(normalized_matrix.sum(axis=1), 1.0)

        # Normalize the matrix
        result = normalizer.normalize(normalized_matrix)

        # The result should be very close to the input
        assert np.allclose(result, normalized_matrix, rtol=1e-5, atol=1e-5)

    def test_normalization_uneven(self, normalizer, uneven_matrix):
        """Test normalization of a matrix with uneven sums."""
        # Sums before normalization
        row_sums_before = uneven_matrix.sum(axis=1)
        col_sums_before = uneven_matrix.sum(axis=0)

        assert not np.allclose(row_sums_before, 1.0)
        assert not np.allclose(col_sums_before, 1.0)

        normalized = normalizer.normalize(uneven_matrix)

        # Check sums after normalization
        row_sums_after = normalized.sum(axis=1)
        col_sums_after = normalized.sum(axis=0)

        assert np.allclose(row_sums_after, 1.0)
        assert np.allclose(col_sums_after, 1.0)

    def test_empty_matrix(self, normalizer):
        """Test normalization of an empty matrix."""
        empty_matrix = np.array([])
        result = normalizer.normalize(empty_matrix)
        assert result.size == 0

    def test_non_square_matrix(self, normalizer):
        """Test that non-square matrices are rejected."""
        non_square = np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]])  # 2x3 matrix

        with pytest.raises(ValueError) as excinfo:
            normalizer.normalize(non_square)

        assert "square" in str(excinfo.value)

    def test_negative_values(self, normalizer):
        """Test that matrices with negative values are rejected."""
        negative_values = np.array([[0.7, 0.3], [-0.1, 1.1]])

        with pytest.raises(ValueError) as excinfo:
            normalizer.normalize(negative_values)

        assert "between 0 and 1" in str(excinfo.value)

    def test_values_greater_than_one(self, normalizer):
        """Test that matrices with values > 1 are rejected."""
        large_values = np.array([[0.7, 0.3], [1.1, -0.1]])

        with pytest.raises(ValueError) as excinfo:
            normalizer.normalize(large_values)

        assert "between 0 and 1" in str(excinfo.value)

    def test_max_iterations(self):
        """Test that the algorithm converges within max_iter iterations."""
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])

        # Set a very small tolerance to force more iterations
        normalizer = SinkhornNormalizer(max_iter=3, tolerance=1e-10)

        # The algorithm should stop after max_iter even if not converged
        normalized = normalizer.normalize(matrix)

        # Check the result is still close to a valid normalized matrix
        row_sums = normalized.sum(axis=1)
        col_sums = normalized.sum(axis=0)

        # May not be exactly 1.0 due to early stopping
        assert np.all(np.abs(row_sums - 1.0) < 0.1)
        assert np.all(np.abs(col_sums - 1.0) < 0.1)


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
