import numpy as np
import pytest

from gridrival_ai.probabilities.converters import BasicConverter


class TestBasicConverter:
    """Test suite for BasicConverter."""

    def test_convert_basic(self):
        """Test basic odds conversion."""
        converter = BasicConverter()
        odds = [2.0, 4.0]
        probs = converter.convert(odds)

        # Check that probabilities sum to 1.0
        assert np.sum(probs) == pytest.approx(1.0)

        # Check values (1/2.0 = 0.5, 1/4.0 = 0.25, normalized to 1.0)
        expected = np.array([0.5, 0.25]) / 0.75  # Sum of raw probs is 0.75
        assert np.allclose(probs, expected)

    def test_convert_target_sum(self):
        """Test conversion with custom target sum."""
        converter = BasicConverter()
        odds = [2.0, 4.0]
        probs = converter.convert(odds, target_sum=3.0)

        # Check that probabilities sum to target
        assert np.sum(probs) == pytest.approx(3.0)

    def test_invalid_odds(self):
        """Test error on invalid odds."""
        converter = BasicConverter()
        with pytest.raises(ValueError, match="All odds must be greater than 1.0"):
            converter.convert([0.5, 2.0])

    def test_multiple_drivers(self):
        """Test conversion with multiple drivers."""
        converter = BasicConverter()
        odds = [1.5, 3.0, 6.0, 10.0, 15.0]
        probs = converter.convert(odds)

        # Check sum
        assert np.sum(probs) == pytest.approx(1.0)

        # Check individual values
        raw_probs = np.array([1 / 1.5, 1 / 3.0, 1 / 6.0, 1 / 10.0, 1 / 15.0])
        expected = raw_probs / raw_probs.sum()
        assert np.allclose(probs, expected)

    def test_identical_odds(self):
        """Test conversion with identical odds."""
        converter = BasicConverter()
        odds = [3.0, 3.0, 3.0]
        probs = converter.convert(odds)

        # All probabilities should be equal
        assert np.allclose(probs, np.array([1 / 3, 1 / 3, 1 / 3]))

    def test_extreme_odds_range(self):
        """Test conversion with extreme odds range."""
        converter = BasicConverter()
        odds = [1.1, 100.0]  # Very likely vs very unlikely
        probs = converter.convert(odds)

        # Check values
        raw_probs = np.array([1 / 1.1, 1 / 100.0])
        expected = raw_probs / raw_probs.sum()
        assert np.allclose(probs, expected)

        # The first probability should be much higher
        assert probs[0] > 0.98

    def test_zero_target_sum(self):
        """Test conversion with zero target sum."""
        converter = BasicConverter()
        odds = [2.0, 4.0]
        probs = converter.convert(odds, target_sum=0.0)

        # All probabilities should be zero
        assert np.allclose(probs, np.zeros(2))
