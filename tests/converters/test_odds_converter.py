from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gridrival_ai.probabilities.converters.odds_converter import OddsConverter


class ConcreteConverter(OddsConverter):
    """Concrete implementation for testing the abstract base class."""

    def convert(self, odds, target_sum=1.0):
        """Simple implementation that normalizes inverse odds."""
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be > 1.0")

        raw_probs = np.array([1 / o for o in odds])
        return raw_probs * (target_sum / raw_probs.sum())


class TestOddsConverter:
    def setup_method(self):
        self.converter = ConcreteConverter()
        self.sample_odds = [2.0, 4.0, 10.0]  # 50%, 25%, 10% before normalization

    def test_convert_abstract_method(self):
        """Test that the abstract method can't be instantiated directly."""
        with pytest.raises(TypeError):
            OddsConverter()

    def test_convert_to_dict(self):
        """Test conversion of odds to position-probability dictionary."""
        result = self.converter.convert_to_dict(self.sample_odds)

        # Check dictionary structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2, 3}

        # Check values sum to 1.0 (allowing for floating point precision)
        assert pytest.approx(sum(result.values()), 1e-10) == 1.0

        # Check relative magnitudes (P1 > P2 > P3)
        assert result[1] > result[2] > result[3]

        # Check specific values based on our concrete implementation
        expected = {
            1: 0.5882352941176471,
            2: 0.29411764705882354,
            3: 0.11764705882352941,
        }
        for pos, prob in expected.items():
            assert pytest.approx(result[pos], 1e-10) == prob

    def test_convert_with_custom_target_sum(self):
        """Test conversion with a non-default target sum."""
        target_sum = 1.5  # For markets like "Top 3 finish" probabilities can sum to >1
        result_dict = self.converter.convert_to_dict(self.sample_odds, target_sum)

        # Check sum of probabilities
        assert pytest.approx(sum(result_dict.values()), 1e-10) == target_sum

    def test_to_position_distribution(self):
        """Test conversion to PositionDistribution object."""
        with patch(
            "gridrival_ai.probabilities.converters.odds_converter.PositionDistribution"
        ) as mock_dist:
            mock_instance = MagicMock()
            mock_dist.return_value = mock_instance

            # Call the method
            result = self.converter.to_position_distribution(self.sample_odds)

            # Check PositionDistribution was created with correct parameters
            expected_probs = {
                1: 0.5882352941176471,
                2: 0.29411764705882354,
                3: 0.11764705882352941,
            }
            mock_dist.assert_called_once()
            call_args = mock_dist.call_args[0][0]

            # Check the probabilities dict passed to PositionDistribution
            for pos, prob in expected_probs.items():
                assert pytest.approx(call_args[pos], 1e-10) == prob

            # Check validate parameter was passed correctly
            assert mock_dist.call_args[1]["_validate"] is True

            # Check the result
            assert result == mock_instance

    def test_to_position_distribution_no_validation(self):
        """Test conversion to PositionDistribution without validation."""
        with patch(
            "gridrival_ai.probabilities.converters.odds_converter.PositionDistribution"
        ) as mock_dist:
            self.converter.to_position_distribution(self.sample_odds, validate=False)
            # Check validate parameter was passed correctly
            assert mock_dist.call_args[1]["_validate"] is False

    def test_invalid_odds(self):
        """Test handling of invalid odds (≤ 1.0)."""
        invalid_odds = [0.5, 2.0, 3.0]  # First item is invalid

        with pytest.raises(ValueError):
            self.converter.convert(invalid_odds)

        with pytest.raises(ValueError):
            self.converter.convert_to_dict(invalid_odds)

        with pytest.raises(ValueError):
            self.converter.to_position_distribution(invalid_odds)
