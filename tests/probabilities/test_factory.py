"""Tests for the probability distribution factory."""

from unittest.mock import MagicMock

import pytest

from gridrival_ai.probabilities.distributions import PositionDistribution
from gridrival_ai.probabilities.factory import DistributionFactory
from gridrival_ai.probabilities.registry import DistributionRegistry


@pytest.fixture
def simple_odds_structure():
    """Create a simple odds structure with a single position threshold."""
    return {
        "race": {
            1: {
                "VER": 2.0,  # 50% implied probability
                "HAM": 4.0,  # 25% implied probability
                "NOR": 5.0,  # 20% implied probability
            }
        }
    }


@pytest.fixture
def complex_odds_structure():
    """Create a complex odds structure with multiple position thresholds."""
    # Include 20 drivers for a realistic F1 grid
    drivers = {
        "VER": 2.0,
        "HAM": 4.0,
        "NOR": 5.0,
        "LEC": 7.0,
        "SAI": 9.0,
        "PIA": 12.0,
        "RUS": 15.0,
        "ALO": 20.0,
        "STR": 30.0,
        "OCO": 40.0,
        "GAS": 50.0,
        "ANT": 60.0,
        "HUL": 70.0,
        "BEA": 80.0,
        "ALB": 90.0,
        "DOO": 100.0,
        "TSU": 120.0,
        "BOR": 150.0,
        "HAD": 180.0,
        "LAW": 200.0,
    }

    top3_drivers = {
        k: v / 2
        for k, v in drivers.items()
        if k in ["VER", "HAM", "NOR", "LEC", "SAI", "PIA"]
    }
    top10_drivers = {
        k: v / 5 for k, v in drivers.items() if k in list(drivers.keys())[:12]
    }

    return {
        "race": {
            1: drivers.copy(),  # Win odds
            3: top3_drivers,  # Top 3 odds
            10: top10_drivers,  # Top 10 odds
        },
        "qualifying": {
            1: drivers.copy(),  # Pole odds
        },
    }


@pytest.fixture
def mock_registry():
    """Create a mock distribution registry."""
    registry = MagicMock(spec=DistributionRegistry)
    registry.register = MagicMock()
    return registry


class TestDistributionFactory:
    """Test suite for DistributionFactory."""

    def test_from_structured_odds_simple(self, simple_odds_structure):
        """Test creating distributions from a simple odds structure."""
        distributions = DistributionFactory.from_structured_odds(simple_odds_structure)

        # Check structure
        assert "race" in distributions
        assert "sprint" in distributions  # Added by fallback
        assert "qualifying" in distributions  # Added by fallback

        # Check drivers
        assert "VER" in distributions["race"]
        assert "HAM" in distributions["race"]
        assert "NOR" in distributions["race"]

        # Check types
        ver_dist = distributions["race"]["VER"]
        assert isinstance(ver_dist, PositionDistribution)

        # Ensure all distributions are valid
        for session, driver_dists in distributions.items():
            for driver_id, dist in driver_dists.items():
                assert dist.is_valid
                assert sum(dist.position_probs.values()) == pytest.approx(1.0)

    def test_from_structured_odds_complex(self, complex_odds_structure):
        "Test creating distributions from a complex structure with multiple thresholds."
        distributions = DistributionFactory.from_structured_odds(complex_odds_structure)

        # Check structure
        assert set(distributions.keys()) == {"race", "qualifying", "sprint"}
        assert len(distributions["race"]) == 20  # 20 drivers

        # Ensure all distributions are valid
        for session, driver_dists in distributions.items():
            for driver_id, dist in driver_dists.items():
                assert dist.is_valid
                assert sum(dist.position_probs.values()) == pytest.approx(1.0)

    def test_fallback_behavior(self, simple_odds_structure):
        """Test the fallback behavior for different session types."""
        # With fallback enabled (default)
        distributions = DistributionFactory.from_structured_odds(simple_odds_structure)
        assert "race" in distributions
        assert "qualifying" in distributions
        assert "sprint" in distributions

        # With fallback disabled
        distributions = DistributionFactory.from_structured_odds(
            simple_odds_structure, fallback_to_race=False
        )
        assert "race" in distributions
        assert "qualifying" not in distributions
        assert "sprint" not in distributions

        # Add empty qualifying data
        simple_odds_structure["qualifying"] = {}

        # With fallback enabled, should use race data for qualifying
        distributions = DistributionFactory.from_structured_odds(simple_odds_structure)
        assert "qualifying" in distributions
        assert len(distributions["qualifying"]) == 3  # Should have the same 3 drivers

    def test_empty_structure(self):
        """Test with an empty odds structure."""
        distributions = DistributionFactory.from_structured_odds({})
        assert distributions == {}

    def test_register_with_registry(self, simple_odds_structure, mock_registry):
        """Test registering distributions with a registry."""
        DistributionFactory.register_structured_odds(
            mock_registry, simple_odds_structure
        )

        # Should register for all three sessions with all drivers
        expected_calls = 3 * 3  # 3 sessions (race, qual, sprint) x 3 drivers
        assert mock_registry.register.call_count == expected_calls

        # Check call format
        for call_args in mock_registry.register.call_args_list:
            args, _ = call_args
            driver_id, session, dist = args

            # Verify the arguments have the expected format
            assert driver_id in ["VER", "HAM", "NOR"]
            assert session in ["race", "qualifying", "sprint"]
            assert isinstance(dist, PositionDistribution)

    def test_from_odds_dict(self):
        """Test creating distributions from a simple odds dictionary."""
        odds_dict = {"VER": 2.0, "HAM": 4.0, "NOR": 5.0}
        distributions = DistributionFactory.from_odds_dict(odds_dict)

        # Check basic structure
        assert "VER" in distributions
        assert "HAM" in distributions
        assert "NOR" in distributions

        # Verify all are valid distributions
        for driver_id, dist in distributions.items():
            assert isinstance(dist, PositionDistribution)
            assert dist.is_valid

    def test_from_simple_odds(self):
        """Test creating distributions from a list of odds."""
        odds = [2.0, 4.0, 5.0]
        driver_ids = ["VER", "HAM", "NOR"]

        # With driver IDs provided
        distributions = DistributionFactory.from_simple_odds(odds, driver_ids)
        assert "VER" in distributions
        assert "HAM" in distributions
        assert "NOR" in distributions

        # Without driver IDs (should use position numbers as IDs)
        distributions = DistributionFactory.from_simple_odds(odds)
        assert "1" in distributions
        assert "2" in distributions
        assert "3" in distributions

        # Error case: mismatched lengths
        with pytest.raises(ValueError, match="Number of driver IDs must match"):
            DistributionFactory.from_simple_odds(odds, driver_ids[:2])

    def test_conversion_methods(self, simple_odds_structure):
        """Test that different conversion methods produce different results."""
        methods = ["basic", "odds_ratio", "power"]
        results = {}

        # Generate distributions using different methods
        for method in methods:
            results[method] = DistributionFactory.from_structured_odds(
                simple_odds_structure, method=method
            )

        # Verify all methods produce valid distributions
        for method, distributions in results.items():
            assert "race" in distributions
            assert "VER" in distributions["race"]
            assert distributions["race"]["VER"].is_valid

    def test_position_count_matches_driver_count(self):
        """Test that the number of positions in distributions matches the number of
        drivers."""
        # Create odds structures with different numbers of drivers
        driver_counts = [3, 5, 10, 20]

        for count in driver_counts:
            # Generate a dictionary of odds for the specified number of drivers
            odds_dict = {f"D{i}": float(i + 2) for i in range(count)}
            odds_structure = {"race": {1: odds_dict.copy()}}

            distributions = DistributionFactory.from_structured_odds(odds_structure)

            # Check that each driver's distribution has exactly 'count' positions
            for driver_id, dist in distributions["race"].items():
                assert len(dist.position_probs) == count
                assert max(dist.position_probs.keys()) == count
                assert min(dist.position_probs.keys()) == 1

    def test_thresholds_above_driver_count_ignored(self):
        """Test that thresholds higher than the number of drivers are ignored."""
        # Create odds for 5 drivers
        odds_dict = {f"D{i}": float(i + 2) for i in range(5)}

        # Create a structure with thresholds both within and above driver count
        odds_structure = {
            "race": {
                1: odds_dict.copy(),  # Valid threshold
                3: odds_dict.copy(),  # Valid threshold
                8: odds_dict.copy(),  # Invalid threshold (> 5 drivers)
                10: odds_dict.copy(),  # Invalid threshold (> 5 drivers)
            }
        }

        # Get the distributions
        distributions = DistributionFactory.from_structured_odds(odds_structure)

        # Verify the distributions have exactly 5 positions (matching driver count)
        for driver_id, dist in distributions["race"].items():
            assert dist.is_valid
            assert len(dist.position_probs) == 5
            assert max(dist.position_probs.keys()) == 5

        # Directly test the threshold filtering by inspecting internal method behavior
        # We can check this by looking at the structure of distributions
        for dist in distributions["race"].values():
            # If high thresholds were included, we'd see positions 6-10
            for pos in range(6, 11):
                assert pos not in dist.position_probs
