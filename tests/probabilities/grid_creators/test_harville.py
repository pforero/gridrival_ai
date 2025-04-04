"""
Tests for the Harville grid creator.

This module contains tests for the HarvilleGridCreator class, which implements
the Harville method for creating position probability distributions.
"""

import numpy as np
import pytest

from gridrival_ai.probabilities.distributions import (
    PositionDistribution,
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.probabilities.grid_creators.harville import HarvilleGridCreator
from gridrival_ai.probabilities.odds_structure import OddsStructure


class TestHarvilleGridCreator:
    """Tests for the HarvilleGridCreator class."""

    @pytest.fixture
    def sample_odds(self):
        """Sample odds dictionary for testing."""
        return {"race": {1: {"VER": 2.0, "HAM": 4.0, "NOR": 5.0}}}

    @pytest.fixture
    def sample_odds_with_quali(self):
        """Sample odds dictionary with qualifying data for testing."""
        return {
            "race": {1: {"VER": 2.0, "HAM": 4.0, "NOR": 5.0}},
            "qualifying": {1: {"VER": 1.8, "HAM": 3.5, "NOR": 4.0}},
        }

    @pytest.fixture
    def sample_odds_with_multiple_markets(self):
        """Sample odds dictionary with multiple markets (win and top 3)."""
        return {
            "race": {
                1: {"VER": 2.0, "HAM": 4.0, "NOR": 5.0},  # Win odds
                3: {"VER": 1.1, "HAM": 1.5, "NOR": 1.8},  # Top 3 odds
            }
        }

    @pytest.fixture
    def sample_odds_structure(self, sample_odds):
        """Sample OddsStructure for testing."""
        return OddsStructure(sample_odds)

    @pytest.fixture
    def sample_odds_with_quali_structure(self, sample_odds_with_quali):
        """Sample OddsStructure with qualifying for testing."""
        return OddsStructure(sample_odds_with_quali)

    @pytest.fixture
    def sample_odds_with_multiple_markets_structure(
        self, sample_odds_with_multiple_markets
    ):
        """Sample OddsStructure with multiple markets for testing."""
        return OddsStructure(sample_odds_with_multiple_markets)

    @pytest.fixture
    def creator(self):
        """Return a HarvilleGridCreator."""
        return HarvilleGridCreator()

    def test_create_session_distribution(self, creator, sample_odds):
        """Test create_session_distribution method with standard inputs."""
        session_dist = creator.create_session_distribution(sample_odds)

        # Check the result is a SessionDistribution with correct session type
        assert isinstance(session_dist, SessionDistribution)
        assert session_dist.session_type == "race"

        # Check we have distributions for all drivers
        driver_distributions = session_dist.driver_distributions
        assert set(driver_distributions.keys()) == {"VER", "HAM", "NOR"}

        # Check each driver distribution is valid and normalized
        for driver, dist in driver_distributions.items():
            assert isinstance(dist, PositionDistribution)
            assert dist.is_valid
            assert np.isclose(sum(dist.position_probs.values()), 1.0)
            assert len(dist.position_probs) == 3  # 3 drivers = 3 positions

        # Check grid constraints - each position sums to 1.0 across drivers
        for pos in range(1, 4):
            pos_sum = sum(dist.get(pos) for dist in driver_distributions.values())
            assert np.isclose(pos_sum, 1.0)

        # Check relative probabilities match the relative betting odds
        # Lower betting odds = higher probability
        ver_p1 = driver_distributions["VER"].get(1)
        ham_p1 = driver_distributions["HAM"].get(1)
        nor_p1 = driver_distributions["NOR"].get(1)

        assert ver_p1 > ham_p1 > nor_p1, "P1 probabilities should match relative odds"

        # Test with larger grid
        # Create odds for 10 drivers with descending strength
        driver_ids = [f"D{i}" for i in range(1, 11)]
        win_odds = {driver_id: 1.1 + i for i, driver_id in enumerate(driver_ids)}
        large_odds = {"race": {1: win_odds}}

        large_session = creator.create_session_distribution(large_odds)
        large_distributions = large_session.driver_distributions

        # Check grid constraints for large grid
        for pos in range(1, 11):
            pos_sum = sum(dist.get(pos) for dist in large_distributions.values())
            assert np.isclose(pos_sum, 1.0)

    def test_create_race_distribution(self, creator, sample_odds):
        """Test create_race_distribution method with race session only."""
        race_dist = creator.create_race_distribution(sample_odds)

        # Check we have a RaceDistribution with all required sessions
        assert isinstance(race_dist, RaceDistribution)
        assert isinstance(race_dist.race, SessionDistribution)
        assert isinstance(race_dist.qualifying, SessionDistribution)
        assert isinstance(race_dist.sprint, SessionDistribution)

        # Check session types
        assert race_dist.race.session_type == "race"
        assert race_dist.qualifying.session_type == "qualifying"
        assert race_dist.sprint.session_type == "sprint"

        # Since only race odds provided, qualifying and sprint should match race
        race_ver_p1 = race_dist.race.get_driver_distribution("VER").get(1)
        quali_ver_p1 = race_dist.qualifying.get_driver_distribution("VER").get(1)
        sprint_ver_p1 = race_dist.sprint.get_driver_distribution("VER").get(1)

        assert np.isclose(race_ver_p1, quali_ver_p1)
        assert np.isclose(race_ver_p1, sprint_ver_p1)

    def test_create_race_distribution_with_quali(
        self, creator, sample_odds_with_quali_structure
    ):
        """Test create_race_distribution with race and qualifying odds."""
        race_dist = creator.create_race_distribution(sample_odds_with_quali_structure)

        # Get probabilities for VER in P1 for each session
        race_ver_p1 = race_dist.race.get_driver_distribution("VER").get(1)
        quali_ver_p1 = race_dist.qualifying.get_driver_distribution("VER").get(1)
        sprint_ver_p1 = race_dist.sprint.get_driver_distribution("VER").get(1)

        # Verify that probabilities are valid
        assert 0 < race_ver_p1 < 1, "Race probability should be between 0 and 1"
        assert 0 < quali_ver_p1 < 1, "Qualifying probability should be between 0 and 1"
        assert 0 < sprint_ver_p1 < 1, "Sprint probability should be between 0 and 1"

        # Sprint should match race (derived from race since not in odds)
        assert np.isclose(race_ver_p1, sprint_ver_p1)

    def test_only_win_odds_used(
        self, creator, sample_odds, sample_odds_with_multiple_markets_structure
    ):
        """Only win odds (market 1) are used even when multiple markets are provided."""
        # Create distribution using only win odds
        win_only_dist = creator.create_session_distribution(sample_odds)

        # Create distribution using multiple markets (win and top 3)
        multiple_markets_dist = creator.create_session_distribution(
            sample_odds_with_multiple_markets_structure
        )

        # The results should be identical since Harville only uses win odds
        win_only_drivers = win_only_dist.driver_distributions
        multiple_markets_drivers = multiple_markets_dist.driver_distributions

        # Check that both distributions have the same drivers
        assert set(win_only_drivers.keys()) == set(multiple_markets_drivers.keys())

        # Check that each driver has identical position probabilities
        for driver in win_only_drivers:
            for pos in range(1, 4):
                win_only_prob = win_only_drivers[driver].get(pos)
                multiple_markets_prob = multiple_markets_drivers[driver].get(pos)
                assert np.isclose(
                    win_only_prob, multiple_markets_prob
                ), f"Probabilities differ for driver {driver} at position {pos}"

    def test_harville_properties(self, creator):
        """Test the mathematical properties of the Harville method."""
        # Simple case with two drivers and 2:1 strength ratio
        odds = {"race": {1: {"VER": 2.0, "HAM": 4.0}}}  # 2:1 ratio of strengths
        session_dist = creator.create_session_distribution(odds)
        distributions = session_dist.driver_distributions

        # With Harville method and these odds:
        # VER strength = 1/2.0 = 0.5, HAM strength = 1/4.0 = 0.25
        # Total strength = 0.75
        # VER P1 = 0.5/0.75 = 2/3, HAM P1 = 0.25/0.75 = 1/3
        assert np.isclose(distributions["VER"].get(1), 2 / 3)
        assert np.isclose(distributions["HAM"].get(1), 1 / 3)

        # For P2, only the remaining driver is available:
        # VER P2 = 0, HAM P2 = 1 if VER is P1 (with probability 2/3)
        # VER P2 = 1, HAM P2 = 0 if HAM is P1 (with probability 1/3)
        # So VER P2 = 1/3, HAM P2 = 2/3
        assert np.isclose(distributions["VER"].get(2), 1 / 3)
        assert np.isclose(distributions["HAM"].get(2), 2 / 3)

    def test_empty_input(self, creator):
        """Test error handling with empty input."""
        with pytest.raises(ValueError):
            creator.create_session_distribution({})

    def test_single_driver(self, creator):
        """Test behavior with a single driver."""
        odds = {"race": {1: {"VER": 1.5}}}
        session_dist = creator.create_session_distribution(odds)

        # With one driver, they should get P1 with 100% probability
        distributions = session_dist.driver_distributions
        assert len(distributions) == 1
        assert "VER" in distributions
        assert distributions["VER"].get(1) == 1.0
