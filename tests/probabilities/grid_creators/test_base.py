"""
Tests for the GridCreator base class.

This module contains tests for the GridCreator base class and its methods.
"""

from unittest import mock

import pytest

from gridrival_ai.probabilities.distributions import (
    PositionDistribution,
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.probabilities.grid_creators.base import GridCreator
from gridrival_ai.probabilities.normalizers.factory import get_grid_normalizer
from gridrival_ai.probabilities.odds_converters import BasicConverter
from gridrival_ai.probabilities.odds_structure import OddsStructure


class SimpleGridCreator(GridCreator):
    """Simple concrete implementation of GridCreator for testing."""

    def create_session_distribution(
        self, odds_input, session_type="race", **kwargs
    ) -> SessionDistribution:
        """Create a simple session distribution with 3 drivers."""
        self._ensure_odds_structure(odds_input)

        # Create simple position distributions for each driver
        driver_distributions = {}
        for driver_id in ["VER", "HAM", "NOR"]:
            # Create a simple position distribution
            if driver_id == "VER":
                pos_probs = {1: 0.6, 2: 0.3, 3: 0.1}
            elif driver_id == "HAM":
                pos_probs = {1: 0.3, 2: 0.6, 3: 0.1}
            else:  # NOR
                pos_probs = {1: 0.1, 2: 0.1, 3: 0.8}

            driver_distributions[driver_id] = PositionDistribution(pos_probs)

        return SessionDistribution(driver_distributions, session_type=session_type)

    def create_race_distribution(
        self, odds_input, include_qualifying=True, include_sprint=True, **kwargs
    ) -> RaceDistribution:
        """Create a simple race distribution."""
        # Create race session distribution
        race_session = self.create_session_distribution(
            odds_input, session_type="race", **kwargs
        )

        # Create qualifying and sprint distributions if requested
        qualifying_session = None
        sprint_session = None

        if include_qualifying:
            qualifying_session = self.create_session_distribution(
                odds_input, session_type="qualifying", **kwargs
            )

        if include_sprint:
            sprint_session = self.create_session_distribution(
                odds_input, session_type="sprint", **kwargs
            )

        return RaceDistribution(
            race=race_session, qualifying=qualifying_session, sprint=sprint_session
        )


class TestGridCreator:
    """Tests for the GridCreator base class."""

    @pytest.fixture
    def grid_creator(self):
        """Create a SimpleGridCreator instance for testing."""
        return SimpleGridCreator()

    @pytest.fixture
    def odds_structure(self):
        """Create a simple OddsStructure for testing."""
        return OddsStructure(
            {
                "race": {
                    1: {  # Position threshold for win odds
                        "VER": 1.8,
                        "HAM": 3.0,
                        "NOR": 5.0,
                    },
                    3: {  # Position threshold for top-3 odds
                        "VER": 1.2,
                        "HAM": 1.5,
                        "NOR": 2.0,
                    },
                },
                "qualifying": {
                    1: {  # Position threshold for win odds
                        "VER": 1.6,
                        "HAM": 2.5,
                        "NOR": 4.5,
                    }
                },
            }
        )

    @pytest.fixture
    def raw_odds_dict(self):
        """Create a simple raw odds dictionary for testing."""
        return {
            "race": {
                1: {  # Position threshold for win odds
                    "VER": 1.8,
                    "HAM": 3.0,
                    "NOR": 5.0,
                }
            }
        }

    def test_init(self, grid_creator):
        """Test initializing GridCreator."""
        assert isinstance(grid_creator.odds_converter, BasicConverter)
        assert isinstance(grid_creator.grid_normalizer, type(get_grid_normalizer()))

    def test_ensure_odds_structure_with_dict(self, grid_creator, raw_odds_dict):
        """Test _ensure_odds_structure with a dictionary."""
        result = grid_creator._ensure_odds_structure(raw_odds_dict)
        assert isinstance(result, OddsStructure)
        assert result.odds == raw_odds_dict

    def test_ensure_odds_structure_with_odds_structure(
        self, grid_creator, odds_structure
    ):
        """Test _ensure_odds_structure with an OddsStructure."""
        result = grid_creator._ensure_odds_structure(odds_structure)
        assert result is odds_structure

    def test_create_session_distribution(self, grid_creator, odds_structure):
        """Test create_session_distribution."""
        session = grid_creator.create_session_distribution(odds_structure)

        # Check session type
        assert session.session_type == "race"

        # Check driver IDs
        assert set(session.get_driver_ids()) == {"VER", "HAM", "NOR"}

        # Check VER probabilities
        ver_dist = session.get_driver_distribution("VER")
        assert ver_dist[1] == 0.6
        assert ver_dist[2] == 0.3
        assert ver_dist[3] == 0.1

    def test_create_race_distribution(self, grid_creator, odds_structure):
        """Test create_race_distribution."""
        race_dist = grid_creator.create_race_distribution(odds_structure)

        # Check race session
        race_session = race_dist.race
        assert race_session.session_type == "race"
        assert set(race_session.get_driver_ids()) == {"VER", "HAM", "NOR"}

        # Check qualifying session
        quali_session = race_dist.qualifying
        assert quali_session.session_type == "qualifying"
        assert set(quali_session.get_driver_ids()) == {"VER", "HAM", "NOR"}

        # Check sprint session
        sprint_session = race_dist.sprint
        assert sprint_session.session_type == "sprint"
        assert set(sprint_session.get_driver_ids()) == {"VER", "HAM", "NOR"}

    def test_create_race_distribution_no_quali_no_sprint(
        self, grid_creator, odds_structure
    ):
        """Test create_race_distribution with qualifying and sprint disabled."""
        # Create a real session for testing to avoid RaceDistribution.__post_init__ issues
        real_session = grid_creator.create_session_distribution(
            odds_structure, session_type="race"
        )

        with mock.patch.object(
            grid_creator, "create_session_distribution"
        ) as mock_create:
            # Set up mock to return a real session distribution
            mock_create.return_value = real_session

            # Call create_race_distribution with qualifying and sprint disabled
            result = grid_creator.create_race_distribution(
                odds_structure, include_qualifying=False, include_sprint=False
            )

            # Check create_session_distribution was called only once for race
            mock_create.assert_called_once_with(
                odds_structure,
                session_type="race",
            )

            # Check result is a RaceDistribution
            assert isinstance(result, RaceDistribution)

            # The race attribute should be the real session we provided
            assert result.race == real_session
