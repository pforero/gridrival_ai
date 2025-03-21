"""
Tests for the OddsStructure module.

This module contains tests for the OddsStructure class, which
represents a validated odds structure for F1 races.
"""

import pytest

from gridrival_ai.probabilities.odds_structure import OddsStructure


class TestOddsStructure:
    """Tests for the OddsStructure class."""

    def test_initialization_minimal(self):
        """Test initialization with minimal valid input."""
        # Just race win odds
        odds = {"race": {1: {"VER": 2.5, "HAM": 4.0, "NOR": 5.0}}}

        odds_structure = OddsStructure(odds)
        assert odds_structure.drivers == {"VER", "HAM", "NOR"}
        assert set(odds_structure.sessions) == {"race", "qualifying", "sprint"}

        # Check auto-complete worked
        assert odds_structure.get_win_odds("qualifying") == {
            "VER": 2.5,
            "HAM": 4.0,
            "NOR": 5.0,
        }
        assert odds_structure.get_win_odds("sprint") == {
            "VER": 2.5,
            "HAM": 4.0,
            "NOR": 5.0,
        }

    def test_initialization_full(self):
        """Test initialization with complete input."""
        odds = {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0, "NOR": 5.0},
                3: {"VER": 1.2, "HAM": 1.5, "NOR": 2.0},
            },
            "qualifying": {1: {"VER": 2.2, "HAM": 3.5, "NOR": 4.5}},
            "sprint": {1: {"VER": 2.3, "HAM": 3.8, "NOR": 4.8}},
        }

        odds_structure = OddsStructure(odds)
        assert odds_structure.drivers == {"VER", "HAM", "NOR"}
        assert set(odds_structure.sessions) == {"race", "qualifying", "sprint"}

        # Check that each session has its own odds
        assert odds_structure.get_win_odds("race") == {
            "VER": 2.5,
            "HAM": 4.0,
            "NOR": 5.0,
        }
        assert odds_structure.get_win_odds("qualifying") == {
            "VER": 2.2,
            "HAM": 3.5,
            "NOR": 4.5,
        }
        assert odds_structure.get_win_odds("sprint") == {
            "VER": 2.3,
            "HAM": 3.8,
            "NOR": 4.8,
        }

    def test_initialization_without_auto_complete(self):
        """Test initialization without auto-completing sessions."""
        odds = {"race": {1: {"VER": 2.5, "HAM": 4.0, "NOR": 5.0}}}

        odds_structure = OddsStructure(odds, auto_complete=False)
        assert odds_structure.drivers == {"VER", "HAM", "NOR"}
        assert odds_structure.sessions == ["race"]

        # Other sessions should not be available
        with pytest.raises(ValueError):
            odds_structure.get_win_odds("qualifying")

    def test_invalid_session(self):
        """Test validation with invalid session."""
        odds = {
            "race": {1: {"VER": 2.5, "HAM": 4.0}},
            "invalid_session": {1: {"VER": 2.5, "HAM": 4.0}},
        }

        with pytest.raises(ValueError, match="Invalid session"):
            OddsStructure(odds)

    def test_invalid_position(self):
        """Test validation with invalid position threshold."""
        odds = {"race": {0: {"VER": 2.5, "HAM": 4.0}}}  # 0 is invalid, must be >= 1

        with pytest.raises(ValueError, match="Position thresholds must be positive"):
            OddsStructure(odds)

    def test_position_exceeds_driver_count(self):
        """Test validation with position threshold exceeding driver count."""
        odds = {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0},
                3: {"VER": 1.2, "HAM": 1.5},
                # 3 exceeds driver count of 2
            }
        }

        with pytest.raises(
            ValueError, match="Position threshold 3 exceeds driver count 2"
        ):
            OddsStructure(odds)

    def test_invalid_odds(self):
        """Test validation with invalid odds values."""
        odds = {"race": {1: {"VER": 2.5, "HAM": 0.8}}}  # 0.8 is invalid, must be > 1.0

        with pytest.raises(ValueError, match="Invalid odd 0.8"):
            OddsStructure(odds)

    def test_missing_race_win_odds(self):
        """Test validation with missing race win odds."""
        odds = {
            "race": {3: {"VER": 1.2, "HAM": 1.5, "NOR": 1.8}}
        }  # Missing position 1 (win odds)

        with pytest.raises(ValueError, match="Race odds must include win odds"):
            OddsStructure(odds)

    def test_get_session_odds(self):
        """Test get_session_odds method."""
        odds = {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0, "NOR": 7.0},
                3: {"VER": 1.2, "HAM": 1.5, "NOR": 2.5},
            }
        }

        odds_structure = OddsStructure(odds)
        race_odds = odds_structure.get_session_odds("race")

        assert race_odds == {
            1: {"VER": 2.5, "HAM": 4.0, "NOR": 7.0},
            3: {"VER": 1.2, "HAM": 1.5, "NOR": 2.5},
        }

        # Invalid session
        with pytest.raises(ValueError, match="Invalid session"):
            odds_structure.get_session_odds("invalid")

    def test_get_driver_odds(self):
        """Test get_driver_odds method."""
        odds = {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0, "NOR": 7.0},
                3: {"VER": 1.2, "HAM": 1.5, "NOR": 2.5},
            }
        }

        odds_structure = OddsStructure(odds)
        ver_odds = odds_structure.get_driver_odds("VER", "race")

        assert ver_odds == {1: 2.5, 3: 1.2}

        # Driver not in odds
        assert odds_structure.get_driver_odds("RUS", "race") == {}

    def test_get_position_odds(self):
        """Test get_position_odds method."""
        odds = {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0, "NOR": 7.0},
                3: {"VER": 1.2, "HAM": 1.5, "NOR": 2.5},
            }
        }

        odds_structure = OddsStructure(odds)
        win_odds = odds_structure.get_position_odds(1, "race")

        assert win_odds == {"VER": 2.5, "HAM": 4.0, "NOR": 7.0}

        # Position not in session
        with pytest.raises(ValueError, match="Position 5 not found"):
            odds_structure.get_position_odds(5, "race")

    def test_get_win_odds(self):
        """Test get_win_odds method."""
        odds = {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0, "NOR": 7.0},
                3: {"VER": 1.2, "HAM": 1.5, "NOR": 2.5},
            }
        }

        odds_structure = OddsStructure(odds)
        win_odds = odds_structure.get_win_odds("race")

        assert win_odds == {"VER": 2.5, "HAM": 4.0, "NOR": 7.0}

    def test_get_win_odds_list(self):
        """Test get_win_odds_list method."""
        odds = {"race": {1: {"VER": 2.5, "HAM": 4.0, "NOR": 5.0}}}

        odds_structure = OddsStructure(odds)
        odds_list, driver_ids = odds_structure.get_win_odds_list("race")

        # Check that the order is preserved
        assert len(odds_list) == 3
        assert len(driver_ids) == 3

        # Check that odds match drivers
        for i, driver in enumerate(driver_ids):
            assert odds_list[i] == odds["race"][1][driver]

    def test_get_thresholds(self):
        """Test get_thresholds method."""
        drivers = ["VER", "NOR", "HAM", "LEC", "ALO", "SAI", "RUS", "PIA", "ANT", "TSU"]
        odds = {
            "race": {
                3: {driver_id: 1 / (3 / 10) for driver_id in drivers},
                1: {driver_id: 1 / (1 / 10) for driver_id in drivers},
                6: {driver_id: 1 / (6 / 10) for driver_id in drivers},
            }
        }

        odds_structure = OddsStructure(odds)
        thresholds = odds_structure.get_thresholds("race")

        assert thresholds == [1, 3, 6]  # Sorted

    def test_from_win_odds(self):
        """Test from_win_odds class method."""
        win_odds = {"VER": 2.5, "HAM": 4.0, "NOR": 5.0}

        odds_structure = OddsStructure.from_win_odds(win_odds)

        assert odds_structure.drivers == {"VER", "HAM", "NOR"}
        assert odds_structure.get_win_odds("race") == win_odds

        # Custom session
        quali_structure = OddsStructure.from_win_odds(win_odds, "qualifying")
        assert quali_structure.get_win_odds("qualifying") == win_odds

        # Invalid session
        with pytest.raises(ValueError, match="Invalid session"):
            OddsStructure.from_win_odds(win_odds, "invalid")

    def test_from_win_odds_list(self):
        """Test from_win_odds_list class method."""
        odds_list = [2.5, 4.0, 5.0]
        driver_ids = ["VER", "HAM", "NOR"]

        odds_structure = OddsStructure.from_win_odds_list(odds_list, driver_ids)

        assert odds_structure.drivers == {"VER", "HAM", "NOR"}
        assert odds_structure.get_win_odds("race") == {
            "VER": 2.5,
            "HAM": 4.0,
            "NOR": 5.0,
        }

        # Mismatched lengths
        with pytest.raises(ValueError, match="Length of odds_list"):
            OddsStructure.from_win_odds_list(odds_list, driver_ids[:-1])

    def test_consistency_with_different_drivers(self):
        """Test handling of odds with different drivers across positions."""
        odds = {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0, "NOR": 5.0},
                3: {"VER": 1.2, "HAM": 1.5},  # Missing NOR
            }
        }

        odds_structure = OddsStructure(odds)

        # Should include all drivers
        assert odds_structure.drivers == {"VER", "HAM", "NOR"}

        # Get driver odds should handle missing drivers
        assert odds_structure.get_driver_odds("NOR", "race") == {1: 5.0}

    def test_empty_initialization(self):
        """Test initialization with empty dictionary."""
        # Should fail validation
        with pytest.raises(ValueError, match="Race odds must include win odds"):
            OddsStructure({})

    def test_non_dict_initialization(self):
        """Test initialization with non-dictionary."""
        with pytest.raises(TypeError, match="Odds must be a dictionary"):
            OddsStructure([])
