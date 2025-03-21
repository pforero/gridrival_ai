import math

import pytest

from gridrival_ai.probabilities.distributions import (
    PositionDistribution,
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.probabilities.grid_creators import (
    CumulativeGridCreator,
    get_grid_creator,
)
from gridrival_ai.probabilities.odds_structure import OddsStructure


class TestCumulativeGridCreator:
    """Test suite for CumulativeGridCreator."""

    @pytest.fixture
    def sample_odds(self):
        """Fixture with sample odds structure for testing."""
        return {
            "race": {
                1: {"VER": 2.2, "HAM": 4.0, "NOR": 7.0, "PIA": 13.0},  # Win odds
                3: {"VER": 1.2, "HAM": 1.5, "NOR": 1.8, "PIA": 2.5},  # Top-3 odds
            }
        }

    @pytest.fixture
    def win_only_odds(self):
        """Fixture with win-only odds."""
        return {
            "race": {
                1: {"VER": 2.2, "HAM": 4.0, "NOR": 7.0, "PIA": 13.0},  # Win odds only
            }
        }

    @pytest.fixture
    def creator(self):
        """Fixture providing a standard grid creator instance."""
        return CumulativeGridCreator(
            max_position=20,
            baseline_method="exponential",
            baseline_params={"decay": 0.5},
        )

    def test_create_session_distribution(self, creator, sample_odds):
        """Test creating session distribution with valid inputs."""
        # Create session distribution
        session_dist = creator.create_session_distribution(sample_odds)

        # Check that it's the right type
        assert isinstance(session_dist, SessionDistribution)
        assert session_dist.session_type == "race"

        # Check that we have distributions for all drivers
        driver_ids = ["VER", "HAM", "NOR", "PIA"]
        for driver_id in driver_ids:
            assert driver_id in session_dist.get_driver_ids()

            # Get driver distribution
            driver_dist = session_dist.get_driver_distribution(driver_id)
            assert isinstance(driver_dist, PositionDistribution)
            assert driver_dist.is_valid

        # Check that win probabilities are preserved
        # These should approximate the implied win probabilities
        assert 0.4 < session_dist.get_driver_distribution("VER").get(1) < 0.5  # ~0.45
        assert 0.2 < session_dist.get_driver_distribution("HAM").get(1) < 0.3  # ~0.25
        assert 0.1 < session_dist.get_driver_distribution("NOR").get(1) < 0.2  # ~0.14
        assert 0.05 < session_dist.get_driver_distribution("PIA").get(1) < 0.1  # ~0.077

    def test_win_only_fallback(self, creator, win_only_odds):
        """Test fallback to Harville method with win-only odds."""
        # Create session distribution
        session_dist = creator.create_session_distribution(win_only_odds)

        # Check that it's the right type
        assert isinstance(session_dist, SessionDistribution)

        # Check that all distributions are valid
        for driver_id in session_dist.get_driver_ids():
            driver_dist = session_dist.get_driver_distribution(driver_id)
            assert driver_dist.is_valid
            assert math.isclose(
                sum(driver_dist.get(p) for p in range(1, 21)), 1.0, abs_tol=1e-6
            )

        # Check that win probabilities match the implied probabilities
        ver_win_prob = session_dist.get_driver_distribution("VER").get(1)
        ham_win_prob = session_dist.get_driver_distribution("HAM").get(1)
        nor_win_prob = session_dist.get_driver_distribution("NOR").get(1)
        pia_win_prob = session_dist.get_driver_distribution("PIA").get(1)

        # Win probabilities should sum to approximately 1.0
        assert math.isclose(
            ver_win_prob + ham_win_prob + nor_win_prob + pia_win_prob, 1.0, abs_tol=1e-6
        )

    def test_create_race_distribution(self, creator, sample_odds):
        """Test creating race distribution with multiple sessions."""
        # Add qualifying odds to the sample odds
        odds_with_quali = sample_odds.copy()
        odds_with_quali["qualifying"] = {
            1: {"VER": 1.8, "HAM": 3.5, "NOR": 6.0, "PIA": 15.0},  # Qualifying win odds
        }

        # Create race distribution
        race_dist = creator.create_race_distribution(odds_with_quali)

        # Check that it's the right type
        assert isinstance(race_dist, RaceDistribution)

        # Check that race and quali sessions have valid distributions
        assert race_dist.race is not None
        assert race_dist.qualifying is not None

        # Check that distributions are valid
        for session in [race_dist.race, race_dist.qualifying]:
            for driver_id in session.get_driver_ids():
                driver_dist = session.get_driver_distribution(driver_id)
                assert driver_dist.is_valid

    def test_different_baseline_methods(self, sample_odds):
        """Test different baseline weighting methods."""
        # Create grid creators with different baseline methods and parameters
        # Use significantly different parameters to ensure distinct distributions
        exponential_creator = CumulativeGridCreator(
            baseline_method="exponential",
            baseline_params={"decay": 0.1},  # Low decay
        )
        linear_creator = CumulativeGridCreator(
            baseline_method="linear",
        )
        uniform_creator = CumulativeGridCreator(
            baseline_method="uniform",
        )

        # Use single session type to avoid ambiguity
        single_session_odds = {"race": sample_odds["race"]}

        # Create session distributions with each method
        exp_dist = exponential_creator.create_session_distribution(single_session_odds)
        lin_dist = linear_creator.create_session_distribution(single_session_odds)
        uni_dist = uniform_creator.create_session_distribution(single_session_odds)

        # Check that all produce valid distributions
        for dist in [exp_dist, lin_dist, uni_dist]:
            for driver_id in dist.get_driver_ids():
                assert dist.get_driver_distribution(driver_id).is_valid

        # Get distributions for a specific driver for comparison
        ver_exp = exp_dist.get_driver_distribution("VER")
        ver_lin = lin_dist.get_driver_distribution("VER")
        ver_uni = uni_dist.get_driver_distribution("VER")

        # Compare probabilities across multiple positions to ensure they differ
        for pos in range(
            2, 10
        ):  # Skip position 1 which is constrained by the win probability
            probs = [ver_exp.get(pos), ver_lin.get(pos), ver_uni.get(pos)]

            # Only assert if probabilities are significant enough to compare
            if all(p > 0.01 for p in probs):
                # Check if there's variation in the probabilities
                # Using a very small epsilon to detect any difference
                max_diff = max(probs) - min(probs)
                assert (
                    max_diff > 0.001
                ), f"Probabilities at position {pos} should differ: {probs}"

    def test_column_constraints(self, creator, sample_odds):
        """Test that column constraints are enforced."""
        # Create session distribution
        session_dist = creator.create_session_distribution(sample_odds)

        # For each position, sum across all drivers should be close to 1.0
        for pos in range(1, 5):  # Check first few positions
            pos_sum = sum(
                session_dist.get_driver_distribution(driver_id).get(pos)
                for driver_id in session_dist.get_driver_ids()
            )
            assert 0.99 <= pos_sum <= 1.01, f"Position {pos} sum is {pos_sum}"

    def test_integration_with_factory(self, sample_odds):
        """Test integration with factory."""
        # Get creator via factory
        creator = get_grid_creator("cumulative", max_position=10)

        # Test that it works
        session_dist = creator.create_session_distribution(sample_odds)

        # Check basic properties
        assert isinstance(session_dist, SessionDistribution)
        assert len(session_dist.get_driver_ids()) == 4

        # Check that max_position parameter worked
        for driver_id in session_dist.get_driver_ids():
            dist = session_dist.get_driver_distribution(driver_id)
            # No positions beyond 10 should have probability
            for pos in range(11, 21):
                assert dist.get(pos) == 0

    def test_use_odds_structure(self, creator):
        """Test using OddsStructure as input."""
        # Create odds structure
        odds_dict = {
            "race": {
                1: {"VER": 2.2, "HAM": 4.0},
            }
        }
        odds_structure = OddsStructure(odds_dict)

        # Use directly with create_session_distribution
        session_dist = creator.create_session_distribution(odds_structure)

        # Check that it works
        assert isinstance(session_dist, SessionDistribution)
        assert "VER" in session_dist.get_driver_ids()
        assert "HAM" in session_dist.get_driver_ids()


if __name__ == "__main__":
    # Run tests manually
    pytest.main(["-xvs", __file__])
