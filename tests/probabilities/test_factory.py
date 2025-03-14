"""Tests for probability distribution factory."""

import json
import os
import tempfile

import pytest

from gridrival_ai.probabilities.core import JointDistribution, PositionDistribution
from gridrival_ai.probabilities.factory import DistributionBuilder, DistributionFactory


class TestDistributionFactory:
    """Test suite for DistributionFactory."""

    def test_from_odds(self):
        """Test creating distribution from odds."""
        odds = [1.5, 3.0, 6.0]
        dist = DistributionFactory.from_odds(odds)

        # Should be a position distribution
        assert isinstance(dist, PositionDistribution)

        # Should have right probabilities
        total_prob = 1 / 1.5 + 1 / 3.0 + 1 / 6.0
        assert dist[1] == pytest.approx(1 / 1.5 / total_prob)
        assert dist[2] == pytest.approx(1 / 3.0 / total_prob)
        assert dist[3] == pytest.approx(1 / 6.0 / total_prob)

        # Should sum to 1.0
        assert sum(dist.position_probs.values()) == pytest.approx(1.0)

    def test_from_odds_with_method(self):
        """Test creating distribution from odds with different method."""
        odds = [1.5, 3.0, 6.0]
        dist = DistributionFactory.from_odds(odds, method="shin")

        # Should be a position distribution
        assert isinstance(dist, PositionDistribution)

        # Should have right order (lower odds = higher probability)
        assert dist[1] > dist[2] > dist[3]

        # Should sum to 1.0
        assert sum(dist.position_probs.values()) == pytest.approx(1.0)

    def test_from_odds_dict(self):
        """Test creating distributions from odds dictionary."""
        # Sample odds dictionary
        odds_dict = {"VER": 1.5, "HAM": 3.0, "NOR": 6.0}

        # Create distributions
        distributions = DistributionFactory.from_odds_dict(odds_dict)

        # Should have distributions for all drivers
        assert set(distributions.keys()) == {"VER", "HAM", "NOR"}

        # Should be position distributions
        assert isinstance(distributions["VER"], PositionDistribution)
        assert isinstance(distributions["HAM"], PositionDistribution)
        assert isinstance(distributions["NOR"], PositionDistribution)

        # Each distribution should be valid
        for dist in distributions.values():
            assert dist.is_valid

        # Lower odds should have higher win probability
        assert (
            distributions["VER"][1] > distributions["HAM"][1] > distributions["NOR"][1]
        )

        # Sum of P1 probabilities should be close to 1.0 for Harville method
        p1_sum = sum(dist[1] for dist in distributions.values())
        assert p1_sum == pytest.approx(1.0)

    def test_from_probabilities(self):
        """Test creating distribution from probabilities."""
        probs = {1: 0.6, 2: 0.4}
        dist = DistributionFactory.from_probabilities(probs)

        # Should have right values
        assert dist[1] == 0.6
        assert dist[2] == 0.4

    def test_from_probabilities_without_validation(self):
        """Test creating distribution without validation."""
        # Invalid probabilities (don't sum to 1.0)
        probs = {1: 0.6, 2: 0.2}

        # Should raise error with validation
        with pytest.raises(Exception):
            DistributionFactory.from_probabilities(probs)

        # Should not raise error without validation
        dist = DistributionFactory.from_probabilities(probs, validate=False)
        assert dist[1] == 0.6
        assert dist[2] == 0.2

    def test_from_probabilities_dict(self):
        """Test creating distributions from probabilities dictionary."""
        # Sample probabilities dictionary
        probs_dict = {
            "VER": {1: 0.6, 2: 0.3, 3: 0.1},
            "HAM": {1: 0.3, 2: 0.5, 3: 0.2},
            "NOR": {1: 0.1, 2: 0.2, 3: 0.7},
        }

        # Create distributions
        distributions = DistributionFactory.from_probabilities_dict(probs_dict)

        # Should have distributions for all drivers
        assert set(distributions.keys()) == {"VER", "HAM", "NOR"}

        # Should be position distributions
        assert isinstance(distributions["VER"], PositionDistribution)
        assert isinstance(distributions["HAM"], PositionDistribution)
        assert isinstance(distributions["NOR"], PositionDistribution)

        # Each distribution should be valid
        for dist in distributions.values():
            assert dist.is_valid

        # Should have the original probabilities
        assert distributions["VER"][1] == 0.6
        assert distributions["HAM"][2] == 0.5
        assert distributions["NOR"][3] == 0.7

        # Positions not specified should have zero probability
        assert distributions["VER"][4] == 0.0
        assert distributions["HAM"][4] == 0.0
        assert distributions["NOR"][4] == 0.0

    def test_from_json_position(self):
        """Test creating position distribution from JSON."""
        json_str = '{"type": "position", "probabilities": {"1": 0.6, "2": 0.4}}'
        dist = DistributionFactory.from_json(json_str)

        # Should be a position distribution
        assert isinstance(dist, PositionDistribution)

        # Should have right values
        assert dist[1] == 0.6
        assert dist[2] == 0.4

    def test_from_json_joint(self):
        """Test creating joint distribution from JSON."""
        json_str = (
            '{"type": "joint", '
            '"outcome1_name": "qualifying", "outcome2_name": "race", '
            '"probabilities": {"(1, 1)": 0.4, "(1, 2)": 0.2, "(2, 1)": 0.1, "(2, 2)": 0.3}}'
        )
        dist = DistributionFactory.from_json(json_str)

        # Should be a joint distribution
        assert isinstance(dist, JointDistribution)

        # Should have right outcome names
        assert dist.outcome1_name == "qualifying"
        assert dist.outcome2_name == "race"

        # Should have right values
        assert dist[(1, 1)] == 0.4
        assert dist[(1, 2)] == 0.2
        assert dist[(2, 1)] == 0.1
        assert dist[(2, 2)] == 0.3

    def test_from_json_invalid(self):
        """Test errors on invalid JSON."""
        # Invalid JSON syntax
        with pytest.raises(ValueError, match="Invalid JSON"):
            DistributionFactory.from_json("{invalid json")

        # Missing type field
        with pytest.raises(ValueError, match="Missing 'type' field"):
            DistributionFactory.from_json('{"probabilities": {"1": 0.6}}')

        # Unknown type
        with pytest.raises(ValueError, match="Unknown distribution type"):
            DistributionFactory.from_json(
                '{"type": "unknown", "probabilities": {"1": 0.6}}'
            )

        # Missing probabilities field
        with pytest.raises(ValueError, match="Missing 'probabilities' field"):
            DistributionFactory.from_json('{"type": "position"}')

    def test_from_file(self):
        """Test creating distribution from file."""
        # Create a temporary file with distribution data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"type": "position", "probabilities": {"1": 0.6, "2": 0.4}}, f)
            f.flush()
            file_path = f.name

        try:
            # Load from file
            dist = DistributionFactory.from_file(file_path)

            # Should be a position distribution
            assert isinstance(dist, PositionDistribution)

            # Should have right values
            assert dist[1] == 0.6
            assert dist[2] == 0.4
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_from_file_nonexistent(self):
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DistributionFactory.from_file("nonexistent_file.json")

    def test_from_dict(self):
        """Test creating distribution from dictionary."""
        # Position distribution
        data = {"type": "position", "probabilities": {1: 0.6, 2: 0.4}}
        dist = DistributionFactory.from_dict(data)

        assert isinstance(dist, PositionDistribution)
        assert dist[1] == 0.6
        assert dist[2] == 0.4

        # Joint distribution
        data = {
            "type": "joint",
            "outcome1_name": "qual",
            "outcome2_name": "race",
            "probabilities": {(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3},
        }
        dist = DistributionFactory.from_dict(data)

        assert isinstance(dist, JointDistribution)
        assert dist.outcome1_name == "qual"
        assert dist.outcome2_name == "race"
        assert dist[(1, 1)] == 0.4

    def test_from_dict_invalid(self):
        """Test errors on invalid dictionary."""
        # Missing type
        with pytest.raises(ValueError, match="Missing 'type' field"):
            DistributionFactory.from_dict({"probabilities": {1: 0.6}})

        # Unknown type
        with pytest.raises(ValueError, match="Unknown distribution type"):
            DistributionFactory.from_dict(
                {"type": "unknown", "probabilities": {1: 0.6}}
            )

        # Missing probabilities
        with pytest.raises(ValueError, match="Missing 'probabilities' field"):
            DistributionFactory.from_dict({"type": "position"})

    def test_grid_from_odds(self):
        """Test creating grid of distributions from odds."""
        odds = [1.5, 3.0]
        driver_ids = ["VER", "HAM"]
        dists = DistributionFactory.grid_from_odds(odds, driver_ids)

        # Should have distributions for both drivers
        assert set(dists.keys()) == set(driver_ids)

        # Should be position distributions
        assert isinstance(dists["VER"], PositionDistribution)
        assert isinstance(dists["HAM"], PositionDistribution)

        # Should have valid probabilities
        assert dists["VER"].is_valid
        assert dists["HAM"].is_valid

        # Higher odds driver should have lower P1 probability
        assert dists["VER"][1] > dists["HAM"][1]

    def test_joint_independent(self):
        """Test creating independent joint distribution."""
        dist1 = DistributionFactory.from_probabilities({1: 0.6, 2: 0.4})
        dist2 = DistributionFactory.from_probabilities({1: 0.3, 2: 0.7})

        joint = DistributionFactory.joint_independent(dist1, dist2)

        # Should have right probabilities
        assert joint[(1, 1)] == pytest.approx(0.6 * 0.3)
        assert joint[(1, 2)] == pytest.approx(0.6 * 0.7)
        assert joint[(2, 1)] == pytest.approx(0.4 * 0.3)
        assert joint[(2, 2)] == pytest.approx(0.4 * 0.7)

        # Should be independent
        assert joint.is_independent()

    def test_joint_constrained(self):
        """Test creating constrained joint distribution."""
        dist1 = DistributionFactory.from_probabilities({1: 0.6, 2: 0.4})
        dist2 = DistributionFactory.from_probabilities({1: 0.3, 2: 0.7})

        joint = DistributionFactory.joint_constrained(dist1, dist2)

        # Should have zero probability for same positions
        assert joint[(1, 1)] == 0.0
        assert joint[(2, 2)] == 0.0

        # Should be valid
        assert joint.is_valid

        # Should not be independent
        assert not joint.is_independent()

    def test_joint_conditional(self):
        """Test creating conditional joint distribution."""
        marginal = DistributionFactory.from_probabilities({1: 0.6, 2: 0.4})

        def get_conditional(x):
            if x == 1:
                return DistributionFactory.from_probabilities({1: 0.2, 2: 0.8})
            else:  # x == 2
                return DistributionFactory.from_probabilities({1: 0.7, 2: 0.3})

        joint = DistributionFactory.joint_conditional(marginal, get_conditional)

        # Should have right probabilities
        assert joint[(1, 1)] == pytest.approx(0.6 * 0.2)
        assert joint[(1, 2)] == pytest.approx(0.6 * 0.8)
        assert joint[(2, 1)] == pytest.approx(0.4 * 0.7)
        assert joint[(2, 2)] == pytest.approx(0.4 * 0.3)

        # Should not be independent (in general)
        assert not joint.is_independent()

    def test_builder(self):
        """Test getting a builder."""
        builder = DistributionFactory.builder()
        assert isinstance(builder, DistributionBuilder)


class TestDistributionBuilder:
    """Test suite for DistributionBuilder."""

    def test_basic_build(self):
        """Test building basic distribution."""
        dist = DistributionBuilder().from_odds([1.5, 3.0, 6.0]).build()

        assert isinstance(dist, PositionDistribution)
        assert dist.is_valid
        assert dist[1] > dist[2] > dist[3]

    def test_with_entity_and_context(self):
        """Test building with entity and context (metadata only)."""
        dist = (
            DistributionBuilder()
            .for_entity("VER")
            .in_context("qualifying")
            .from_odds([1.5, 3.0, 6.0])
            .build()
        )

        assert isinstance(dist, PositionDistribution)
        assert dist.is_valid
        # Entity and context are metadata only, not stored in distribution

    def test_from_probabilities(self):
        """Test building from probabilities."""
        dist = DistributionBuilder().from_probabilities({1: 0.6, 2: 0.4}).build()

        assert isinstance(dist, PositionDistribution)
        assert dist[1] == 0.6
        assert dist[2] == 0.4

    def test_from_distribution(self):
        """Test building from existing distribution."""
        existing = PositionDistribution({1: 0.6, 2: 0.4})
        dist = DistributionBuilder().from_distribution(existing).build()

        assert dist is existing  # Should be the same object

    def test_using_method(self):
        """Test using different odds conversion method."""
        dist1 = (
            DistributionBuilder()
            .from_odds([1.5, 3.0, 6.0])
            .using_method("basic")
            .build()
        )

        dist2 = (
            DistributionBuilder()
            .from_odds([1.5, 3.0, 6.0])
            .using_method("shin")
            .build()
        )

        # Different methods should give different results
        assert dist1[1] != dist2[1]

        # But both should be valid
        assert dist1.is_valid
        assert dist2.is_valid

    def test_with_smoothing(self):
        """Test applying smoothing."""
        # Create a distribution with a single peak
        dist = (
            DistributionBuilder()
            .from_probabilities({1: 1.0, 2: 0.0})
            .with_smoothing(0.2)
            .build()
        )

        # Should have smoothed out the peak
        assert dist[1] < 1.0
        assert dist[2] > 0.0

        # Should still be valid
        assert dist.is_valid

    def test_invalid_smoothing(self):
        """Test error on invalid smoothing parameter."""
        builder = DistributionBuilder().from_odds([1.5, 3.0, 6.0])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            builder.with_smoothing(1.5)

    def test_build_without_data(self):
        """Test error when building without data source."""
        builder = DistributionBuilder()

        with pytest.raises(ValueError, match="No data source provided"):
            builder.build()

    def test_chain_building(self):
        """Test chaining of build methods (later ones override earlier ones)."""
        # Start with probabilities
        builder = DistributionBuilder().from_probabilities({1: 0.6, 2: 0.4})

        # Then override with odds
        dist = builder.from_odds([1.5, 3.0, 6.0]).build()

        # Should use the odds
        assert dist[1] != 0.6
