"""Tests for probability distribution registry."""

import pytest

from gridrival_ai.probabilities.core import JointDistribution, PositionDistribution
from gridrival_ai.probabilities.registry import DistributionRegistry


@pytest.fixture
def registry():
    """Create a distribution registry for testing."""
    return DistributionRegistry()


@pytest.fixture
def sample_dist():
    """Create a sample position distribution."""
    return PositionDistribution({1: 0.6, 2: 0.4})


@pytest.fixture
def populated_registry(registry, sample_dist):
    """Create a registry with sample distributions."""
    registry.register("VER", "qualifying", sample_dist)
    registry.register("VER", "race", sample_dist)
    registry.register("HAM", "race", sample_dist)
    return registry


class TestDistributionRegistry:
    """Test suite for DistributionRegistry."""

    def test_register_and_get(self, registry, sample_dist):
        """Test registering and retrieving a distribution."""
        registry.register("VER", "qualifying", sample_dist)
        retrieved = registry.get("VER", "qualifying")
        assert retrieved is sample_dist
        assert retrieved[1] == 0.6

    def test_has(self, registry, sample_dist):
        """Test checking if a distribution exists."""
        assert not registry.has("VER", "qualifying")
        registry.register("VER", "qualifying", sample_dist)
        assert registry.has("VER", "qualifying")

    def test_get_missing(self, registry):
        """Test error when getting a missing distribution."""
        with pytest.raises(KeyError, match="No distribution found"):
            registry.get("VER", "qualifying")

    def test_get_or_default(self, registry, sample_dist):
        """Test get_or_default with existing and missing distributions."""
        # Missing distribution should return default
        default = PositionDistribution({1: 0.5, 2: 0.5})
        assert registry.get_or_default("VER", "qualifying", default) is default

        # Registered distribution should be returned
        registry.register("VER", "qualifying", sample_dist)
        assert registry.get_or_default("VER", "qualifying", default) is sample_dist

    def test_get_entities(self, populated_registry):
        """Test getting all entities."""
        entities = populated_registry.get_entities()
        assert set(entities) == {"VER", "HAM"}

    def test_get_entities_with_context(self, populated_registry):
        """Test getting entities with a specific context."""
        entities = populated_registry.get_entities("qualifying")
        assert entities == ["VER"]

        entities = populated_registry.get_entities("race")
        assert set(entities) == {"VER", "HAM"}

    def test_get_contexts(self, populated_registry):
        """Test getting all contexts."""
        contexts = populated_registry.get_contexts()
        assert set(contexts) == {"qualifying", "race"}

    def test_get_contexts_with_entity(self, populated_registry):
        """Test getting contexts for a specific entity."""
        contexts = populated_registry.get_contexts("VER")
        assert set(contexts) == {"qualifying", "race"}

        contexts = populated_registry.get_contexts("HAM")
        assert contexts == ["race"]

    def test_get_contexts_missing_entity(self, populated_registry):
        """Test getting contexts for a missing entity."""
        contexts = populated_registry.get_contexts("LEC")
        assert contexts == []

    def test_get_joint(self, populated_registry):
        """Test getting a joint distribution."""
        joint = populated_registry.get_joint("VER", "HAM", "race")

        # Should be an actual joint distribution
        assert isinstance(joint, JointDistribution)

        # Should have outcome names set correctly
        assert joint.outcome1_name == "HAM"
        assert joint.outcome2_name == "VER"

        # Should be constrained (no positions can be the same)
        assert joint[(1, 1)] == 0.0

        # Sum of probabilities should be 1.0
        total = sum(joint.joint_probs.values())
        assert total == pytest.approx(1.0)

    def test_get_joint_independent(self, populated_registry):
        """Test getting a joint distribution with independence."""
        joint = populated_registry.get_joint("VER", "HAM", "race", constrained=False)

        # Should have (1,1) position possible with independent joint
        assert joint[(1, 1)] > 0.0

        # Should have right probability calculated
        assert joint[(1, 1)] == pytest.approx(0.6 * 0.6)

    def test_get_joint_nonexistent(self, registry, sample_dist):
        """Test error when getting joint with missing distributions."""
        registry.register("VER", "race", sample_dist)

        with pytest.raises(KeyError, match="No distribution found"):
            registry.get_joint("VER", "HAM", "race")

    def test_get_joint_caching(self, populated_registry):
        """Test joint distribution caching."""
        # Get joint distribution twice
        joint1 = populated_registry.get_joint("VER", "HAM", "race")
        joint2 = populated_registry.get_joint("VER", "HAM", "race")

        # Should be the same object (cached)
        assert joint1 is joint2

        # Should work with reversed order too
        joint3 = populated_registry.get_joint("HAM", "VER", "race")
        assert joint1 is joint3

    def test_clear_joint_cache(self, populated_registry):
        """Test clearing joint cache when registering new distributions."""
        # Get initial joint distribution
        joint1 = populated_registry.get_joint("VER", "HAM", "race")

        # Register new distribution for VER in race context
        new_dist = PositionDistribution({1: 0.8, 2: 0.2})
        populated_registry.register("VER", "race", new_dist)

        # Get joint distribution again - should be different
        joint2 = populated_registry.get_joint("VER", "HAM", "race")
        assert joint1 is not joint2

        # Probabilities should reflect new distribution
        assert joint2[(2, 1)] == pytest.approx(0.2 * 0.6 / (0.8 * 0.4 + 0.2 * 0.6), 5)

    def test_disable_entity(self, populated_registry):
        """Test disabling an entity."""
        # Disable VER
        populated_registry.disable_entity("VER")

        # Should still be able to get distributions but with warning
        with pytest.warns(UserWarning, match="disabled entity"):
            dist = populated_registry.get("VER", "qualifying")
            assert dist[1] == 0.6

        # Should warn when registering for disabled entity
        with pytest.warns(UserWarning, match="disabled but registering"):
            populated_registry.register("VER", "sprint", PositionDistribution({1: 1.0}))

        # Should warn for joint distributions with disabled entity
        with pytest.warns(UserWarning, match="disabled entity"):
            populated_registry.get_joint("VER", "HAM", "race")

    def test_enable_entity(self, populated_registry):
        """Test enabling a previously disabled entity."""
        # Disable then enable
        populated_registry.disable_entity("VER")
        populated_registry.enable_entity("VER")

        # Should be able to get without warning
        dist = populated_registry.get("VER", "qualifying")
        assert dist[1] == 0.6

    def test_clear(self, populated_registry):
        """Test clearing the registry."""
        populated_registry.clear()
        assert not populated_registry.has("VER", "qualifying")
        assert not populated_registry.has("VER", "race")
        assert not populated_registry.has("HAM", "race")
