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

    def test_register(self, registry, sample_dist):
        """Test registering a distribution."""
        registry.register("VER", "qualifying", sample_dist)
        assert "VER" in registry.distributions
        assert "qualifying" in registry.distributions["VER"]
        assert registry.distributions["VER"]["qualifying"] is sample_dist

    def test_get(self, registry, sample_dist):
        """Test all get functionality."""
        # Test getting a missing distribution without default (should raise KeyError)
        with pytest.raises(KeyError, match="No distribution found"):
            registry.get("VER", "qualifying")

        # Test getting a missing distribution with default
        default = PositionDistribution({1: 0.5, 2: 0.5})
        assert registry.get("VER", "qualifying", default) is default

        # Register a distribution
        registry.register("VER", "qualifying", sample_dist)

        # Test getting an existing distribution
        retrieved = registry.get("VER", "qualifying")
        assert retrieved is sample_dist
        assert retrieved[1] == 0.6

        # Test getting an existing distribution with default (should ignore default)
        assert registry.get("VER", "qualifying", default) is sample_dist

    def test_has(self, registry, sample_dist):
        """Test checking if a distribution exists."""
        assert not registry.has("VER", "qualifying")
        registry.register("VER", "qualifying", sample_dist)
        assert registry.has("VER", "qualifying")

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
        assert joint.outcome1_name == "VER"
        assert joint.outcome2_name == "HAM"

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
