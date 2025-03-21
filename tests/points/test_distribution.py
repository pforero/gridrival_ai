"""
Tests for the distribution adapter module.

This module tests the DistributionAdapter class, which serves as a bridge
between the probability distribution API and the points calculation system.
"""

from unittest.mock import MagicMock, patch

import pytest

from gridrival_ai.points.distributions import DistributionAdapter
from gridrival_ai.probabilities.distributions import (
    JointDistribution,
    PositionDistribution,
)
from gridrival_ai.probabilities.registry import DistributionRegistry


@pytest.fixture
def position_distribution():
    """Create a sample position distribution."""
    return PositionDistribution({1: 0.6, 2: 0.4})


@pytest.fixture
def joint_distribution():
    """Create a sample joint distribution."""
    probs = {(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
    return JointDistribution(probs, outcome1_name="qual", outcome2_name="race")


@pytest.fixture
def mock_registry(position_distribution, joint_distribution):
    """Create a mock distribution registry."""
    registry = MagicMock(spec=DistributionRegistry)

    # Mock get method
    def mock_get(entity_id, context):
        if context == "qualifying":
            return position_distribution
        elif context == "race":
            return position_distribution
        elif context == "sprint":
            return position_distribution
        elif context == "completion":
            return PositionDistribution({1: 0.95, 2: 0.05})
        elif context == "probability":
            return PositionDistribution({1: 0.9, 2: 0.1})
        elif context == "qual_race":
            return joint_distribution
        elif context == "pair":
            # Create a special mock for pair distributions
            pair_dist = MagicMock(spec=PositionDistribution)
            # Set up the position_probs attribute with integer keys
            pair_dist.position_probs = {1: 0.5, 2: 0.5}
            # Set up __getitem__ to return driver IDs
            pair_dist.__getitem__ = lambda self, key: "VER" if key == 1 else "PER"
            return pair_dist

        raise KeyError(f"No distribution for {entity_id} in {context}")

    registry.get.side_effect = mock_get

    # Mock has method
    def mock_has(entity_id, context):
        if context in ["qualifying", "race", "sprint"]:
            return True
        elif context == "completion" and entity_id == "VER":
            return True
        elif context == "probability" and entity_id.endswith("_completion"):
            return True
        elif context == "qual_race" and entity_id.endswith("_correlation"):
            return True
        elif context == "pair" and entity_id.endswith("_drivers"):
            return True
        return False

    registry.has.side_effect = mock_has

    # Mock get_joint method
    def mock_get_joint(entity1, entity2, context, constrained=True):
        return joint_distribution

    registry.get_joint.side_effect = mock_get_joint

    return registry


@pytest.fixture
def adapter(mock_registry):
    """Create a distribution adapter with mocked registry."""
    return DistributionAdapter(mock_registry)


class TestDistributionAdapter:
    """Test suite for the DistributionAdapter class."""

    def test_get_position_distribution(self, adapter, position_distribution):
        """Test getting position distribution."""
        dist = adapter.get_position_distribution("VER", "qualifying")
        assert dist is position_distribution
        assert dist[1] == 0.6
        assert dist[2] == 0.4

    def test_get_position_distribution_not_found(self, adapter, mock_registry):
        """Test error handling when position distribution not found."""
        # Set up mock to raise KeyError
        mock_registry.get.side_effect = KeyError("No distribution found")

        # Should raise KeyError
        with pytest.raises(KeyError):
            adapter.get_position_distribution("XXX", "qualifying")

    def test_get_position_distribution_safe(self, adapter, position_distribution):
        """Test getting position distribution with fallback."""
        # Existing distribution
        dist = adapter.get_position_distribution_safe("VER", "qualifying")
        assert dist is position_distribution

        # Mock registry to raise KeyError
        adapter.registry.get.side_effect = KeyError("No distribution found")

        # Non-existing distribution, no default
        dist = adapter.get_position_distribution_safe("XXX", "qualifying")
        assert dist is None

        # Non-existing distribution with default
        default_dist = PositionDistribution({1: 0.5, 2: 0.5})
        dist = adapter.get_position_distribution_safe("XXX", "qualifying", default_dist)
        assert dist is default_dist

    def test_get_joint_distribution(self, adapter, joint_distribution):
        """Test getting joint distribution."""
        dist = adapter.get_joint_distribution("VER", "HAM", "race")
        assert dist is joint_distribution
        assert dist[(1, 1)] == 0.4
        assert dist[(2, 2)] == 0.3

    def test_get_joint_distribution_constrained(self, adapter):
        """Test getting joint distribution with constraints."""
        adapter.get_joint_distribution("VER", "HAM", "race", constrained=True)
        # Verify constrained=True was passed
        adapter.registry.get_joint.assert_called_with(
            "VER", "HAM", "race", constrained=True
        )

        adapter.get_joint_distribution("VER", "HAM", "race", constrained=False)
        # Verify constrained=False was passed
        adapter.registry.get_joint.assert_called_with(
            "VER", "HAM", "race", constrained=False
        )

    def test_get_joint_distribution_safe(self, adapter, joint_distribution):
        """Test getting joint distribution with fallback."""
        # Existing joint distribution
        dist = adapter.get_joint_distribution_safe("VER", "HAM", "race")
        assert dist is joint_distribution

        # Mock registry to raise KeyError
        adapter.registry.get_joint.side_effect = KeyError("No distribution found")

        # Non-existing joint distribution, no default
        dist = adapter.get_joint_distribution_safe("XXX", "YYY", "race")
        assert dist is None

        # Non-existing joint distribution with default
        default_dist = JointDistribution({(1, 1): 1.0}, "a", "b")
        dist = adapter.get_joint_distribution_safe(
            "XXX", "YYY", "race", default=default_dist
        )
        assert dist is default_dist

    def test_get_qualifying_race_distribution_strategy1(
        self, adapter, joint_distribution
    ):
        """Test getting qual/race joint distribution - Strategy 1."""
        # Set up registry mock for strategy 1
        adapter.registry.get_joint.return_value = joint_distribution

        dist = adapter.get_qualifying_race_distribution("VER")
        assert dist is joint_distribution

        # Verify attempt to get explicit joint distribution
        adapter.registry.get_joint.assert_called_with(
            "VER_qualifying", "VER_race", "correlation", constrained=False
        )

    def test_get_qualifying_race_distribution_strategy2(
        self, adapter, joint_distribution, mock_registry
    ):
        """Test getting qual/race joint distribution - Strategy 2."""

        # Set up registry mock to fail strategy 1 but succeed for strategy 2
        def mock_get_joint(entity1, entity2, context, constrained=True):
            if "VER_qualifying" in (entity1, entity2):
                raise KeyError("No joint distribution found")
            return joint_distribution

        adapter.registry.get_joint.side_effect = mock_get_joint

        dist = adapter.get_qualifying_race_distribution("VER")
        assert dist is joint_distribution

        # Verify qual_race correlation check
        mock_registry.has.assert_any_call("VER_correlation", "qual_race")

    def test_get_qualifying_race_distribution_fallback(
        self, adapter, position_distribution
    ):
        """Test getting qual/race joint distribution - Fallback strategy."""
        # Set up registry mock to fail both strategies
        adapter.registry.get_joint.side_effect = KeyError("No joint distribution found")

        # Mock has to return True for qualifying and race, but False for correlation
        def mock_has(entity_id, context):
            if context in ["qualifying", "race"]:
                return True
            return False

        adapter.registry.has.side_effect = mock_has

        # Create a mock joint distribution for the fallback
        mock_joint = JointDistribution({(1, 1): 1.0}, "qualifying", "race")

        # Import the module where create_independent_joint is imported
        import gridrival_ai.points.distributions as dist_module

        # Patch the function at the module level
        with patch.object(
            dist_module, "create_independent_joint", return_value=mock_joint
        ):
            dist = adapter.get_qualifying_race_distribution("VER")
            assert dist is mock_joint

    def test_get_qualifying_race_distribution_missing_distributions(self, adapter):
        """Test error when distributions missing for qual/race joint."""

        # Mock has to return False for specific contexts
        def mock_has(entity_id, context):
            # Return False for qualifying or race contexts
            if context in ["qualifying", "race"]:
                return False
            return True

        adapter.registry.has.side_effect = mock_has

        with pytest.raises(KeyError, match="Missing distributions for driver"):
            adapter.get_qualifying_race_distribution("VER")

    def test_get_completion_probability_strategy1(self, adapter):
        """Test getting completion probability - Strategy 1."""
        # Strategy 1: Dedicated completion distribution
        prob = adapter.get_completion_probability("VER")
        assert prob == 0.95

        # Verify completion check
        adapter.registry.has.assert_any_call("VER", "completion")

    def test_get_completion_probability_strategy2(self, adapter, mock_registry):
        """Test getting completion probability - Strategy 2."""

        # Set up registry mock for strategy 2
        def mock_has(entity_id, context):
            if entity_id == "VER" and context == "completion":
                return False
            return True

        adapter.registry.has.side_effect = mock_has

        prob = adapter.get_completion_probability("VER")
        assert prob == 0.9  # From probability distribution

        # Verify probability attribute check
        mock_registry.has.assert_any_call("VER_completion", "probability")

    def test_get_completion_probability_default(self, adapter):
        """Test getting completion probability - Default."""
        # Override the side_effect to always return False
        adapter.registry.has.side_effect = lambda *args, **kwargs: False

        # We don't need to mock get since has will return False
        # and the code won't try to call get

        # Get with default
        prob = adapter.get_completion_probability("VER")
        assert prob == 0.95  # Default

        # Get with custom default
        prob = adapter.get_completion_probability("VER", default=0.85)
        assert prob == 0.85  # Custom default

    def test_get_constructor_drivers_strategy1(self, adapter):
        """Test getting constructor drivers - Strategy 1."""
        # Create a special mock for the pair distribution
        pair_dist = MagicMock(spec=PositionDistribution)
        # Set position_probs with integer keys
        pair_dist.position_probs = {1: 0.5, 2: 0.5}

        # Mock the __getitem__ method to return driver IDs
        pair_dist.__getitem__ = lambda _, key: "VER" if key == 0 else "PER"

        # Mock the registry.get method to return our special mock
        def mock_get(entity_id, context):
            if context == "pair" and entity_id == "RBR_drivers":
                return pair_dist
            raise KeyError("Not found")

        adapter.registry.get.side_effect = mock_get

        # Mock has to return True for the pair context
        def mock_has(entity_id, context):
            return context == "pair" and entity_id == "RBR_drivers"

        adapter.registry.has.side_effect = mock_has

        # Test the method
        drivers = adapter.get_constructor_drivers("RBR")
        assert drivers == ("VER", "PER")

    def test_get_constructor_drivers_strategy2(self, adapter):
        """Test getting constructor drivers - Strategy 2."""

        # Set up registry to fail strategy 1
        def mock_has(entity_id, context):
            # Return False for all has checks to force using strategy 2
            return False

        adapter.registry.has.side_effect = mock_has

        # Make sure get method doesn't interfere
        adapter.registry.get.side_effect = KeyError("Not found")

        # Mock CONSTRUCTORS reference
        with patch(
            "gridrival_ai.points.distributions.CONSTRUCTORS"
        ) as mock_constructors:
            mock_constructor = MagicMock()
            mock_constructor.drivers = ("HAM", "RUS")
            mock_constructors.get.return_value = mock_constructor

            drivers = adapter.get_constructor_drivers("MER")
            assert drivers == ("HAM", "RUS")

            # Verify reference check
            mock_constructors.get.assert_called_with("MER")

    def test_get_constructor_drivers_not_found(self, adapter):
        """Test error when constructor not found."""

        # Set up registry to fail strategy 1
        def mock_has(entity_id, context):
            # Return False for all has checks to force using strategy 2
            return False

        adapter.registry.has.side_effect = mock_has

        # Make sure get method doesn't interfere
        adapter.registry.get.side_effect = KeyError("Not found")

        # Mock CONSTRUCTORS reference to return None
        with patch(
            "gridrival_ai.points.distributions.CONSTRUCTORS"
        ) as mock_constructors:
            mock_constructors.get.return_value = None

            # Should raise KeyError
            with pytest.raises(KeyError, match="Constructor .* not found"):
                adapter.get_constructor_drivers("XXX")


if __name__ == "__main__":
    pytest.main(["-v"])
