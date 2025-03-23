"""Tests for the PositionDistribution class."""

import math

import pytest

from gridrival_ai.probabilities.distributions.joint import JointDistribution
from gridrival_ai.probabilities.distributions.position import (
    DistributionError,
    PositionDistribution,
)
from gridrival_ai.probabilities.distributions.race import RaceDistribution
from gridrival_ai.probabilities.distributions.session import SessionDistribution


class TestPositionDistribution:
    """Test cases for PositionDistribution class."""

    def test_creation_valid_distribution(self):
        """Test creating a valid distribution."""
        # Simple valid distribution
        dist = PositionDistribution({1: 0.6, 2: 0.4})
        assert dist[1] == 0.6
        assert dist[2] == 0.4
        assert dist.is_valid is True

        # Distribution with more positions
        dist = PositionDistribution({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4})
        assert dist.is_valid is True

    def test_validation_missing_positions(self):
        """Test validation for missing positions."""
        # Missing position 2
        with pytest.raises(DistributionError) as excinfo:
            PositionDistribution({1: 0.5, 3: 0.5})
        assert "Missing consecutive positions" in str(excinfo.value)

        # Starting from position 2 instead of 1
        with pytest.raises(DistributionError) as excinfo:
            PositionDistribution({2: 0.5, 3: 0.5})
        assert "must start at position 1" in str(excinfo.value)

    def test_validation_sum_not_one(self):
        """Test validation for probabilities that don't sum to 1."""
        # Sum > 1.0
        with pytest.raises(DistributionError) as excinfo:
            PositionDistribution({1: 0.6, 2: 0.5})
        assert "sum to 1.0" in str(excinfo.value)

        # Sum < 1.0
        with pytest.raises(DistributionError) as excinfo:
            PositionDistribution({1: 0.3, 2: 0.4})
        assert "sum to 1.0" in str(excinfo.value)

    def test_validation_invalid_probabilities(self):
        """Test validation for invalid probability values."""
        # Negative probability
        with pytest.raises(DistributionError) as excinfo:
            PositionDistribution({1: -0.2, 2: 1.2})
        assert "Invalid probabilities" in str(excinfo.value)

        # Probability > 1.0
        with pytest.raises(DistributionError) as excinfo:
            PositionDistribution({1: 1.2, 2: -0.2})
        assert "Invalid probabilities" in str(excinfo.value)

    def test_get_method(self):
        """Test the get method."""
        dist = PositionDistribution({1: 0.7, 2: 0.3})

        # Get existing position
        assert dist.get(1) == 0.7
        assert dist.get(2) == 0.3

        # Get non-existent position
        assert dist.get(3) == 0.0

        # Dictionary-like access
        assert dist[1] == 0.7
        assert dist[2] == 0.3

    def test_is_valid_property(self):
        """Test the is_valid property."""
        # Valid distribution
        dist = PositionDistribution({1: 0.7, 2: 0.3})
        assert dist.is_valid is True

        # Create invalid distribution without validation
        invalid_dist = PositionDistribution({1: 0.7, 2: 0.4}, _validate=False)
        assert invalid_dist.is_valid is False

    def test_normalize_method(self):
        """Test the normalize method."""
        # Create distribution with sum != 1.0 (without validation)
        dist = PositionDistribution({1: 2.0, 2: 3.0}, _validate=False)

        # Normalize
        normalized = dist.normalize()

        # Check normalized values
        assert normalized[1] == 0.4
        assert normalized[2] == 0.6
        assert normalized.is_valid is True

    def test_cumulative_method(self):
        """Test the cumulative method."""
        dist = PositionDistribution({1: 0.3, 2: 0.2, 3: 0.5})
        cumulative = dist.cumulative()

        # Check cumulative probabilities
        assert cumulative[1] == 0.3
        assert cumulative[2] == 0.5
        assert cumulative[3] == 1.0

    def test_to_dict_method(self):
        """Test the to_dict method."""
        original = {1: 0.3, 2: 0.2, 3: 0.5}
        dist = PositionDistribution(original)

        # to_dict should return a copy of position_probs
        result = dist.to_dict()
        assert result == original
        assert (
            result is not dist.position_probs
        )  # Should be a new dict, not the same object

    def test_items_outcomes_probabilities(self):
        """Test the items, outcomes, and probabilities methods."""
        dist = PositionDistribution({1: 0.3, 2: 0.2, 3: 0.5})

        # Test items
        items = list(dist.items())
        assert items == [(1, 0.3), (2, 0.2), (3, 0.5)]

        # Test outcomes
        outcomes = list(dist.outcomes)
        assert outcomes == [1, 2, 3]

        # Test probabilities
        probs = list(dist.probabilities)
        assert probs == [0.3, 0.2, 0.5]

    def test_expected_value_method(self):
        """Test the expected_value method."""
        # Create a position distribution
        dist = PositionDistribution({1: 0.6, 2: 0.4})

        # Test with a complete value dictionary
        points = {1: 25, 2: 18}
        expected_points = 0.6 * 25 + 0.4 * 18
        assert dist.expected_value(points) == expected_points

        # Test with missing values (should default to 0.0)
        incomplete_points = {1: 25}
        expected_points_incomplete = 0.6 * 25 + 0.4 * 0.0
        assert dist.expected_value(incomplete_points) == expected_points_incomplete

        # Test with extra values (should ignore them)
        extra_points = {1: 25, 2: 18, 3: 15, 4: 12}
        expected_points_extra = 0.6 * 25 + 0.4 * 18
        assert dist.expected_value(extra_points) == expected_points_extra

        # Test with empty value dictionary (should return 0.0)
        assert dist.expected_value({}) == 0.0


class TestJointDistribution:
    """Test cases for JointDistribution class."""

    def test_creation_valid_distribution(self):
        """Test creating a valid joint distribution."""
        # Simple valid distribution
        joint = JointDistribution({(1, 1): 0.2, (1, 2): 0.3, (2, 1): 0.5})
        assert joint[(1, 1)] == 0.2
        assert joint[(1, 2)] == 0.3
        assert joint[(2, 1)] == 0.5
        assert joint.is_valid is True

    def test_validation_sum_not_one(self):
        """Test validation for probabilities that don't sum to 1."""
        # Sum > 1.0
        with pytest.raises(DistributionError) as excinfo:
            JointDistribution({(1, 1): 0.5, (1, 2): 0.6})
        assert "sum to 1.0" in str(excinfo.value)

        # Sum < 1.0
        with pytest.raises(DistributionError) as excinfo:
            JointDistribution({(1, 1): 0.3, (1, 2): 0.4})
        assert "sum to 1.0" in str(excinfo.value)

    def test_validation_invalid_probabilities(self):
        """Test validation for invalid probability values."""
        # Negative probability
        with pytest.raises(DistributionError) as excinfo:
            JointDistribution({(1, 1): -0.2, (1, 2): 1.2})
        assert "Invalid probabilities" in str(excinfo.value)

        # Probability > 1.0
        with pytest.raises(DistributionError) as excinfo:
            JointDistribution({(1, 1): 1.2, (1, 2): -0.2})
        assert "Invalid probabilities" in str(excinfo.value)

    def test_get_method(self):
        """Test the get method."""
        joint = JointDistribution({(1, 1): 0.2, (1, 2): 0.3, (2, 1): 0.5})

        # Get existing outcome pair
        assert joint.get((1, 1)) == 0.2
        assert joint.get((1, 2)) == 0.3
        assert joint.get((2, 1)) == 0.5

        # Get non-existent outcome pair
        assert joint.get((2, 2)) == 0.0

        # Dictionary-like access
        assert joint[(1, 1)] == 0.2
        assert joint[(2, 2)] == 0.0

    def test_is_valid_property(self):
        """Test the is_valid property."""
        # Valid distribution
        joint = JointDistribution({(1, 1): 0.2, (1, 2): 0.3, (2, 1): 0.5})
        assert joint.is_valid is True

        # Create invalid distribution without validation
        invalid_joint = JointDistribution({(1, 1): 0.7, (1, 2): 0.4}, _validate=False)
        assert invalid_joint.is_valid is False

    def test_normalize_method(self):
        """Test the normalize method."""
        # Create distribution with sum != 1.0 (without validation)
        joint = JointDistribution({(1, 1): 2.0, (1, 2): 3.0}, _validate=False)

        # Normalize
        normalized = joint.normalize()

        # Check normalized values
        assert normalized[(1, 1)] == 0.4
        assert normalized[(1, 2)] == 0.6
        assert normalized.is_valid is True

    def test_marginal_distributions(self):
        """Test the marginal distribution methods."""
        joint = JointDistribution({(1, 1): 0.2, (1, 2): 0.3, (2, 1): 0.5})

        # Test marginal1
        marg1 = joint.marginal1()
        assert isinstance(marg1, PositionDistribution)
        assert marg1[1] == 0.5  # 0.2 + 0.3
        assert marg1[2] == 0.5  # 0.5

        # Test marginal2
        marg2 = joint.marginal2()
        assert isinstance(marg2, PositionDistribution)
        assert marg2[1] == 0.7  # 0.2 + 0.5
        assert marg2[2] == 0.3  # 0.3

    def test_create_from_distributions_unconstrained(self):
        """Test unconstrained joint distribution from two position distributions."""
        dist1 = PositionDistribution({1: 0.7, 2: 0.3})
        dist2 = PositionDistribution({1: 0.4, 2: 0.6})

        joint = JointDistribution.create_from_distributions(
            dist1, dist2, entity1_name="VER", entity2_name="HAM", constrained=False
        )

        # Check joint probabilities
        assert math.isclose(
            joint[(1, 1)], 0.7 * 0.4 / (0.7 * 0.4 + 0.7 * 0.6 + 0.3 * 0.4 + 0.3 * 0.6)
        )
        assert math.isclose(
            joint[(1, 2)], 0.7 * 0.6 / (0.7 * 0.4 + 0.7 * 0.6 + 0.3 * 0.4 + 0.3 * 0.6)
        )
        assert math.isclose(
            joint[(2, 1)], 0.3 * 0.4 / (0.7 * 0.4 + 0.7 * 0.6 + 0.3 * 0.4 + 0.3 * 0.6)
        )
        assert math.isclose(
            joint[(2, 2)], 0.3 * 0.6 / (0.7 * 0.4 + 0.7 * 0.6 + 0.3 * 0.4 + 0.3 * 0.6)
        )

        # Check entity names
        assert joint.entity1_name == "VER"
        assert joint.entity2_name == "HAM"

    def test_create_from_distributions_constrained(self):
        """Test constrained joint distribution from two position distributions."""
        dist1 = PositionDistribution({1: 0.7, 2: 0.3})
        dist2 = PositionDistribution({1: 0.4, 2: 0.6})

        joint = JointDistribution.create_from_distributions(
            dist1, dist2, constrained=True
        )

        # Check joint probabilities
        assert joint[(1, 1)] == 0.0  # Constrained - can't both be position 1
        assert math.isclose(joint[(1, 2)], 0.7 * 0.6 / (0.7 * 0.6 + 0.3 * 0.4))
        assert math.isclose(joint[(2, 1)], 0.3 * 0.4 / (0.7 * 0.6 + 0.3 * 0.4))
        assert joint[(2, 2)] == 0.0  # Constrained - can't both be position 2


class TestSessionDistribution:
    """Test cases for SessionDistribution class."""

    def test_creation_valid_distribution(self):
        """Test creating a valid session distribution."""
        # Create position distributions for two drivers
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        # Create session distribution
        session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Basic checks
        assert session.session_type == "race"
        assert len(session.driver_distributions) == 2
        assert "VER" in session.driver_distributions
        assert "HAM" in session.driver_distributions

    def test_validation_session_type(self):
        """Test validation for session type."""
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        # Invalid session type
        with pytest.raises(ValueError) as excinfo:
            SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "invalid_type")
        assert "Invalid session type" in str(excinfo.value)

    def test_validation_different_positions(self):
        """Test validation for drivers with different positions."""
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.6, 2: 0.2, 3: 0.2})  # Different positions

        # Different positions
        with pytest.raises(ValueError) as excinfo:
            SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")
        assert "different positions" in str(excinfo.value)

    def test_validation_max_position_matches_drivers(self):
        """Test validation that max position matches number of drivers."""
        dist_2 = PositionDistribution({1: 0.6, 2: 0.4})

        # Three positions but only two drivers
        dist_3 = PositionDistribution({1: 0.3, 2: 0.3, 3: 0.4})

        # Max position doesn't match number of drivers
        with pytest.raises(ValueError) as excinfo:
            SessionDistribution({"VER": dist_2, "HAM": dist_2, "LEC": dist_2}, "race")
        assert "Maximum position" in str(excinfo.value)

        # Valid with three drivers and max position 3
        session = SessionDistribution(
            {"VER": dist_3, "HAM": dist_3, "LEC": dist_3}, "race"
        )
        assert session.session_type == "race"

    def test_get_driver_distribution(self):
        """Test the get_driver_distribution method."""
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Get existing driver
        assert session.get_driver_distribution("VER") is ver_dist
        assert session.get_driver_distribution("HAM") is ham_dist

        # Get non-existent driver
        with pytest.raises(KeyError):
            session.get_driver_distribution("LEC")

    def test_get_position_probabilities(self):
        """Test the get_position_probabilities method."""
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Get probabilities for positions
        pos1_probs = session.get_position_probabilities(1)
        assert pos1_probs == {"VER": 0.6, "HAM": 0.4}

        pos2_probs = session.get_position_probabilities(2)
        assert pos2_probs == {"VER": 0.4, "HAM": 0.6}

        # Non-existent position returns zeros
        pos3_probs = session.get_position_probabilities(3)
        assert pos3_probs == {"VER": 0.0, "HAM": 0.0}

    def test_get_driver_ids(self):
        """Test the get_driver_ids method."""
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Get driver IDs
        driver_ids = session.get_driver_ids()
        assert sorted(driver_ids) == ["HAM", "VER"]

    def test_get_positions(self):
        """Test the get_positions method."""
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Get positions
        positions = session.get_positions()
        assert positions == [1, 2]

        # Empty session
        empty_session = SessionDistribution({}, "race", _validate=False)
        assert empty_session.get_positions() == []

    def test_get_joint_distribution(self):
        """Test the get_joint_distribution method."""
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Get joint distribution
        joint = session.get_joint_distribution("VER", "HAM")

        # Check result is a JointDistribution
        assert isinstance(joint, JointDistribution)

        # Check correct marginals
        marg1 = joint.marginal1()
        marg2 = joint.marginal2()
        assert marg1[1] == 0.6
        assert marg1[2] == 0.4
        assert marg2[1] == 0.4
        assert marg2[2] == 0.6

        # Check correct constraints: one driver can't have the same position as another
        assert joint[(1, 1)] == 0.0
        assert joint[(2, 2)] == 0.0


class TestRaceDistribution:
    """Test cases for RaceDistribution class."""

    def test_creation_with_race_only(self):
        """Test creating a race distribution with only race session."""
        # Create position distributions for drivers
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        # Create race session
        race_session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Create race distribution with only race session
        race_dist = RaceDistribution(race_session)

        # Basic checks
        assert race_dist.race.session_type == "race"
        assert race_dist.qualifying.session_type == "qualifying"
        assert race_dist.sprint.session_type == "sprint"

        # Check that qualifying and sprint were copied from race
        assert set(race_dist.qualifying.get_driver_ids()) == {"VER", "HAM"}
        assert set(race_dist.sprint.get_driver_ids()) == {"VER", "HAM"}

        # Check distributions were copied correctly
        assert race_dist.qualifying.get_driver_distribution("VER")[1] == 0.6
        assert race_dist.sprint.get_driver_distribution("HAM")[2] == 0.6

    def test_creation_with_all_sessions(self):
        """Test creating a race distribution with all sessions specified."""
        # Create position distributions for drivers
        ver_race = PositionDistribution({1: 0.6, 2: 0.4})
        ham_race = PositionDistribution({1: 0.4, 2: 0.6})

        ver_quali = PositionDistribution({1: 0.7, 2: 0.3})
        ham_quali = PositionDistribution({1: 0.3, 2: 0.7})

        ver_sprint = PositionDistribution({1: 0.5, 2: 0.5})
        ham_sprint = PositionDistribution({1: 0.5, 2: 0.5})

        # Create sessions
        race_session = SessionDistribution({"VER": ver_race, "HAM": ham_race}, "race")

        quali_session = SessionDistribution(
            {"VER": ver_quali, "HAM": ham_quali}, "qualifying"
        )

        sprint_session = SessionDistribution(
            {"VER": ver_sprint, "HAM": ham_sprint}, "sprint"
        )

        # Create race distribution with all sessions
        race_dist = RaceDistribution(race_session, quali_session, sprint_session)

        # Check sessions were assigned correctly
        assert race_dist.race is race_session
        assert race_dist.qualifying is quali_session
        assert race_dist.sprint is sprint_session

        # Check driver distributions
        assert race_dist.get_driver_distribution("VER", "race")[1] == 0.6
        assert race_dist.get_driver_distribution("VER", "qualifying")[1] == 0.7
        assert race_dist.get_driver_distribution("VER", "sprint")[1] == 0.5

    def test_validation_session_types(self):
        """Test validation of session types."""
        # Create position distributions for drivers
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        # Create sessions with incorrect types
        wrong_race = SessionDistribution(
            {"VER": ver_dist, "HAM": ham_dist},
            "qualifying",  # Should be "race"
        )

        wrong_quali = SessionDistribution(
            {"VER": ver_dist, "HAM": ham_dist},
            "race",  # Should be "qualifying"
        )

        # Test wrong race type
        with pytest.raises(ValueError) as excinfo:
            RaceDistribution(wrong_race)
        assert "must have session_type 'race'" in str(excinfo.value)

        # Create correct race but wrong qualifying
        correct_race = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        with pytest.raises(ValueError) as excinfo:
            RaceDistribution(correct_race, wrong_quali)
        assert "must have session_type 'qualifying'" in str(excinfo.value)

    def test_validation_different_drivers(self):
        """Test validation for sessions with different drivers."""
        # Create position distributions for different drivers
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})
        lec_dist = PositionDistribution({1: 0.5, 2: 0.5})

        # Create sessions with different drivers
        race_session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        quali_session = SessionDistribution(
            {"VER": ver_dist, "LEC": lec_dist},
            "qualifying",  # Different driver set
        )

        # Test different drivers
        with pytest.raises(ValueError) as excinfo:
            RaceDistribution(race_session, quali_session)
        assert "different drivers" in str(excinfo.value)

    def test_get_session(self):
        """Test the get_session method."""
        # Create position distributions for drivers
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        # Create race session
        race_session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Create race distribution
        race_dist = RaceDistribution(race_session)

        # Get sessions
        assert race_dist.get_session("race") is race_dist.race
        assert race_dist.get_session("qualifying") is race_dist.qualifying
        assert race_dist.get_session("sprint") is race_dist.sprint

        # Invalid session type
        with pytest.raises(ValueError) as excinfo:
            race_dist.get_session("invalid")
        assert "Invalid session type" in str(excinfo.value)

    def test_get_driver_distribution(self):
        """Test the get_driver_distribution method."""
        # Create position distributions for drivers
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        # Create race session
        race_session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Create race distribution
        race_dist = RaceDistribution(race_session)

        # Get driver distributions
        assert race_dist.get_driver_distribution("VER", "race") is ver_dist
        assert race_dist.get_driver_distribution("HAM", "qualifying")[1] == 0.4

        # Non-existent driver
        with pytest.raises(KeyError):
            race_dist.get_driver_distribution("LEC", "race")

    def test_get_driver_ids(self):
        """Test the get_driver_ids method."""
        # Create position distributions for drivers
        ver_dist = PositionDistribution({1: 0.6, 2: 0.4})
        ham_dist = PositionDistribution({1: 0.4, 2: 0.6})

        # Create race session
        race_session = SessionDistribution({"VER": ver_dist, "HAM": ham_dist}, "race")

        # Create race distribution
        race_dist = RaceDistribution(race_session)

        # Get driver IDs
        driver_ids = race_dist.get_driver_ids()
        assert driver_ids == {"VER", "HAM"}


def test_create_independent_joint():
    """Test the create_independent_joint function."""
    from gridrival_ai.probabilities.distributions import create_independent_joint

    # Create position distributions
    dist1 = PositionDistribution({1: 0.7, 2: 0.3})
    dist2 = PositionDistribution({1: 0.4, 2: 0.6})

    # Create independent joint distribution
    joint = create_independent_joint(dist1, dist2, "VER", "HAM")

    # Check joint probabilities
    assert math.isclose(joint[(1, 1)], 0.7 * 0.4)
    assert math.isclose(joint[(1, 2)], 0.7 * 0.6)
    assert math.isclose(joint[(2, 1)], 0.3 * 0.4)
    assert math.isclose(joint[(2, 2)], 0.3 * 0.6)

    # Check entity names
    assert joint.entity1_name == "VER"
    assert joint.entity2_name == "HAM"

    # Make sure joint is valid
    assert joint.is_valid


def test_create_constrained_joint():
    """Test the create_constrained_joint function."""
    from gridrival_ai.probabilities.distributions import create_constrained_joint

    # Create position distributions
    dist1 = PositionDistribution({1: 0.7, 2: 0.3})
    dist2 = PositionDistribution({1: 0.4, 2: 0.6})

    # Create constrained joint distribution
    joint = create_constrained_joint(dist1, dist2, "VER", "HAM")

    # Check joint probabilities - same positions should have 0 probability
    assert joint[(1, 1)] == 0.0
    assert joint[(2, 2)] == 0.0

    # Other positions should be normalized
    total_prob = 0.7 * 0.6 + 0.3 * 0.4  # for valid combinations
    assert math.isclose(joint[(1, 2)], 0.7 * 0.6 / total_prob)
    assert math.isclose(joint[(2, 1)], 0.3 * 0.4 / total_prob)

    # Check entity names
    assert joint.entity1_name == "VER"
    assert joint.entity2_name == "HAM"

    # Make sure probabilities sum to 1.0
    assert math.isclose(sum(joint.probabilities), 1.0)

    # Make sure joint is valid
    assert joint.is_valid
