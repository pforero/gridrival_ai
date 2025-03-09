"""Tests for core probability distribution classes."""

import pytest

from gridrival_ai.probabilities.core import (
    DistributionError,
    JointDistribution,
    PositionDistribution,
    create_conditional_joint,
    create_constrained_joint,
    create_independent_joint,
)


class TestPositionDistribution:
    """Test suite for PositionDistribution class."""

    def test_initialization(self):
        """Test basic initialization."""
        dist = PositionDistribution({1: 0.6, 2: 0.4})
        assert dist[1] == 0.6
        assert dist[2] == 0.4
        assert dist.is_valid

    def test_get_missing_position(self):
        """Test getting probability for position not in distribution."""
        dist = PositionDistribution({1: 0.6, 2: 0.4})
        assert dist[3] == 0.0

    def test_invalid_position_validation(self):
        """Test validation of invalid positions."""
        with pytest.raises(DistributionError, match="Invalid positions"):
            PositionDistribution({0: 0.5, 1: 0.5})

        with pytest.raises(DistributionError, match="Invalid positions"):
            PositionDistribution({1: 0.5, 21: 0.5})

    def test_probability_sum_validation(self):
        """Test validation of probability sum."""
        with pytest.raises(DistributionError, match="must sum to 1.0"):
            PositionDistribution({1: 0.6, 2: 0.7})

    def test_negative_probability_validation(self):
        """Test validation of negative probabilities."""
        with pytest.raises(DistributionError, match="Invalid probabilities"):
            PositionDistribution({1: -0.1, 2: 1.1})

    def test_validation_skipping(self):
        """Test skipping validation."""
        # This would normally raise an error
        dist = PositionDistribution({1: 0.6, 2: 0.7}, _validate=False)
        assert not dist.is_valid
        assert dist[1] == 0.6
        assert dist[2] == 0.7

    def test_normalization(self):
        """Test probability normalization."""
        unnormalized = PositionDistribution({1: 2.0, 2: 3.0}, _validate=False)
        normalized = unnormalized.normalize()
        assert normalized[1] == 0.4
        assert normalized[2] == 0.6
        assert normalized.is_valid

    def test_normalization_zero_sum(self):
        """Test normalization of zero-sum distribution."""
        with pytest.raises(DistributionError, match="zero-sum distribution"):
            PositionDistribution({1: 0.0, 2: 0.0}, _validate=False).normalize()

    def test_filter_positions(self):
        """Test position filtering."""
        dist = PositionDistribution({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4})
        top2 = dist.filter_positions(1, 2)
        assert top2[1] == pytest.approx(0.1 / 0.3)
        assert top2[2] == pytest.approx(0.2 / 0.3)
        assert top2[3] == 0.0
        assert top2[4] == 0.0
        assert top2.is_valid

    def test_filter_no_positions(self):
        """Test filtering with no positions in range."""
        dist = PositionDistribution({5: 0.5, 6: 0.5})
        with pytest.raises(DistributionError):
            dist.filter_positions(1, 3)

    def test_cumulative(self):
        """Test cumulative distribution."""
        dist = PositionDistribution({1: 0.2, 3: 0.3, 5: 0.5})
        cdf = dist.cumulative()
        assert cdf[1] == 0.2
        assert cdf[3] == 0.5
        assert cdf[5] == 1.0

    def test_expectation(self):
        """Test expectation calculation."""
        dist = PositionDistribution({1: 0.2, 3: 0.3, 5: 0.5})
        assert dist.expectation() == 0.2 * 1 + 0.3 * 3 + 0.5 * 5

    def test_blend(self):
        """Test blending of distributions."""
        dist1 = PositionDistribution({1: 0.8, 2: 0.2})
        dist2 = PositionDistribution({1: 0.2, 2: 0.8})

        blend1 = dist1.blend(dist2, 0.5)  # Equal weight
        assert blend1[1] == 0.5
        assert blend1[2] == 0.5

        blend2 = dist1.blend(dist2, 0.8)  # More weight to dist1
        assert blend2[1] == pytest.approx(0.68)
        assert blend2[2] == pytest.approx(0.32)

    def test_blend_different_positions(self):
        """Test blending of distributions with different positions."""
        dist1 = PositionDistribution({1: 0.8, 2: 0.2})
        dist2 = PositionDistribution({1: 0.2, 3: 0.8})

        blend = dist1.blend(dist2, 0.5)
        assert blend[1] == 0.5
        assert blend[2] == 0.1
        assert blend[3] == 0.4
        assert blend.is_valid

    def test_invalid_blend_weight(self):
        """Test error on invalid blend weight."""
        dist1 = PositionDistribution({1: 0.8, 2: 0.2})
        dist2 = PositionDistribution({1: 0.2, 2: 0.8})

        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            dist1.blend(dist2, 1.5)

    def test_smooth(self):
        """Test smoothing of distribution."""
        dist = PositionDistribution({1: 1.0, 2: 0.0})
        smoothed = dist.smooth(0.2)  # 20% smoothing

        # Position 1 should still have highest probability
        assert smoothed[1] > smoothed[2]
        # But less than the original
        assert smoothed[1] < 1.0
        # Total should still sum to 1.0
        assert sum(smoothed.position_probs.values()) == pytest.approx(1.0)

    def test_invalid_smooth_alpha(self):
        """Test error on invalid smoothing parameter."""
        dist = PositionDistribution({1: 1.0})
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            dist.smooth(1.5)

    def test_expected_value(self):
        """Test expected value calculation."""
        dist = PositionDistribution({1: 0.3, 2: 0.7})
        values = {1: 25, 2: 18}
        assert dist.expected_value(values) == 0.3 * 25 + 0.7 * 18
        assert dist.expected_value(values) == 20.1

    def test_entropy(self):
        """Test entropy calculation."""
        # Deterministic distribution (entropy = 0)
        certain = PositionDistribution({1: 1.0})
        assert certain.entropy() == 0.0

        # Uniform distribution over 2 outcomes (max entropy = 1 bit)
        uniform = PositionDistribution({1: 0.5, 2: 0.5})
        assert uniform.entropy() == pytest.approx(1.0)

        # Skewed distribution
        skewed = PositionDistribution({1: 0.9, 2: 0.1})
        assert 0.0 < skewed.entropy() < 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        probs = {1: 0.6, 2: 0.4}
        dist = PositionDistribution(probs)
        assert dist.to_dict() == probs


class TestJointDistribution:
    """Test suite for JointDistribution class."""

    def test_initialization(self):
        """Test basic initialization."""
        dist = JointDistribution({(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3})
        assert dist[(1, 1)] == 0.4
        assert dist[(1, 2)] == 0.2
        assert dist[(2, 1)] == 0.1
        assert dist[(2, 2)] == 0.3
        assert dist.is_valid

    def test_get_missing_outcome(self):
        """Test getting probability for outcome not in distribution."""
        dist = JointDistribution({(1, 1): 0.4, (1, 2): 0.6})
        assert dist[(2, 1)] == 0.0

    def test_probability_sum_validation(self):
        """Test validation of probability sum."""
        with pytest.raises(DistributionError, match="must sum to 1.0"):
            JointDistribution({(1, 1): 0.4, (1, 2): 0.7})

    def test_negative_probability_validation(self):
        """Test validation of negative probabilities."""
        with pytest.raises(DistributionError, match="Invalid probabilities"):
            JointDistribution({(1, 1): -0.1, (1, 2): 1.1})

    def test_validation_skipping(self):
        """Test skipping validation."""
        # This would normally raise an error
        dist = JointDistribution({(1, 1): 0.4, (1, 2): 0.7}, _validate=False)
        assert not dist.is_valid
        assert dist[(1, 1)] == 0.4
        assert dist[(1, 2)] == 0.7

    def test_normalization(self):
        """Test probability normalization."""
        unnormalized = JointDistribution({(1, 1): 2.0, (1, 2): 3.0}, _validate=False)
        normalized = unnormalized.normalize()
        assert normalized[(1, 1)] == 0.4
        assert normalized[(1, 2)] == 0.6
        assert normalized.is_valid

    def test_normalization_zero_sum(self):
        """Test normalization of zero-sum distribution."""
        with pytest.raises(DistributionError, match="zero-sum distribution"):
            JointDistribution({(1, 1): 0.0, (1, 2): 0.0}, _validate=False).normalize()

    def test_marginal1(self):
        """Test first marginal distribution."""
        joint = JointDistribution({(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3})
        marginal = joint.marginal1()
        assert marginal[1] == pytest.approx(0.6)  # P(X=1) = 0.4 + 0.2
        assert marginal[2] == pytest.approx(0.4)  # P(X=2) = 0.1 + 0.3
        assert marginal.is_valid

    def test_marginal2(self):
        """Test second marginal distribution."""
        joint = JointDistribution({(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3})
        marginal = joint.marginal2()
        assert marginal[1] == 0.5  # P(Y=1) = 0.4 + 0.1
        assert marginal[2] == 0.5  # P(Y=2) = 0.2 + 0.3
        assert marginal.is_valid

    def test_conditional1(self):
        """Test first conditional distribution."""
        joint = JointDistribution({(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3})
        # P(X|Y=1)
        cond = joint.conditional1(1)
        assert cond[1] == 0.8  # P(X=1|Y=1) = 0.4 / 0.5
        assert cond[2] == 0.2  # P(X=2|Y=1) = 0.1 / 0.5
        assert cond.is_valid

        # P(X|Y=2)
        cond2 = joint.conditional1(2)
        assert cond2[1] == 0.4  # P(X=1|Y=2) = 0.2 / 0.5
        assert cond2[2] == 0.6  # P(X=2|Y=2) = 0.3 / 0.5
        assert cond2.is_valid

    def test_conditional2(self):
        """Test second conditional distribution."""
        joint = JointDistribution({(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3})
        # P(Y|X=1)
        cond = joint.conditional2(1)
        assert cond[1] == pytest.approx(2 / 3)  # P(Y=1|X=1) = 0.4 / 0.6
        assert cond[2] == pytest.approx(1 / 3)  # P(Y=2|X=1) = 0.2 / 0.6
        assert cond.is_valid

        # P(Y|X=2)
        cond2 = joint.conditional2(2)
        assert cond2[1] == pytest.approx(0.25)  # P(Y=1|X=2) = 0.1 / 0.4
        assert cond2[2] == pytest.approx(0.75)  # P(Y=2|X=2) = 0.3 / 0.4
        assert cond2.is_valid

    def test_conditional_zero_marginal(self):
        """Test error on conditional with zero marginal."""
        joint = JointDistribution({(1, 1): 0.5, (1, 2): 0.5})
        with pytest.raises(DistributionError, match="P\\(var2=3\\) = 0"):
            joint.conditional1(3)

        with pytest.raises(DistributionError, match="P\\(var1=2\\) = 0"):
            joint.conditional2(2)

    def test_custom_variable_names(self):
        """Test custom variable names."""
        joint = JointDistribution(
            {(1, 1): 0.4, (1, 2): 0.6}, outcome1_name="qualifying", outcome2_name="race"
        )

        # Names should be used in error messages
        with pytest.raises(DistributionError, match="P\\(race=3\\) = 0"):
            joint.conditional1(3)

        with pytest.raises(DistributionError, match="P\\(qualifying=2\\) = 0"):
            joint.conditional2(2)

    def test_correlation(self):
        """Test correlation calculation."""
        # Perfect positive correlation
        perfect_pos = JointDistribution({(1, 1): 0.5, (2, 2): 0.5})
        assert perfect_pos.get_correlation() == pytest.approx(1.0)

        # Perfect negative correlation
        perfect_neg = JointDistribution({(1, 2): 0.5, (2, 1): 0.5})
        assert perfect_neg.get_correlation() == pytest.approx(-1.0)

        # Independence (zero correlation)
        independent = JointDistribution(
            {(1, 1): 0.25, (1, 2): 0.25, (2, 1): 0.25, (2, 2): 0.25}
        )
        assert independent.get_correlation() == pytest.approx(0.0, abs=1e-10)

    def test_correlation_with_zero_variance(self):
        """Test correlation with zero variance."""
        # No variance in X
        zero_var_x = JointDistribution({(1, 1): 0.5, (1, 2): 0.5})
        assert zero_var_x.get_correlation() == 0.0

        # No variance in Y
        zero_var_y = JointDistribution({(1, 1): 0.5, (2, 1): 0.5})
        assert zero_var_y.get_correlation() == 0.0

    def test_mutual_information(self):
        """Test mutual information calculation."""
        # Independence (MI = 0)
        independent = JointDistribution(
            {(1, 1): 0.25, (1, 2): 0.25, (2, 1): 0.25, (2, 2): 0.25}
        )
        assert independent.mutual_information() == pytest.approx(0.0, abs=1e-10)

        # Perfect dependence (1 bit of information)
        dependent = JointDistribution({(1, 1): 0.5, (2, 2): 0.5})
        assert dependent.mutual_information() == pytest.approx(1.0)

        # Partial dependence
        partial = JointDistribution(
            {(1, 1): 0.4, (1, 2): 0.1, (2, 1): 0.1, (2, 2): 0.4}
        )
        assert 0.0 < partial.mutual_information() < 1.0

    def test_is_independent(self):
        """Test independence check."""
        # Independent
        independent = JointDistribution(
            {(1, 1): 0.25, (1, 2): 0.25, (2, 1): 0.25, (2, 2): 0.25}
        )
        assert independent.is_independent()

        # Dependent
        dependent = JointDistribution({(1, 1): 0.5, (2, 2): 0.5})
        assert not dependent.is_independent()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        probs = {(1, 1): 0.4, (1, 2): 0.6}
        dist = JointDistribution(probs)
        assert dist.to_dict() == probs


class TestJointConstructionFunctions:
    """Test suite for joint distribution construction functions."""

    def test_create_independent_joint(self):
        """Test creating independent joint distribution."""
        dist1 = PositionDistribution({1: 0.6, 2: 0.4})
        dist2 = PositionDistribution({1: 0.3, 2: 0.7})

        joint = create_independent_joint(dist1, dist2)

        # Check values
        assert joint[(1, 1)] == pytest.approx(0.18)  # 0.6 * 0.3
        assert joint[(1, 2)] == pytest.approx(0.42)  # 0.6 * 0.7
        assert joint[(2, 1)] == pytest.approx(0.12)  # 0.4 * 0.3
        assert joint[(2, 2)] == pytest.approx(0.28)  # 0.4 * 0.7

        # Check independence
        assert joint.is_independent()

        # Check marginals
        marginal1 = joint.marginal1()
        assert marginal1[1] == pytest.approx(0.6)
        assert marginal1[2] == pytest.approx(0.4)

        marginal2 = joint.marginal2()
        assert marginal2[1] == pytest.approx(0.3)
        assert marginal2[2] == pytest.approx(0.7)

    def test_create_constrained_joint(self):
        """Test creating constrained joint distribution."""
        dist1 = PositionDistribution({1: 0.6, 2: 0.4})
        dist2 = PositionDistribution({1: 0.3, 2: 0.7})

        joint = create_constrained_joint(dist1, dist2)

        # Check constraint is enforced
        assert joint[(1, 1)] == 0.0
        assert joint[(2, 2)] == 0.0

        # Check values are normalized
        assert joint[(1, 2)] == pytest.approx(0.6 * 0.7 / (0.6 * 0.7 + 0.4 * 0.3))
        assert joint[(2, 1)] == pytest.approx(0.4 * 0.3 / (0.6 * 0.7 + 0.4 * 0.3))

        # Check is valid
        assert joint.is_valid

        # Check not independent
        assert not joint.is_independent()

    def test_create_constrained_joint_error(self):
        "Test error when creating constrained joint with incompatible distributions."
        dist1 = PositionDistribution({1: 1.0})
        dist2 = PositionDistribution({1: 1.0})

        with pytest.raises(DistributionError, match="non-overlapping outcomes"):
            create_constrained_joint(dist1, dist2)

    def test_create_conditional_joint(self):
        """Test creating joint distribution from marginal and conditional."""
        # Define marginal distribution
        marginal = PositionDistribution({1: 0.6, 2: 0.4})

        # Define conditional distributions
        def get_conditional(x):
            if x == 1:
                return PositionDistribution({1: 0.1, 2: 0.9})
            else:  # x == 2
                return PositionDistribution({1: 0.8, 2: 0.2})

        joint = create_conditional_joint(marginal, get_conditional)

        # Check values
        assert joint[(1, 1)] == pytest.approx(0.06)  # 0.6 * 0.1
        assert joint[(1, 2)] == pytest.approx(0.54)  # 0.6 * 0.9
        assert joint[(2, 1)] == pytest.approx(0.32)  # 0.4 * 0.8
        assert joint[(2, 2)] == pytest.approx(0.08)  # 0.4 * 0.2

        # Check marginals match
        marginal1 = joint.marginal1()
        assert marginal1[1] == pytest.approx(0.6)
        assert marginal1[2] == pytest.approx(0.4)

        # Check conditionals match
        cond1 = joint.conditional2(1)
        assert cond1[1] == pytest.approx(0.1)
        assert cond1[2] == pytest.approx(0.9)

        cond2 = joint.conditional2(2)
        assert cond2[1] == pytest.approx(0.8)
        assert cond2[2] == pytest.approx(0.2)
