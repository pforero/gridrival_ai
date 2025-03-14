"""
Core probability distribution classes.

This module provides the fundamental distribution types for representing and
manipulating probability distributions in racing contexts. These distributions
are used to model finishing positions and other probabilistic outcomes.

Classes
-------
Distribution : ABC
    Abstract base class for all probability distributions.
PositionDistribution : Distribution
    Probability distribution for racing positions.
JointDistribution : Distribution
    Joint probability distribution between two variables.
DistributionError
    Exception raised for distribution validation errors.

Examples
--------
>>> # Create a position distribution
>>> pos_dist = PositionDistribution({1: 0.6, 2: 0.4})
>>> pos_dist[1]  # Get probability of P1
0.6

>>> # Create a joint distribution
>>> joint_dist = JointDistribution({(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3})
>>> joint_dist[(1, 1)]  # Get probability of (P1, P1)
0.4

>>> # Get marginal distribution
>>> marginal = joint_dist.marginal1()
>>> marginal[1]  # Probability of P1 in first variable
0.6
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple, TypeVar

# Type aliases
OutcomeType = TypeVar("OutcomeType")
JointOutcomeType = Tuple[Any, Any]
DistributionMapping = Dict[OutcomeType, float]
JointMapping = Dict[JointOutcomeType, float]

# Constants
TOLERANCE = 1e-6
MAX_POSITION = 20


class DistributionError(Exception):
    """Exception raised for distribution validation errors."""

    pass


class Distribution(ABC):
    """
    Abstract base class for all probability distributions.

    This class defines the common interface for working with probability
    distributions. Concrete implementations must provide methods for
    accessing probabilities and iterating over outcomes.

    See Also
    --------
    PositionDistribution : Distribution for race positions
    JointDistribution : Joint distribution between two variables
    """

    @abstractmethod
    def get(self, outcome: Any) -> float:
        """
        Get probability of a specific outcome.

        Parameters
        ----------
        outcome : Any
            The outcome to get probability for.

        Returns
        -------
        float
            Probability of the outcome.
        """
        pass

    @abstractmethod
    def items(self) -> Iterator[Tuple[Any, float]]:
        """
        Iterate over (outcome, probability) pairs.

        Returns
        -------
        Iterator[Tuple[Any, float]]
            Iterator over outcome-probability pairs.
        """
        pass

    @property
    def outcomes(self) -> Iterator[Any]:
        """
        Iterate over all outcomes with non-zero probability.

        Returns
        -------
        Iterator[Any]
            Iterator over outcomes.
        """
        return (outcome for outcome, _ in self.items())

    @property
    def probabilities(self) -> Iterator[float]:
        """
        Iterate over all probabilities.

        Returns
        -------
        Iterator[float]
            Iterator over probabilities.
        """
        return (prob for _, prob in self.items())

    @property
    def is_valid(self) -> bool:
        """
        Check if distribution is valid.

        A valid distribution has probabilities that sum to 1.0
        (within tolerance) and all probabilities are between 0 and 1.

        Returns
        -------
        bool
            True if distribution is valid, False otherwise.
        """
        total = sum(self.probabilities)
        if not math.isclose(total, 1.0, abs_tol=TOLERANCE):
            return False

        return all(0.0 <= p <= 1.0 for p in self.probabilities)

    def validate(self) -> None:
        """
        Validate distribution and raise error if invalid.

        Raises
        ------
        DistributionError
            If distribution is invalid.
        """
        # Get all probabilities as a list
        probs_list = list(self.probabilities)

        # If the distribution is empty, it's valid
        if not probs_list:
            return

        total = sum(probs_list)
        if not math.isclose(total, 1.0, abs_tol=TOLERANCE):
            raise DistributionError(f"Probabilities must sum to 1.0 (got {total:.6f})")

        invalid_probs = [p for p in probs_list if p < 0.0 or p > 1.0]
        if invalid_probs:
            raise DistributionError(
                f"Invalid probabilities found: {invalid_probs[:5]}..."
                if len(invalid_probs) > 5
                else f"Invalid probabilities found: {invalid_probs}"
            )

    @abstractmethod
    def normalize(self) -> "Distribution":
        """
        Normalize probabilities to sum to 1.0.

        Returns
        -------
        Distribution
            Normalized distribution.
        """
        pass

    def expected_value(self, values: Dict[Any, float]) -> float:
        """
        Calculate expected value given outcome values.

        Parameters
        ----------
        values : Dict[Any, float]
            Mapping from outcomes to their values.

        Returns
        -------
        float
            Expected value.

        Examples
        --------
        >>> dist = PositionDistribution({1: 0.7, 2: 0.2, 3: 0.1})
        >>> points = {1: 25, 2: 18, 3: 15}
        >>> dist.expected_value(points)
        22.5
        """
        return sum(self.get(outcome) * values[outcome] for outcome in values)

    def entropy(self) -> float:
        """
        Calculate Shannon entropy of the distribution.

        Returns
        -------
        float
            Entropy in bits.

        Examples
        --------
        >>> # Uniform distribution has maximum entropy
        >>> uniform = PositionDistribution({i: 1/3 for i in range(1, 4)})
        >>> uniform.entropy()
        1.5849625007211563

        >>> # Deterministic distribution has zero entropy
        >>> certain = PositionDistribution({1: 1.0, 2: 0.0})
        >>> certain.entropy()
        0.0
        """
        entropy = 0.0
        for _, p in self.items():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def __getitem__(self, outcome: Any) -> float:
        """
        Get probability for an outcome using dictionary-like access.

        Parameters
        ----------
        outcome : Any
            The outcome to get probability for.

        Returns
        -------
        float
            Probability of the outcome.
        """
        return self.get(outcome)


@dataclass(frozen=True)
class PositionDistribution(Distribution):
    """
    Probability distribution for racing positions.

    This class represents a probability distribution over racing positions,
    such as the probability of a driver finishing in each position.

    Parameters
    ----------
    position_probs : Dict[int, float]
        Mapping from positions to probabilities.
    validate : bool, optional
        Whether to validate probabilities, by default True.

    Raises
    ------
    DistributionError
        If validation fails and validate=True.

    Examples
    --------
    >>> # Create a position distribution
    >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
    >>> dist[1]  # Probability of P1
    0.6
    >>> dist[2]  # Probability of P2
    0.4
    >>> list(dist.items())
    [(1, 0.6), (2, 0.4)]
    """

    position_probs: Dict[int, float]
    _validate: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        """Validate probabilities if required."""
        if self._validate:
            if not self.position_probs:
                raise DistributionError("Cannot create empty distribution")
            self._validate_positions()
            self.validate()

    def _validate_positions(self) -> None:
        """
        Validate position keys are within valid range.

        Raises
        ------
        DistributionError
            If positions are not valid.
        """
        invalid_pos = [
            p for p in self.position_probs.keys() if not 1 <= p <= MAX_POSITION
        ]
        if invalid_pos:
            raise DistributionError(
                f"Invalid positions found: {invalid_pos} "
                f"(must be between 1 and {MAX_POSITION})"
            )

    def get(self, position: int) -> float:
        """
        Get probability for a specific position.

        Parameters
        ----------
        position : int
            Racing position (1-based).

        Returns
        -------
        float
            Probability of finishing in that position.
        """
        return self.position_probs.get(position, 0.0)

    def items(self) -> Iterator[Tuple[int, float]]:
        """
        Iterate over (position, probability) pairs.

        Returns
        -------
        Iterator[Tuple[int, float]]
            Iterator over position-probability pairs.
        """
        return iter(self.position_probs.items())

    def normalize(self) -> "PositionDistribution":
        """
        Normalize probabilities to sum to 1.0.

        Returns
        -------
        PositionDistribution
            Normalized distribution.

        Examples
        --------
        >>> dist = PositionDistribution({1: 2.0, 2: 3.0}, validate=False)
        >>> normalized = dist.normalize()
        >>> normalized[1]
        0.4
        >>> normalized[2]
        0.6
        """
        total = sum(self.position_probs.values())
        if math.isclose(total, 0.0, abs_tol=TOLERANCE):
            raise DistributionError("Cannot normalize zero-sum distribution")

        normalized = {k: v / total for k, v in self.position_probs.items()}
        return PositionDistribution(normalized)

    def filter_positions(
        self, min_pos: int = 1, max_pos: int = MAX_POSITION
    ) -> "PositionDistribution":
        """
        Filter to positions within range and renormalize.

        Parameters
        ----------
        min_pos : int, optional
            Minimum position to include, by default 1.
        max_pos : int, optional
            Maximum position to include, by default MAX_POSITION.

        Returns
        -------
        PositionDistribution
            Filtered and normalized distribution.

        Examples
        --------
        >>> dist = PositionDistribution({1: 0.2, 2: 0.3, 3: 0.4, 4: 0.1})
        >>> top3 = dist.filter_positions(1, 3)
        >>> top3[1]
        0.2222222222222222
        >>> top3[2]
        0.3333333333333333
        >>> top3[3]
        0.4444444444444444
        >>> top3[4]
        0.0
        """
        filtered = {
            k: v for k, v in self.position_probs.items() if min_pos <= k <= max_pos
        }

        # If nothing passed the filter, return empty distribution
        if not filtered:
            raise DistributionError(
                f"Filter resulted in empty distribution. No positions between {min_pos}"
                f" and {max_pos}."
            )

        total = sum(filtered.values())
        normalized = {k: v / total for k, v in filtered.items()}
        return PositionDistribution(normalized)

    def cumulative(self) -> Dict[int, float]:
        """
        Get cumulative distribution function.

        Returns
        -------
        Dict[int, float]
            Mapping from positions to cumulative probabilities.

        Examples
        --------
        >>> dist = PositionDistribution({1: 0.3, 2: 0.2, 3: 0.5})
        >>> dist.cumulative()
        {1: 0.3, 2: 0.5, 3: 1.0}
        """
        result = {}
        total = 0.0
        for pos in sorted(self.position_probs.keys()):
            total += self.position_probs[pos]
            result[pos] = total
        return result

    def expectation(self) -> float:
        """
        Calculate expected position.

        Returns
        -------
        float
            Expected position.

        Examples
        --------
        >>> dist = PositionDistribution({1: 0.3, 2: 0.2, 3: 0.5})
        >>> dist.expectation()
        2.2
        """
        return sum(pos * prob for pos, prob in self.items())

    def blend(
        self, other: "PositionDistribution", weight: float = 0.5
    ) -> "PositionDistribution":
        """
        Blend with another distribution using weighted average.

        Parameters
        ----------
        other : PositionDistribution
            Other distribution to blend with.
        weight : float, optional
            Weight of current distribution, by default 0.5.
            The weight of the other distribution is (1 - weight).

        Returns
        -------
        PositionDistribution
            Blended distribution.

        Examples
        --------
        >>> dist1 = PositionDistribution({1: 0.8, 2: 0.2})
        >>> dist2 = PositionDistribution({1: 0.2, 2: 0.8})
        >>> blended = dist1.blend(dist2, 0.7)
        >>> blended[1]
        0.62
        >>> blended[2]
        0.38
        """
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")

        # Get all unique positions
        all_positions = set(self.position_probs.keys()) | set(
            other.position_probs.keys()
        )

        # Blend probabilities
        blended = {}
        for pos in all_positions:
            p1 = self.get(pos)
            p2 = other.get(pos)
            blended[pos] = weight * p1 + (1 - weight) * p2

        return PositionDistribution(blended)

    def smooth(self, alpha: float = 0.1) -> "PositionDistribution":
        """
        Apply smoothing to the distribution.

        Parameters
        ----------
        alpha : float, optional
            Smoothing parameter, by default 0.1.
            Higher values create more uniform distributions.

        Returns
        -------
        PositionDistribution
            Smoothed distribution.

        Examples
        --------
        >>> dist = PositionDistribution({1: 1.0})
        >>> smoothed = dist.smooth(0.2)
        >>> smoothed[1]  # Highest but less than 1.0
        0.8333333333333334
        >>> smoothed[2]  # Small probability of adjacent positions
        0.08333333333333333
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        # Get all positions with non-zero probability
        positions = sorted(self.position_probs.keys())
        if not positions:
            return PositionDistribution({})

        # Get min and max positions
        min_pos = min(positions)
        max_pos = max(positions)

        # Apply smoothing
        smoothed = {}
        for pos in range(min_pos, max_pos + 1):
            # Original probability (0 if not in original distribution)
            orig_prob = self.get(pos)

            # Apply smoothing
            smoothed[pos] = (1 - alpha) * orig_prob + alpha / (max_pos - min_pos + 1)

        return PositionDistribution(smoothed)

    def to_dict(self) -> Dict[int, float]:
        """
        Convert to dictionary.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping from positions to probabilities.
        """
        return dict(self.position_probs)


@dataclass(frozen=True)
class JointDistribution(Distribution):
    """
    Joint probability distribution between two variables.

    This class represents a joint probability distribution between
    two variables, such as qualifying and race positions.

    Parameters
    ----------
    joint_probs : Dict[Tuple[Any, Any], float]
        Mapping from outcome pairs to probabilities.
    outcome1_name : str, optional
        Name of first variable, by default "var1".
    outcome2_name : str, optional
        Name of second variable, by default "var2".
    validate : bool, optional
        Whether to validate probabilities, by default True.

    Raises
    ------
    DistributionError
        If validation fails and validate=True.

    Examples
    --------
    >>> # Create a joint distribution for qualifying and race
    >>> joint = JointDistribution({
    ...     (1, 1): 0.4,  # P1 in qualifying, P1 in race
    ...     (1, 2): 0.2,  # P1 in qualifying, P2 in race
    ...     (2, 1): 0.1,  # P2 in qualifying, P1 in race
    ...     (2, 2): 0.3,  # P2 in qualifying, P2 in race
    ... }, outcome1_name="qualifying", outcome2_name="race")
    >>> joint[(1, 1)]  # P(qualifying=1, race=1)
    0.4
    >>> marginal1 = joint.marginal1()  # Qualifying distribution
    >>> marginal1[1]  # P(qualifying=1)
    0.6
    """

    joint_probs: Dict[Tuple[Any, Any], float]
    outcome1_name: str = "var1"
    outcome2_name: str = "var2"
    _validate: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        """Validate probabilities if required."""
        if self._validate:
            if not self.joint_probs:
                raise DistributionError("Cannot create empty distribution")
            self.validate()

    def get(self, outcome_pair: Tuple[Any, Any]) -> float:
        """
        Get probability for a specific outcome pair.

        Parameters
        ----------
        outcome_pair : Tuple[Any, Any]
            The outcome pair to get probability for.

        Returns
        -------
        float
            Probability of the outcome pair.
        """
        return self.joint_probs.get(outcome_pair, 0.0)

    def items(self) -> Iterator[Tuple[Tuple[Any, Any], float]]:
        """
        Iterate over (outcome_pair, probability) pairs.

        Returns
        -------
        Iterator[Tuple[Tuple[Any, Any], float]]
            Iterator over outcome_pair-probability pairs.
        """
        return iter(self.joint_probs.items())

    def normalize(self) -> "JointDistribution":
        """
        Normalize probabilities to sum to 1.0.

        Returns
        -------
        JointDistribution
            Normalized distribution.

        Examples
        --------
        >>> dist = JointDistribution({(1, 1): 2.0, (1, 2): 2.0}, validate=False)
        >>> normalized = dist.normalize()
        >>> normalized[(1, 1)]
        0.5
        >>> normalized[(1, 2)]
        0.5
        """
        total = sum(self.joint_probs.values())
        if math.isclose(total, 0.0, abs_tol=TOLERANCE):
            raise DistributionError("Cannot normalize zero-sum distribution")

        normalized = {k: v / total for k, v in self.joint_probs.items()}
        return JointDistribution(
            normalized,
            outcome1_name=self.outcome1_name,
            outcome2_name=self.outcome2_name,
        )

    def marginal1(self) -> "Distribution":
        """
        Get marginal distribution for first variable.

        Returns
        -------
        Distribution
            Marginal distribution.

        Examples
        --------
        >>> joint = JointDistribution({
        ...     (1, 1): 0.4, (1, 2): 0.2,
        ...     (2, 1): 0.1, (2, 2): 0.3
        ... })
        >>> marginal = joint.marginal1()
        >>> marginal[1]  # P(var1=1)
        0.6
        >>> marginal[2]  # P(var1=2)
        0.4
        """
        marginal = {}
        for (val1, _), prob in self.items():
            marginal[val1] = marginal.get(val1, 0.0) + prob

        return PositionDistribution(marginal)

    def marginal2(self) -> "Distribution":
        """
        Get marginal distribution for second variable.

        Returns
        -------
        Distribution
            Marginal distribution.

        Examples
        --------
        >>> joint = JointDistribution({
        ...     (1, 1): 0.4, (1, 2): 0.2,
        ...     (2, 1): 0.1, (2, 2): 0.3
        ... })
        >>> marginal = joint.marginal2()
        >>> marginal[1]  # P(var2=1)
        0.5
        >>> marginal[2]  # P(var2=2)
        0.5
        """
        marginal = {}
        for (_, val2), prob in self.items():
            marginal[val2] = marginal.get(val2, 0.0) + prob

        return PositionDistribution(marginal)

    def conditional1(self, value2: Any) -> "Distribution":
        """
        Get conditional distribution P(var1 | var2=value2).

        Parameters
        ----------
        value2 : Any
            Value of second variable to condition on.

        Returns
        -------
        Distribution
            Conditional distribution.

        Raises
        ------
        DistributionError
            If P(var2=value2) = 0.

        Examples
        --------
        >>> joint = JointDistribution({
        ...     (1, 1): 0.4, (1, 2): 0.2,
        ...     (2, 1): 0.1, (2, 2): 0.3
        ... })
        >>> cond = joint.conditional1(1)  # P(var1 | var2=1)
        >>> cond[1]  # P(var1=1 | var2=1)
        0.8
        >>> cond[2]  # P(var1=2 | var2=1)
        0.2
        """
        # Sum P(var1, var2=value2) for all var1
        marginal_prob = sum(prob for (_, val2), prob in self.items() if val2 == value2)

        if math.isclose(marginal_prob, 0.0, abs_tol=TOLERANCE):
            raise DistributionError(f"P({self.outcome2_name}={value2}) = 0")

        # Calculate P(var1, var2=value2) / P(var2=value2)
        conditional = {}
        for (val1, val2), prob in self.items():
            if val2 == value2:
                conditional[val1] = prob / marginal_prob

        return PositionDistribution(conditional)

    def conditional2(self, value1: Any) -> "Distribution":
        """
        Get conditional distribution P(var2 | var1=value1).

        Parameters
        ----------
        value1 : Any
            Value of first variable to condition on.

        Returns
        -------
        Distribution
            Conditional distribution.

        Raises
        ------
        DistributionError
            If P(var1=value1) = 0.

        Examples
        --------
        >>> joint = JointDistribution({
        ...     (1, 1): 0.4, (1, 2): 0.2,
        ...     (2, 1): 0.1, (2, 2): 0.3
        ... })
        >>> cond = joint.conditional2(1)  # P(var2 | var1=1)
        >>> cond[1]  # P(var2=1 | var1=1)
        0.6666666666666666
        >>> cond[2]  # P(var2=2 | var1=1)
        0.3333333333333333
        """
        # Sum P(var1=value1, var2) for all var2
        marginal_prob = sum(prob for (val1, _), prob in self.items() if val1 == value1)

        if math.isclose(marginal_prob, 0.0, abs_tol=TOLERANCE):
            raise DistributionError(f"P({self.outcome1_name}={value1}) = 0")

        # Calculate P(var1=value1, var2) / P(var1=value1)
        conditional = {}
        for (val1, val2), prob in self.items():
            if val1 == value1:
                conditional[val2] = prob / marginal_prob

        return PositionDistribution(conditional)

    def get_correlation(self) -> float:
        """
        Calculate correlation coefficient between variables.

        Returns
        -------
        float
            Correlation coefficient between -1 and 1.

        Examples
        --------
        >>> # Perfect positive correlation
        >>> perfect_pos = JointDistribution({(1, 1): 0.5, (2, 2): 0.5})
        >>> perfect_pos.get_correlation()
        1.0

        >>> # Perfect negative correlation
        >>> perfect_neg = JointDistribution({(1, 2): 0.5, (2, 1): 0.5})
        >>> perfect_neg.get_correlation()
        -1.0

        >>> # Independence (correlation = 0)
        >>> independent = JointDistribution({
        ...    (1, 1): 0.25, (1, 2): 0.25,
        ...    (2, 1): 0.25, (2, 2): 0.25
        ... })
        >>> independent.get_correlation()
        0.0
        """
        # Get marginal distributions
        marginal1 = self.marginal1()
        marginal2 = self.marginal2()

        # Calculate expected values
        e_x = sum(x * marginal1[x] for x in marginal1.outcomes)
        e_y = sum(y * marginal2[y] for y in marginal2.outcomes)
        e_xy = sum(x * y * self.get((x, y)) for x, y in self.joint_probs)

        # Calculate variances
        var_x = sum((x - e_x) ** 2 * marginal1[x] for x in marginal1.outcomes)
        var_y = sum((y - e_y) ** 2 * marginal2[y] for y in marginal2.outcomes)

        # Calculate correlation
        if math.isclose(var_x, 0.0) or math.isclose(var_y, 0.0):
            return 0.0  # No correlation if no variance

        return (e_xy - e_x * e_y) / math.sqrt(var_x * var_y)

    def mutual_information(self) -> float:
        """
        Calculate mutual information between variables.

        Returns
        -------
        float
            Mutual information in bits.

        Examples
        --------
        >>> # Independence (MI = 0)
        >>> independent = JointDistribution({
        ...    (1, 1): 0.25, (1, 2): 0.25,
        ...    (2, 1): 0.25, (2, 2): 0.25
        ... })
        >>> independent.mutual_information()
        0.0

        >>> # Perfect dependence
        >>> dependent = JointDistribution({(1, 1): 0.5, (2, 2): 0.5})
        >>> dependent.mutual_information()
        1.0
        """
        # Get marginal distributions
        marginal1 = self.marginal1()
        marginal2 = self.marginal2()

        # Calculate mutual information
        mi = 0.0
        for (x, y), p_xy in self.items():
            if p_xy > 0:
                p_x = marginal1[x]
                p_y = marginal2[y]
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

        return mi

    def is_independent(self, tolerance: float = TOLERANCE) -> bool:
        """
        Check if variables are independent.

        Parameters
        ----------
        tolerance : float, optional
            Tolerance for independence check, by default TOLERANCE.

        Returns
        -------
        bool
            True if variables are independent, False otherwise.

        Examples
        --------
        >>> # Independent joint distribution
        >>> independent = JointDistribution({
        ...    (1, 1): 0.25, (1, 2): 0.25,
        ...    (2, 1): 0.25, (2, 2): 0.25
        ... })
        >>> independent.is_independent()
        True

        >>> # Dependent joint distribution
        >>> dependent = JointDistribution({(1, 1): 0.5, (2, 2): 0.5})
        >>> dependent.is_independent()
        False
        """
        # Get marginal distributions
        marginal1 = self.marginal1()
        marginal2 = self.marginal2()

        # Check if P(x,y) = P(x)P(y) for all x,y
        for (x, y), p_xy in self.items():
            p_x = marginal1[x]
            p_y = marginal2[y]
            expected = p_x * p_y
            if abs(p_xy - expected) > tolerance:
                return False

        return True

    def to_dict(self) -> Dict[Tuple[Any, Any], float]:
        """
        Convert to dictionary.

        Returns
        -------
        Dict[Tuple[Any, Any], float]
            Dictionary mapping from outcome pairs to probabilities.
        """
        return dict(self.joint_probs)


def create_independent_joint(
    dist1: Distribution, dist2: Distribution, name1: str = "var1", name2: str = "var2"
) -> JointDistribution:
    """
    Create joint distribution assuming independence.

    Parameters
    ----------
    dist1 : Distribution
        First marginal distribution.
    dist2 : Distribution
        Second marginal distribution.
    name1 : str, optional
        Name of first variable, by default "var1".
    name2 : str, optional
        Name of second variable, by default "var2".

    Returns
    -------
    JointDistribution
        Joint distribution with P(x,y) = P(x)P(y).

    Examples
    --------
    >>> dist1 = PositionDistribution({1: 0.6, 2: 0.4})
    >>> dist2 = PositionDistribution({1: 0.3, 2: 0.7})
    >>> joint = create_independent_joint(dist1, dist2)
    >>> joint[(1, 1)]  # P(1,1) = P(1)P(1) = 0.6*0.3
    0.18
    >>> joint[(1, 2)]  # P(1,2) = P(1)P(2) = 0.6*0.7
    0.42
    >>> joint.is_independent()
    True
    """
    joint_probs = {}
    for x, px in dist1.items():
        for y, py in dist2.items():
            joint_probs[(x, y)] = px * py

    return JointDistribution(joint_probs, outcome1_name=name1, outcome2_name=name2)


def create_constrained_joint(
    dist1: PositionDistribution,
    dist2: PositionDistribution,
    name1: str = "var1",
    name2: str = "var2",
) -> JointDistribution:
    """
    Create joint distribution with constraint that outcomes cannot be equal.

    This is useful for race positions where two drivers cannot finish
    in the same position.

    Parameters
    ----------
    dist1 : PositionDistribution
        First marginal position distribution.
    dist2 : PositionDistribution
        Second marginal position distribution.
    name1 : str, optional
        Name of first variable, by default "var1".
    name2 : str, optional
        Name of second variable, by default "var2".

    Returns
    -------
    JointDistribution
        Constrained joint distribution.

    Examples
    --------
    >>> dist1 = PositionDistribution({1: 0.6, 2: 0.4})
    >>> dist2 = PositionDistribution({1: 0.3, 2: 0.7})
    >>> joint = create_constrained_joint(dist1, dist2)
    >>> joint[(1, 1)]  # Cannot both finish P1
    0.0
    >>> joint[(1, 2)]  # More likely than independent case
    0.6666666666666666
    >>> joint.is_independent()
    False
    """
    # Start with product distribution but exclude equal outcomes
    joint_probs = {}
    for x, px in dist1.items():
        for y, py in dist2.items():
            if x != y:  # Constraint: cannot have same position
                joint_probs[(x, y)] = px * py

    # Normalize to sum to 1.0
    total = sum(joint_probs.values())
    if math.isclose(total, 0.0, abs_tol=TOLERANCE):
        raise DistributionError(
            "Cannot create constrained joint distribution "
            "with non-overlapping outcomes"
        )

    normalized = {k: v / total for k, v in joint_probs.items()}
    return JointDistribution(normalized, outcome1_name=name1, outcome2_name=name2)


def create_conditional_joint(
    marginal: Distribution, conditional_func, name1: str = "var1", name2: str = "var2"
) -> JointDistribution:
    """
    Create joint distribution from marginal and conditional distributions.

    Parameters
    ----------
    marginal : Distribution
        Marginal distribution for first variable.
    conditional_func : Callable[[Any], Distribution]
        Function that takes a value of the first variable and returns
        the conditional distribution for the second variable.
    name1 : str, optional
        Name of first variable, by default "var1".
    name2 : str, optional
        Name of second variable, by default "var2".

    Returns
    -------
    JointDistribution
        Joint distribution with P(x,y) = P(x)P(y|x).

    Examples
    --------
    >>> # Define marginal distribution
    >>> marginal = PositionDistribution({1: 0.6, 2: 0.4})
    >>>
    >>> # Define conditional distributions
    >>> def get_conditional(x):
    ...     if x == 1:
    ...         return PositionDistribution({1: 0.1, 2: 0.9})
    ...     else:  # x == 2
    ...         return PositionDistribution({1: 0.8, 2: 0.2})
    ...
    >>> joint = create_conditional_joint(marginal, get_conditional)
    >>> joint[(1, 1)]  # P(1,1) = P(1)P(1|1) = 0.6*0.1
    0.06
    >>> joint[(1, 2)]  # P(1,2) = P(1)P(2|1) = 0.6*0.9
    0.54
    """
    joint_probs = {}
    for x, px in marginal.items():
        conditional = conditional_func(x)
        for y, py_given_x in conditional.items():
            joint_probs[(x, y)] = px * py_given_x

    return JointDistribution(joint_probs, outcome1_name=name1, outcome2_name=name2)


def marginalize_joint(joint: JointDistribution, dims: Tuple[Any, ...]) -> Distribution:
    """
    Marginalize joint distribution to specified dimensions.

    Parameters
    ----------
    joint : JointDistribution
        Joint distribution to marginalize.
    dims : Tuple[Any, ...]
        Dimensions to keep.

    Returns
    -------
    Distribution
        Marginalized distribution.

    Examples
    --------
    >>> joint = JointDistribution({
    ...     (1, 1, 1): 0.2, (1, 1, 2): 0.2,
    ...     (1, 2, 1): 0.1, (1, 2, 2): 0.1,
    ...     (2, 1, 1): 0.1, (2, 1, 2): 0.1,
    ...     (2, 2, 1): 0.1, (2, 2, 2): 0.1
    ... })
    >>> # Marginalize to first dimension
    >>> marg = marginalize_joint(joint, (0,))
    >>> marg[1]  # Sum of all joint probs where first dim is 1
    0.6
    >>> marg[2]  # Sum of all joint probs where first dim is 2
    0.4
    """
    if len(dims) == 1 and dims[0] == 0:
        return joint.marginal1()
    elif len(dims) == 1 and dims[0] == 1:
        return joint.marginal2()
    else:
        raise NotImplementedError(
            "Currently only supports marginalizing to first or second dimension"
        )
