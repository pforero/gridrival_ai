"""
Joint distribution class for representing joint probabilities between two entities.

This module provides the JointDistribution class for modeling joint probability
distributions between two entities, such as between two drivers or a driver's
positions in two different sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple

from gridrival_ai.probabilities.distributions.position import (
    DistributionError,
    PositionDistribution,
)

# Constants
TOLERANCE = 1e-6


@dataclass(frozen=True)
class JointDistribution:
    """
    Joint probability distribution between two variables.

    This class represents a joint probability distribution between
    two variables, such as qualifying and race positions, or positions
    of two different drivers.

    Parameters
    ----------
    joint_probs : Dict[Tuple[Any, Any], float]
        Mapping from outcome pairs to probabilities.
    entity1_name : str, optional
        Name of first entity, by default "var1".
    entity2_name : str, optional
        Name of second entity, by default "var2".
    validate : bool, optional
        Whether to validate probabilities, by default True.

    Raises
    ------
    DistributionError
        If validation fails and validate=True.

    Examples
    --------
    >>> # Create a joint distribution for two drivers
    >>> joint = JointDistribution({
    ...     (1, 1): 0.0,  # VER P1, HAM P1 (impossible in race)
    ...     (1, 2): 0.4,  # VER P1, HAM P2
    ...     (2, 1): 0.3,  # VER P2, HAM P1
    ...     (2, 2): 0.0,  # VER P2, HAM P2 (impossible in race)
    ...     (3, 3): 0.3,  # VER P3, HAM P3 (impossible in race)
    ... }, entity1_name="VER", entity2_name="HAM")
    >>> joint[(1, 2)]  # P(VER=1, HAM=2)
    0.4
    """

    joint_probs: Dict[Tuple[Any, Any], float]
    entity1_name: str = "var1"
    entity2_name: str = "var2"
    _validate: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        """Validate probabilities if required."""
        if self._validate:
            if not self.joint_probs:
                raise DistributionError("Cannot create empty distribution")
            self.validate()

    def validate(self) -> None:
        """
        Validate joint distribution.

        Checks:
        - Probabilities sum to 1.0
        - All probabilities are between 0.0 and 1.0

        Raises
        ------
        DistributionError
            If validation fails
        """
        # Get all probabilities as a list
        probs_list = list(self.probabilities)

        # If the distribution is empty, it's invalid
        if not probs_list:
            raise DistributionError("Distribution cannot be empty")

        # Check probabilities sum to 1.0
        total = sum(probs_list)
        if not math.isclose(total, 1.0, abs_tol=TOLERANCE):
            raise DistributionError(f"Probabilities must sum to 1.0 (got {total:.6f})")

        # Check all probabilities are valid
        invalid_probs = [p for p in probs_list if p < 0.0 or p > 1.0]
        if invalid_probs:
            raise DistributionError(
                f"Invalid probabilities found: {invalid_probs[:5]}..."
                if len(invalid_probs) > 5
                else f"Invalid probabilities found: {invalid_probs}"
            )

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

    @property
    def outcomes(self) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over all outcome pairs with non-zero probability.

        Returns
        -------
        Iterator[Tuple[Any, Any]]
            Iterator over outcome pairs.
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
            entity1_name=self.entity1_name,
            entity2_name=self.entity2_name,
        )

    def marginal1(self) -> PositionDistribution:
        """
        Get marginal distribution for first entity.

        Returns
        -------
        PositionDistribution
            Marginal distribution.

        Examples
        --------
        >>> joint = JointDistribution({
        ...     (1, 1): 0.0,
        ...     (1, 2): 0.4,
        ...     (2, 1): 0.3,
        ...     (2, 2): 0.0,
        ...     (3, 3): 0.3,
        ... })
        >>> marginal = joint.marginal1()
        >>> marginal[1]  # P(entity1=1)
        0.4
        >>> marginal[2]  # P(entity1=2)
        0.3
        """
        marginal = {}
        for (val1, _), prob in self.items():
            if isinstance(val1, int):  # Ensure position is an integer
                marginal[val1] = marginal.get(val1, 0.0) + prob

        return PositionDistribution(marginal)

    def marginal2(self) -> PositionDistribution:
        """
        Get marginal distribution for second entity.

        Returns
        -------
        PositionDistribution
            Marginal distribution.

        Examples
        --------
        >>> joint = JointDistribution({
        ...     (1, 1): 0.0,
        ...     (1, 2): 0.4,
        ...     (2, 1): 0.3,
        ...     (2, 2): 0.0,
        ...     (3, 3): 0.3,
        ... })
        >>> marginal = joint.marginal2()
        >>> marginal[1]  # P(entity2=1)
        0.3
        >>> marginal[2]  # P(entity2=2)
        0.4
        """
        marginal = {}
        for (_, val2), prob in self.items():
            if isinstance(val2, int):  # Ensure position is an integer
                marginal[val2] = marginal.get(val2, 0.0) + prob

        return PositionDistribution(marginal)

    def __getitem__(self, outcome_pair: Tuple[Any, Any]) -> float:
        """
        Get probability for an outcome pair using dictionary-like access.

        Parameters
        ----------
        outcome_pair : Tuple[Any, Any]
            The outcome pair to get probability for.

        Returns
        -------
        float
            Probability of the outcome pair.
        """
        return self.get(outcome_pair)

    @classmethod
    def create_from_distributions(
        cls,
        dist1: PositionDistribution,
        dist2: PositionDistribution,
        entity1_name: str = "var1",
        entity2_name: str = "var2",
        constrained: bool = False,
    ) -> "JointDistribution":
        """
        Create a joint distribution from two position distributions.

        Parameters
        ----------
        dist1 : PositionDistribution
            First entity distribution
        dist2 : PositionDistribution
            Second entity distribution
        entity1_name : str, optional
            Name of first entity, by default "var1"
        entity2_name : str, optional
            Name of second entity, by default "var2"
        constrained : bool, optional
            Whether entities cannot have the same outcome, by default False.
            This is typically True for race positions where two drivers
            cannot finish in the same position.

        Returns
        -------
        JointDistribution
            Joint probability distribution
        """
        # Start with product distribution
        joint_probs = {}

        for pos1, prob1 in dist1.items():
            for pos2, prob2 in dist2.items():
                # Skip if constrained and positions are the same
                if constrained and pos1 == pos2:
                    continue

                joint_probs[(pos1, pos2)] = prob1 * prob2

        # Normalize to ensure sum to 1.0
        total = sum(joint_probs.values())
        if math.isclose(total, 0.0, abs_tol=TOLERANCE):
            raise DistributionError(
                "Cannot create joint distribution with zero probability"
            )

        normalized = {k: v / total for k, v in joint_probs.items()}

        return cls(normalized, entity1_name=entity1_name, entity2_name=entity2_name)
