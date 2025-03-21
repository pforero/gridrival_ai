"""
Position distribution class for representing probabilities of finishing positions.

This module provides the PositionDistribution class for modeling probability
distributions in racing contexts. It represents the probabilities of a single
driver/entity finishing in different positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, Tuple

# Constants
TOLERANCE = 1e-6


class DistributionError(Exception):
    """Exception raised for distribution validation errors."""

    pass


@dataclass(frozen=True)
class PositionDistribution:
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
            self.validate()

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

    @property
    def outcomes(self) -> Iterator[int]:
        """
        Iterate over all positions with non-zero probability.

        Returns
        -------
        Iterator[int]
            Iterator over positions.
        """
        return (position for position, _ in self.items())

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

        Checks:
        - Probabilities sum to 1.0
        - All probabilities are between 0.0 and 1.0
        - All positions are positive integers
        - All consecutive positions from 1 to the maximum position are present

        Raises
        ------
        DistributionError
            If distribution is invalid.
        """
        # Get all positions and probabilities
        positions = sorted(self.position_probs.keys())
        probs_list = list(self.probabilities)

        # Check if there are any positions and probabilities
        if not positions or not probs_list:
            raise DistributionError("Distribution cannot be empty")

        # Check for valid position values
        if any(not isinstance(p, int) or p <= 0 for p in positions):
            invalid_pos = [p for p in positions if not isinstance(p, int) or p <= 0]
            raise DistributionError(
                f"Invalid positions found: {invalid_pos} (must be positive integers)"
            )

        # Check that all positions from 1 to max are present
        min_pos = min(positions)
        max_pos = max(positions)

        if min_pos != 1:
            raise DistributionError(
                f"Distribution must start at position 1, found minimum position "
                f"{min_pos}"
            )

        expected_positions = set(range(1, max_pos + 1))
        actual_positions = set(positions)

        if expected_positions != actual_positions:
            missing = expected_positions - actual_positions
            raise DistributionError(
                f"Missing consecutive positions: {sorted(missing)}. "
                f"All positions from 1 to {max_pos} must have probabilities."
            )

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

    def to_dict(self) -> Dict[int, float]:
        """
        Convert to dictionary.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping from positions to probabilities.
        """
        return dict(self.position_probs)

    def expected_value(self, value_dict: Dict[int, float]) -> float:
        """
        Calculate expected value based on provided values for each position.

        Parameters
        ----------
        value_dict : Dict[int, float]
            Dictionary mapping positions to values (e.g., points)

        Returns
        -------
        float
            Expected value

        Examples
        --------
        >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> points = {1: 25, 2: 18}
        >>> dist.expected_value(points)
        22.2  # 0.6*25 + 0.4*18
        """
        expected = 0.0
        for position, probability in self.items():
            value = value_dict.get(position, 0.0)
            expected += probability * value
        return expected

    def __getitem__(self, position: int) -> float:
        """
        Get probability for a position using dictionary-like access.

        Parameters
        ----------
        position : int
            Racing position (1-based).

        Returns
        -------
        float
            Probability of the position.
        """
        return self.get(position)
