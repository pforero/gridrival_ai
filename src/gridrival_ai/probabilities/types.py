"""
Probability distribution types for GridRival AI.

This module defines the core probability distribution types used throughout
the GridRival AI package. It provides dataclasses for representing single-session
and joint probability distributions for driver positions.

Classes
-------
SessionProbabilities
    Single session probability distribution for positions 1-20
JointProbabilities
    Joint probability distribution between two sessions

Examples
--------
>>> # Create a simple session distribution
>>> probs = {1: 0.6, 2: 0.4}
>>> session_dist = SessionProbabilities(probabilities=probs)
>>> session_dist[1]  # Returns 0.6
>>> session_dist[2]  # Returns 0.4
>>> for pos, prob in session_dist.items():
...     print(f"P{pos}: {prob:.1f}")
P1: 0.6
P2: 0.4

>>> # Create a joint distribution
>>> joint_probs = {(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
>>> joint_dist = JointProbabilities(
...     session1="qualifying",
...     session2="race",
...     probabilities=joint_probs
... )
>>> joint_dist[(1, 1)]  # Returns 0.4
>>> for (pos1, pos2), prob in joint_dist.items():
...     print(f"{session1}P{pos1}-{session2}P{pos2}: {prob:.1f}")
qualifyingP1-raceP1: 0.4
qualifyingP1-raceP2: 0.2
qualifyingP2-raceP1: 0.1
qualifyingP2-raceP2: 0.3
"""

from dataclasses import dataclass
from typing import ItemsView

import numpy as np

from gridrival_ai.scoring.constants import MAX_POSITION


class DistributionError(Exception):
    """Raised when probability distributions are invalid."""

    pass


@dataclass(frozen=True)
class SessionProbabilities:
    """Probability distribution for positions in a single session.

    Parameters
    ----------
    probabilities : Dict[int, float]
        Mapping of positions (1-20) to probabilities

    Notes
    -----
    Probabilities must sum to 1.0 and be between 0 and 1.
    Positions must be between 1 and MAX_POSITION.

    Examples
    --------
    >>> probs = SessionProbabilities({1: 0.6, 2: 0.4})
    >>> probs[1]  # Returns 0.6
    >>> probs[2]  # Returns 0.4
    >>> for pos, prob in probs.items():
    ...     print(f"P{pos}: {prob:.1f}")
    P1: 0.6
    P2: 0.4
    """

    probabilities: dict

    def __post_init__(self) -> None:
        """Validate the probability distribution."""
        # Check positions are valid
        invalid_pos = [p for p in self.probabilities if not 1 <= p <= MAX_POSITION]
        if invalid_pos:
            raise DistributionError(f"Invalid positions in distribution: {invalid_pos}")

        # Check probabilities are valid
        if any(p < 0 or p > 1 for p in self.probabilities.values()):
            raise DistributionError("Invalid probabilities (must be between 0 and 1)")

        # Check sum is close to 1
        total = sum(self.probabilities.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise DistributionError(f"Probabilities sum to {total} (must sum to 1.0)")

    def __getitem__(self, position: int) -> float:
        """Get probability for a position.

        Parameters
        ----------
        position : int
            Position to get probability for (1-20)

        Returns
        -------
        float
            Probability of finishing in that position

        Raises
        ------
        KeyError
            If position not in distribution
        """
        return self.probabilities[position]

    def items(self) -> ItemsView[int, float]:
        """Get items view of position-probability pairs.

        Returns
        -------
        ItemsView[int, float]
            View of (position, probability) pairs

        Examples
        --------
        >>> probs = SessionProbabilities({1: 0.6, 2: 0.4})
        >>> for pos, prob in probs.items():
        ...     print(f"P{pos}: {prob:.1f}")
        P1: 0.6
        P2: 0.4
        """
        return self.probabilities.items()


@dataclass(frozen=True)
class JointProbabilities:
    """Joint probability distribution between two sessions.

    Parameters
    ----------
    session1 : str
        Name of first session
    session2 : str
        Name of second session
    probabilities : Dict[Tuple[int, int], float]
        Mapping of position pairs to probabilities

    Notes
    -----
    Probabilities must sum to 1.0 and marginals must match individual
    session distributions.

    Examples
    --------
    >>> joint_probs = {(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
    >>> dist = JointProbabilities("qualifying", "race", joint_probs)
    >>> dist[(1, 1)]  # Returns 0.4
    >>> for (pos1, pos2), prob in dist.items():
    ...     print(f"{dist.session1}P{pos1}-{dist.session2}P{pos2}: {prob:.1f}")
    qualifyingP1-raceP1: 0.4
    qualifyingP1-raceP2: 0.2
    qualifyingP2-raceP1: 0.1
    qualifyingP2-raceP2: 0.3
    """

    session1: str
    session2: str
    probabilities: dict

    def __post_init__(self) -> None:
        """Validate the joint distribution."""
        # Check positions are valid
        invalid = [
            pos_pair
            for pos_pair in self.probabilities.keys()
            if not (
                1 <= pos_pair[0] <= MAX_POSITION and 1 <= pos_pair[1] <= MAX_POSITION
            )
        ]
        if invalid:
            raise DistributionError(
                f"Invalid positions in joint distribution: {invalid}"
            )

        # Check probabilities are valid
        if any(p < 0 or p > 1 for p in self.probabilities.values()):
            raise DistributionError(
                "Invalid joint probabilities (must be between 0 and 1)"
            )

        # Check sum is close to 1
        total = sum(self.probabilities.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise DistributionError(
                f"Joint probabilities sum to {total} (must sum to 1.0)"
            )

    def __getitem__(self, positions: tuple) -> float:
        """Get joint probability for a pair of positions.

        Parameters
        ----------
        positions : Tuple[int, int]
            Position pair to get probability for (pos1, pos2)

        Returns
        -------
        float
            Joint probability of finishing in those positions

        Raises
        ------
        KeyError
            If position pair not in distribution
        """
        return self.probabilities[positions]

    def items(self) -> ItemsView[tuple, float]:
        """Get items view of position pairs and their probabilities.

        Returns
        -------
        ItemsView[Tuple[int, int], float]
            View of ((position1, position2), probability) pairs

        Examples
        --------
        >>> joint_probs = {(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
        >>> dist = JointProbabilities("qualifying", "race", joint_probs)
        >>> for (pos1, pos2), prob in dist.items():
        ...     print(f"{dist.session1}P{pos1}-{dist.session2}P{pos2}: {prob:.1f}")
        qualifyingP1-raceP1: 0.4
        qualifyingP1-raceP2: 0.2
        qualifyingP2-raceP1: 0.1
        qualifyingP2-raceP2: 0.3
        """
        return self.probabilities.items()
