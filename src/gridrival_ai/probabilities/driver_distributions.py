"""
Driver position probability distributions.

This module provides the DriverDistribution class for managing position probabilities
across different sessions (qualifying, race, sprint) for a single driver.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from gridrival_ai.scoring.constants import MAX_POSITION


class DistributionError(Exception):
    """Raised when probability distributions are invalid."""

    pass


@dataclass(frozen=True)
class DriverDistribution:
    """Position probability distributions for a single driver.

    Parameters
    ----------
    race : Dict[int, float]
        Race position probabilities
    qualifying : Dict[int, float] | None, optional
        Qualifying position probabilities. If None, race probabilities are used.
    sprint : Dict[int, float] | None, optional
        Sprint position probabilities. If None, race probabilities are used.
    joint_qual_race : Dict[Tuple[int, int], float] | None, optional
        Joint qualifying/race position probabilities. If None, independence is assumed.

    Notes
    -----
    All probabilities for each session must sum to 1.0.
    Joint probabilities must be consistent with marginal distributions.
    Positions must be between 1 and MAX_POSITION (20).

    Examples
    --------
    >>> # Basic initialization with just race probabilities
    >>> dist = DriverDistribution(race={1: 0.6, 2: 0.4})
    >>> dist.qualifying == dist.race  # qualifying uses race probs
    True

    >>> # Full initialization with all distributions
    >>> dist = DriverDistribution(
    ...     race={1: 0.5, 2: 0.5},
    ...     qualifying={1: 0.6, 2: 0.4},
    ...     sprint={1: 0.7, 2: 0.3},
    ...     joint_qual_race={(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
    ... )
    """

    race: Dict[int, float]
    qualifying: Dict[int, float] | None = None
    sprint: Dict[int, float] | None = None
    joint_qual_race: Dict[Tuple[int, int], float] | None = None

    def __post_init__(self) -> None:
        """Initialize and validate distributions."""
        # Validate race probabilities (required)
        self._validate_distribution(self.race, "race")

        # Create object.__setattr__ references for frozen dataclass
        object.__setattr__(
            self,
            "qualifying",
            self.qualifying if self.qualifying is not None else self.race.copy(),
        )
        object.__setattr__(
            self, "sprint", self.sprint if self.sprint is not None else self.race.copy()
        )

        # Validate other distributions
        self._validate_distribution(self.qualifying, "qualifying")
        self._validate_distribution(self.sprint, "sprint")

        # Create independent joint distribution if not provided
        if self.joint_qual_race is None:
            joint = {
                (q, r): self.qualifying[q] * self.race[r]
                for q in self.qualifying
                for r in self.race
            }
            object.__setattr__(self, "joint_qual_race", joint)
        else:
            self._validate_joint_distribution()

    def _validate_distribution(self, probs: Dict[int, float], name: str) -> None:
        """Validate a single session distribution.

        Parameters
        ----------
        probs : Dict[int, float]
            Position probabilities to validate
        name : str
            Distribution name for error messages

        Raises
        ------
        DistributionError
            If probabilities are invalid
        """
        # Check positions are valid
        invalid_pos = [p for p in probs if not 1 <= p <= MAX_POSITION]
        if invalid_pos:
            raise DistributionError(
                f"Invalid positions in {name} distribution: {invalid_pos}"
            )

        # Check probabilities are valid
        if any(p < 0 or p > 1 for p in probs.values()):
            raise DistributionError(
                f"Invalid probabilities in {name} distribution"
                " (must be between 0 and 1)"
            )

        # Check sum is close to 1
        total = sum(probs.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise DistributionError(
                f"Probabilities in {name} distribution sum to {total}"
                " (must sum to 1.0)"
            )

    def _validate_joint_distribution(self) -> None:
        """Validate joint qualifying/race distribution.

        Checks that joint probabilities sum to 1.0 and are consistent
        with marginal distributions.

        Raises
        ------
        DistributionError
            If joint distribution is invalid
        """
        # Check joint probabilities are valid
        if any(p < 0 or p > 1 for p in self.joint_qual_race.values()):
            raise DistributionError(
                "Invalid joint probabilities (must be between 0 and 1)"
            )

        # Check positions are valid
        invalid = [
            p
            for p in self.joint_qual_race.keys()
            if not (1 <= p[0] <= MAX_POSITION and 1 <= p[1] <= MAX_POSITION)
        ]
        if invalid:
            raise DistributionError(
                f"Invalid positions in joint distribution: {invalid}"
            )

        # Check sum is close to 1
        total = sum(self.joint_qual_race.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise DistributionError(
                f"Joint probabilities sum to {total} (must sum to 1.0)"
            )

        # Check qualifying marginals match
        qual_marginals = {}
        for (q, _), p in self.joint_qual_race.items():
            qual_marginals[q] = qual_marginals.get(q, 0.0) + p

        for pos in self.qualifying:
            if not np.isclose(
                qual_marginals.get(pos, 0.0), self.qualifying[pos], rtol=1e-5
            ):
                raise DistributionError(
                    "Joint qualifying marginals don't match" " qualifying distribution"
                )

        # Check race marginals match
        race_marginals = {}
        for (_, r), p in self.joint_qual_race.items():
            race_marginals[r] = race_marginals.get(r, 0.0) + p

        for pos in self.race:
            if not np.isclose(race_marginals.get(pos, 0.0), self.race[pos], rtol=1e-5):
                raise DistributionError(
                    "Joint race marginals don't match race distribution"
                )
