"""
Driver position probability distributions.

This module provides the DriverDistribution class for managing position probabilities
across different sessions for a single driver.
"""

from dataclasses import dataclass

from gridrival_ai.probabilities.types import (
    DistributionError,
    JointProbabilities,
    SessionProbabilities,
)


@dataclass(frozen=True)
class DriverDistribution:
    """Position probability distributions for a single driver.

    Parameters
    ----------
    race : SessionProbabilities
        Race position probabilities
    qualifying : SessionProbabilities | None, optional
        Qualifying position probabilities. If None, race probabilities are used.
    sprint : SessionProbabilities | None, optional
        Sprint position probabilities. If None, race probabilities are used.
    joint_qual_race : JointProbabilities | None, optional
        Joint qualifying/race position probabilities. If None, independence is assumed.
    completion_prob : float, optional
        Probability of completing each stage of the race, by default 0.95.
        A completion_prob of 0.95 means:
        - 95% chance of reaching 25% distance
        - 90% chance of reaching 50% distance
        - 86% chance of reaching 75% distance
        - 81% chance of reaching 90% distance

    Notes
    -----
    Race probabilities are required and used as defaults for other sessions.
    If joint probabilities are not provided, independence is assumed.
    Completion probability must be between 0 and 1.

    Examples
    --------
    >>> # Basic initialization with just race probabilities
    >>> race_probs = SessionProbabilities({1: 0.6, 2: 0.4})
    >>> dist = DriverDistribution(race=race_probs)
    >>> dist.qualifying == dist.race  # qualifying uses race probs
    True

    >>> # Full initialization
    >>> dist = DriverDistribution(
    ...     race=SessionProbabilities({1: 0.5, 2: 0.5}),
    ...     qualifying=SessionProbabilities({1: 0.6, 2: 0.4}),
    ...     sprint=SessionProbabilities({1: 0.7, 2: 0.3}),
    ...     joint_qual_race=JointProbabilities(
    ...         session1="qualifying",
    ...         session2="race",
    ...         probabilities={(1, 1): 0.4, (1, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
    ...     ),
    ...     completion_prob=0.90  # More conservative completion probability
    ... )
    """

    race: SessionProbabilities
    qualifying: SessionProbabilities | None = None
    sprint: SessionProbabilities | None = None
    joint_qual_race: JointProbabilities | None = None
    completion_prob: float = 0.95

    def __post_init__(self) -> None:
        """Initialize with defaults if needed and validate completion probability."""
        # Validate completion probability
        if not 0 <= self.completion_prob <= 1:
            raise DistributionError(
                "completion_prob must be between 0 and 1"
                f" (got {self.completion_prob})"
            )

        # Set qualifying and sprint to race probabilities if not provided
        object.__setattr__(
            self,
            "qualifying",
            self.qualifying if self.qualifying is not None else self.race,
        )
        object.__setattr__(
            self, "sprint", self.sprint if self.sprint is not None else self.race
        )

        # Create independent joint distribution if not provided
        if self.joint_qual_race is None:
            joint_probs = {
                (q, r): self.qualifying.probabilities[q] * self.race.probabilities[r]
                for q in self.qualifying.probabilities
                for r in self.race.probabilities
            }
            object.__setattr__(
                self,
                "joint_qual_race",
                JointProbabilities(
                    session1="qualifying", session2="race", probabilities=joint_probs
                ),
            )
