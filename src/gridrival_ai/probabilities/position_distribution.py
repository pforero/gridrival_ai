"""
Position probability distributions container for F1 fantasy optimization.

This module provides a high-level interface for accessing driver position
probabilities across different sessions.
"""

from typing import Dict, Set, Tuple

from gridrival_ai.probabilities.driver_distributions import DriverDistribution
from gridrival_ai.probabilities.types import JointProbabilities, SessionProbabilities


class PositionDistributions:
    """Container for F1 driver position probability distributions.

    This class provides a high-level interface for accessing driver position
    probabilities across different sessions. It acts as a manager for individual
    driver distributions, providing a session-focused query interface.

    Parameters
    ----------
    driver_distributions : Dict[int, DriverDistribution]
        Mapping of driver IDs to their probability distributions

    Notes
    -----
    Each session's probabilities are managed by the underlying DriverDistribution.
    Joint probabilities default to independence if not explicitly provided.

    Examples
    --------
    >>> # Create from existing driver distributions
    >>> driver_dists = {1: DriverDistribution(race=race_probs)}
    >>> dist = PositionDistributions(driver_dists)
    >>> p1_prob = dist.get_session_probabilities(1, "qualifying")[1]

    >>> # Create from raw probability data
    >>> dist = PositionDistributions.from_session_probabilities(
    ...     qualifying_probs, race_probs)
    """

    VALID_SESSIONS = {"qualifying", "race", "sprint"}

    def __init__(self, driver_distributions: Dict[int, DriverDistribution]):
        """Initialize with driver distributions."""
        self.driver_distributions = driver_distributions

    def get_session_probabilities(
        self, driver_id: str, session: str
    ) -> SessionProbabilities:
        """Get position probabilities for a driver in a session.

        Parameters
        ----------
        driver_id : str
            Driver three-letter abbreviation
        session : str
            Session name ("qualifying", "race", or "sprint")

        Returns
        -------
        SessionProbabilities
            Probability distribution object that can be accessed like a dictionary
            mapping positions to probabilities

        Raises
        ------
        KeyError
            If driver_id not found or session not available
        ValueError
            If session name is invalid

        Examples
        --------
        >>> probs = dist.get_session_probabilities("VER", "race")
        >>> p1_prob = probs[1]  # Get probability of P1
        """
        if session not in self.VALID_SESSIONS:
            raise ValueError(f"Invalid session name: {session}")

        return getattr(self.driver_distributions[driver_id], session)

    def get_joint_probabilities(
        self, driver_id: str, session1: str, session2: str
    ) -> JointProbabilities:
        """Get joint position probabilities for a driver across two sessions.

        Parameters
        ----------
        driver_id : str
            Driver three-letter abbreviation
        session1 : str
            First session name
        session2 : str
            Second session name

        Returns
        -------
        JointProbabilities
            Probability distribution object that can be accessed like a dictionary
            mapping position pairs to probabilities

        Raises
        ------
        KeyError
            If driver_id not found or sessions not available
        ValueError
            If session combination is not supported

        Examples
        --------
        >>> joint_probs = dist.get_joint_probabilities("VER", "qualifying", "race")
        >>> p1_p1_prob = joint_probs[(1, 1)]  # Get probability of P1 in both sessions
        """
        if session1 == "qualifying" and session2 == "race":
            return self.driver_distributions[driver_id].joint_qual_race
        raise ValueError(
            f"Joint probabilities not available for {session1} and {session2}"
        )

    def get_completion_probability(self, driver_id: str) -> float:
        """Get completion probability for a driver.

        Parameters
        ----------
        driver_id : str
            Driver three-letter abbreviation

        Returns
        -------
        float
            Probability of completing each stage of the race

        Raises
        ------
        KeyError
            If driver_id not found
        """
        return self.driver_distributions[driver_id].completion_prob

    def get_available_sessions(self) -> Set[str]:
        """Get set of available session names.

        Returns
        -------
        Set[str]
            Names of available sessions
        """
        # All drivers have same sessions due to DriverDistribution defaults
        return {"qualifying", "race", "sprint"}

    @classmethod
    def from_session_probabilities(
        cls,
        qualifying_probs: Dict[int, Dict[int, float]],
        race_probs: Dict[int, Dict[int, float]],
        sprint_probs: Dict[int, Dict[int, float]] | None = None,
        joint_qual_race: Dict[int, Dict[Tuple[int, int], float]] | None = None,
    ) -> "PositionDistributions":
        """Create instance from raw probability dictionaries.

        Parameters
        ----------
        qualifying_probs : Dict[int, Dict[int, float]]
            Qualifying probabilities (driver_id -> position -> probability)
        race_probs : Dict[int, Dict[int, float]]
            Race probabilities
        sprint_probs : Dict[int, Dict[int, float]] | None, optional
            Sprint probabilities
        joint_qual_race : Dict[int, Dict[Tuple[int, int], float]] | None
            Joint qualifying-race probabilities

        Returns
        -------
        PositionDistributions
            Created instance
        """
        # Convert raw probabilities to SessionProbabilities
        driver_distributions = {}

        for driver_id in race_probs:
            # Create session probabilities
            race = SessionProbabilities(race_probs[driver_id])
            qualifying = (
                SessionProbabilities(qualifying_probs[driver_id])
                if qualifying_probs and driver_id in qualifying_probs
                else None
            )
            sprint = (
                SessionProbabilities(sprint_probs[driver_id])
                if sprint_probs and driver_id in sprint_probs
                else None
            )

            # Create joint probabilities if available
            joint = (
                JointProbabilities(
                    session1="qualifying",
                    session2="race",
                    probabilities=joint_qual_race[driver_id],
                )
                if joint_qual_race and driver_id in joint_qual_race
                else None
            )

            # Create driver distribution
            driver_distributions[driver_id] = DriverDistribution(
                race=race, qualifying=qualifying, sprint=sprint, joint_qual_race=joint
            )

        return cls(driver_distributions=driver_distributions)
