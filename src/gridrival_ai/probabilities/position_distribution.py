"""
Position probability distributions container for F1 fantasy optimization.

This module provides a high-level interface for accessing driver position
probabilities across different sessions.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from gridrival_ai.data.reference import CONSTRUCTORS
from gridrival_ai.probabilities.driver_distributions import DriverDistribution
from gridrival_ai.probabilities.types import JointProbabilities, SessionProbabilities


class PositionDistributions:
    """Container for F1 driver position probability distributions.

    This class provides a high-level interface for accessing driver position
    probabilities across different sessions. It acts as a manager for individual
    driver distributions, providing a session-focused query interface.

    Parameters
    ----------
    driver_distributions : dict[str, DriverDistribution]
        Mapping of driver three-letter codes to their probability distributions

    Notes
    -----
    Each session's probabilities are managed by the underlying DriverDistribution.
    Joint probabilities default to independence if not explicitly provided.

    Examples
    --------
    >>> # Create from existing driver distributions
    >>> driver_dists = {"VER": DriverDistribution(race=race_probs)}
    >>> dist = PositionDistributions(driver_dists)
    >>> p1_prob = dist.get_session_probabilities("VER", "qualifying")[1]

    >>> # Create joint distribution for two drivers
    >>> joint = dist.get_driver_pair_distribution("VER", "PER", session="race")
    >>> p1_p2_prob = joint[(1, 2)]  # Probability VER P1, PER P2
    """

    VALID_SESSIONS = {"qualifying", "race", "sprint"}

    def __init__(self, driver_distributions: dict[str, DriverDistribution]):
        """Initialize with driver distributions."""
        self.driver_distributions = driver_distributions

    def get_constructor_drivers(self, constructor_id: str) -> tuple[str, str] | None:
        """
        Get the drivers for a constructor.

        Parameters
        ----------
        constructor_id : str
            Constructor ID to get drivers for.

        Returns
        -------
        tuple[str, str] | None
            Tuple of driver IDs if found, None otherwise.
        """
        constructor = CONSTRUCTORS.get(constructor_id)
        return constructor.drivers if constructor else None

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
        """
        if session not in self.VALID_SESSIONS:
            raise ValueError(f"Invalid session name: {session}")

        return getattr(self.driver_distributions[driver_id], session)

    def get_driver_pair_distribution(
        self, driver1_id: str, driver2_id: str, session: str = "race"
    ) -> JointProbabilities:
        """Create joint distribution for two drivers enforcing position constraints.

        Creates a joint probability distribution ensuring that two drivers cannot
        finish in the same position. This is useful for analyzing relationships
        between drivers' finishing positions, especially teammates.

        Parameters
        ----------
        driver1_id : str
            First driver's ID
        driver2_id : str
            Second driver's ID
        session : str, optional
            Session type ("qualifying", "race", "sprint"), by default "race"

        Returns
        -------
        JointProbabilities
            Joint distribution ensuring drivers can't finish in same position

        Notes
        -----
        The method:
        1. Takes individual driver distributions
        2. Creates joint probabilities assuming independence
        3. Removes impossible cases (same position finishes)
        4. Renormalizes to maintain valid probability distribution

        Examples
        --------
        >>> joint = dist.get_driver_pair_distribution("VER", "PER")
        >>> p1_p2_prob = joint[(1, 2)]  # Probability VER P1, PER P2
        """
        dist1 = self.get_session_probabilities(driver1_id, session)
        dist2 = self.get_session_probabilities(driver2_id, session)

        # Create constrained distribution
        joint_probs = {
            (i, j): dist1[i] * dist2[j]
            for i in dist1.probabilities
            for j in dist2.probabilities
            if i != j  # Exclude same positions
        }

        # Normalize to maintain valid probability distribution
        total = sum(joint_probs.values())
        normalized = {k: v / total for k, v in joint_probs.items()}

        return JointProbabilities(
            session1=session, session2=session, probabilities=normalized
        )

    def get_driver_session_correlation(
        self, driver_id: str, session1: str, session2: str
    ) -> JointProbabilities:
        """Get correlation between a driver's positions across two sessions.

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
            mapping position pairs to probabilities. For example, (3,1) would give
            the probability of P3 in session1 and P1 in session2.

        Raises
        ------
        KeyError
            If driver_id not found or sessions not available
        ValueError
            If session combination is not supported
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

    def get_available_sessions(self) -> set[str]:
        """Get set of available session names.

        Returns
        -------
        set[str]
            Names of available sessions
        """
        # All drivers have same sessions due to DriverDistribution defaults
        return {"qualifying", "race", "sprint"}
