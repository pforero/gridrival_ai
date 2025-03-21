"""
Distribution class for representing probabilities across all sessions in a race weekend.

This module provides the RaceDistribution class for coordinating probability
distributions across race, qualifying, and sprint sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

from gridrival_ai.probabilities.distributions.position import PositionDistribution
from gridrival_ai.probabilities.distributions.session import SessionDistribution


@dataclass(frozen=True)
class RaceDistribution:
    """
    Probability distributions for all sessions in a race weekend.

    This class holds session distributions for 'race', 'qualifying', and 'sprint'.
    If qualifying or sprint are not provided, they are copied from the race session.

    Parameters
    ----------
    race : SessionDistribution
        Distribution for race session
    qualifying : SessionDistribution, optional
        Distribution for qualifying session, by default copied from race
    sprint : SessionDistribution, optional
        Distribution for sprint session, by default copied from race
    validate : bool, optional
        Whether to validate distributions on initialization, by default True

    Raises
    ------
    ValueError
        If validation fails and validate=True

    Examples
    --------
    >>> # Create session distributions
    >>> race_session = SessionDistribution(race_distributions, "race")
    >>>
    >>> # Create race distribution with all sessions derived from race
    >>> race_dist = RaceDistribution(race_session)
    >>>
    >>> # Get distribution for a driver in a session
    >>> race_dist.get_driver_distribution("VER", "race")[1]  # P1 probability in race
    0.6
    """

    race: SessionDistribution
    qualifying: SessionDistribution = None  # Will be set in __post_init__
    sprint: SessionDistribution = None  # Will be set in __post_init__
    _validate: bool = field(default=True, repr=False)

    def __post_init__(self):
        """
        Initialize and validate distributions after initialization.

        If qualifying or sprint are not provided, they are copied from the race session.
        """
        # Create a qualifying distribution if not provided
        object.__setattr__(
            self,
            "qualifying",
            self.qualifying
            or SessionDistribution(
                self.race.driver_distributions, "qualifying", _validate=False
            ),
        )

        # Create a sprint distribution if not provided
        object.__setattr__(
            self,
            "sprint",
            self.sprint
            or SessionDistribution(
                self.race.driver_distributions, "sprint", _validate=False
            ),
        )

        if self._validate:
            self.validate()

    def validate(self) -> None:
        """
        Validate race distributions.

        Checks:
        - Session types match their attribute names
        - All sessions have the same drivers

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check session types
        if not isinstance(self.race, SessionDistribution):
            raise ValueError("Race distribution must be a SessionDistribution")

        if self.race.session_type != "race":
            raise ValueError("Race distribution must have session_type 'race'")

        if self.qualifying.session_type != "qualifying":
            raise ValueError(
                "Qualifying distribution must have session_type 'qualifying'"
            )

        if self.sprint.session_type != "sprint":
            raise ValueError("Sprint distribution must have session_type 'sprint'")

        # Check that all sessions have the same drivers
        race_drivers = set(self.race.get_driver_ids())
        quali_drivers = set(self.qualifying.get_driver_ids())
        sprint_drivers = set(self.sprint.get_driver_ids())

        if race_drivers != quali_drivers:
            raise ValueError(
                f"Race and qualifying sessions have different drivers. "
                f"Race: {race_drivers}, Qualifying: {quali_drivers}"
            )

        if race_drivers != sprint_drivers:
            raise ValueError(
                f"Race and sprint sessions have different drivers. "
                f"Race: {race_drivers}, Sprint: {sprint_drivers}"
            )

    def get_session(self, session_type: str) -> SessionDistribution:
        """
        Get distribution for a specific session.

        Parameters
        ----------
        session_type : str
            Session type ('race', 'qualifying', 'sprint')

        Returns
        -------
        SessionDistribution
            Session distribution

        Raises
        ------
        ValueError
            If session type is invalid
        """
        if session_type == "race":
            return self.race
        elif session_type == "qualifying":
            return self.qualifying
        elif session_type == "sprint":
            return self.sprint
        else:
            raise ValueError(
                f"Invalid session type: {session_type}. "
                f"Must be 'race', 'qualifying', or 'sprint'."
            )

    def get_driver_distribution(
        self, driver_id: str, session_type: str
    ) -> PositionDistribution:
        """
        Get position distribution for a specific driver and session.

        Parameters
        ----------
        driver_id : str
            Driver ID
        session_type : str
            Session type ('race', 'qualifying', 'sprint')

        Returns
        -------
        PositionDistribution
            Position distribution

        Raises
        ------
        ValueError
            If session type is invalid
        KeyError
            If driver not found in the session
        """
        session = self.get_session(session_type)
        return session.get_driver_distribution(driver_id)

    def get_driver_ids(self) -> Set[str]:
        """
        Get all driver IDs (same across all sessions).

        Returns
        -------
        Set[str]
            Set of driver IDs
        """
        return set(self.race.get_driver_ids())
