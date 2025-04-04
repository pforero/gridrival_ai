"""
Distribution class for representing probabilities across all sessions in a race weekend.

This module provides the RaceDistribution class for coordinating probability
distributions across race, qualifying, and sprint sessions, along with
completion probabilities and joint distribution operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Union

from gridrival_ai.probabilities.distributions.joint import (
    JointDistribution,
    create_independent_joint,
)
from gridrival_ai.probabilities.distributions.position import PositionDistribution
from gridrival_ai.probabilities.distributions.session import SessionDistribution
from gridrival_ai.probabilities.odds_structure import OddsStructure


@dataclass(frozen=True)
class RaceDistribution:
    """
    Probability distributions for all sessions in a race weekend.

    This class holds session distributions for 'race', 'qualifying', and 'sprint',
    along with completion probabilities and methods for deriving joint distributions.

    Parameters
    ----------
    race : SessionDistribution
        Distribution for race session
    qualifying : SessionDistribution, optional
        Distribution for qualifying session, by default copied from race
    sprint : SessionDistribution, optional
        Distribution for sprint session, by default copied from race
    completion_probabilities : Dict[str, float], optional
        Dictionary mapping driver IDs to completion probabilities, by default empty
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
    >>>
    >>> # Get joint distribution between qualifying and race
    >>> joint_dist = race_dist.get_qualifying_race_distribution("VER")
    >>> joint_dist[(1, 1)]  # P(quali=1, race=1)
    0.36
    """

    race: SessionDistribution
    qualifying: SessionDistribution = None  # Will be set in __post_init__
    sprint: SessionDistribution = None  # Will be set in __post_init__
    completion_probabilities: Dict[str, float] = field(default_factory=dict)
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
        - Completion probabilities are valid (between 0 and 1)

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

        # Validate completion probabilities
        for driver_id, prob in self.completion_probabilities.items():
            if not 0 <= prob <= 1:
                raise ValueError(
                    f"Completion probability for driver {driver_id} must be "
                    f"between 0 and 1, got {prob}"
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

    def get_qualifying_race_distribution(self, driver_id: str) -> JointDistribution:
        """
        Get joint distribution between qualifying and race positions.

        This method creates a joint distribution representing the relationship
        between a driver's qualifying and race positions, assuming independence
        between the two sessions.

        Parameters
        ----------
        driver_id : str
            Driver ID

        Returns
        -------
        JointDistribution
            Joint distribution between qualifying and race positions

        Raises
        ------
        KeyError
            If driver not found in qualifying or race sessions
        ValueError
            If session types are invalid

        Examples
        --------
        >>> race_dist = RaceDistribution(race_session)
        >>> joint_dist = race_dist.get_qualifying_race_distribution("VER")
        >>> # Probability of qualifying P1 and finishing race P1
        >>> joint_dist[(1, 1)]
        0.36
        """
        qual_dist = self.get_driver_distribution(driver_id, "qualifying")
        race_dist = self.get_driver_distribution(driver_id, "race")

        return create_independent_joint(
            qual_dist, race_dist, name1="qualifying", name2="race"
        )

    def get_completion_probability(self, driver_id: str) -> float:
        """
        Get race completion probability for a driver.

        This method returns the probability that a driver completes the race
        without retiring or being disqualified.

        Parameters
        ----------
        driver_id : str
            Driver ID

        Returns
        -------
        float
            Probability of completing the race (between 0 and 1)

        Raises
        ------
        KeyError
            If driver's completion probability not found

        Examples
        --------
        >>> race_dist = RaceDistribution(
        ...     race_session,
        ...     completion_probabilities={"VER": 0.95, "HAM": 0.98}
        ... )
        >>> race_dist.get_completion_probability("VER")
        0.95
        """
        if driver_id not in self.completion_probabilities:
            raise KeyError(f"Completion probability for driver {driver_id} not found")
        return self.completion_probabilities[driver_id]

    @classmethod
    def from_structured_odds(
        cls,
        odds_structure: Union[Dict, OddsStructure],
        grid_method: str = "harville",
        normalization_method: str = "sinkhorn",
        odds_method: str = "basic",
        completion_probabilities: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> "RaceDistribution":
        from gridrival_ai.probabilities.grid_creators.factory import get_grid_creator
        from gridrival_ai.probabilities.normalizers.factory import get_grid_normalizer
        from gridrival_ai.probabilities.odds_converters.factory import (
            get_odds_converter,
        )

        # Convert to OddsStructure if needed
        if not isinstance(odds_structure, OddsStructure):
            odds_structure = OddsStructure(odds_structure)

        # Create the grid creator with its method and kwargs
        grid_creator = get_grid_creator(method=grid_method, **kwargs)

        # Create and set the converter and normalizer separately
        grid_creator.odds_converter = get_odds_converter(method=odds_method, **kwargs)
        grid_creator.grid_normalizer = get_grid_normalizer(
            method=normalization_method, **kwargs
        )

        # Create session distributions for available sessions
        race_dist = grid_creator.create_session_distribution(
            odds_structure, session_type="race", **kwargs
        )

        quali_dist = None
        if "qualifying" in odds_structure.sessions:
            quali_dist = grid_creator.create_session_distribution(
                odds_structure, session_type="qualifying", **kwargs
            )

        sprint_dist = None
        if "sprint" in odds_structure.sessions:
            sprint_dist = grid_creator.create_session_distribution(
                odds_structure, session_type="sprint", **kwargs
            )

        # Create and return RaceDistribution
        return cls(
            race=race_dist,
            qualifying=quali_dist,
            sprint=sprint_dist,
            completion_probabilities=completion_probabilities or {},
        )
