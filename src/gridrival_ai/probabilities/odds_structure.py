"""
Odds structure data model for GridRival AI.

This module provides a data class for representing and validating
F1 betting odds structures used in the probability calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class OddsStructure:
    """
    Validated structure of F1 betting odds.

    This class represents a validated odds structure for F1 races,
    ensuring consistency across sessions, positions, and drivers.
    It handles validation, normalization, and provides defaults
    for missing data.

    Parameters
    ----------
    odds : Dict[str, Dict[int, Dict[str, float]]]
        Nested dictionary of odds with format:
        {
            "race": {
                1: {"VER": 2.5, "HAM": 4.0, ...},  # Win odds
                3: {"VER": 1.5, "HAM": 2.0, ...},  # Top-3 odds
                ...
            },
            "qualifying": {...},
            "sprint": {...}
        }
    validate : bool, optional
        Whether to validate the odds on initialization, by default True
    auto_complete : bool, optional
        Whether to automatically fill in missing sessions using race data,
        by default True

    Attributes
    ----------
    odds : Dict[str, Dict[int, Dict[str, float]]]
        Validated odds structure
    drivers : Set[str]
        Set of all drivers in the odds structure
    sessions : List[str]
        List of all sessions in the odds structure

    Methods
    -------
    validate()
        Validate the odds structure
    get_session_odds(session)
        Get odds for a specific session
    get_driver_odds(driver_id, session)
        Get odds for a specific driver in a session
    get_position_odds(position, session)
        Get odds for a specific position in a session
    """

    odds: Dict[str, Dict[int, Dict[str, float]]]
    validate: bool = field(default=True, repr=False)
    auto_complete: bool = field(default=True, repr=False)

    # Internal state
    _drivers: Set[str] = field(default_factory=set, init=False, repr=False)
    _sessions: List[str] = field(default_factory=list, init=False, repr=False)
    _is_validated: bool = field(default=False, init=False, repr=False)

    VALID_SESSIONS = ["race", "qualifying", "sprint"]

    def __post_init__(self) -> None:
        """Validate and process the odds structure after initialization."""
        # Ensure odds is a dictionary
        if not isinstance(self.odds, dict):
            raise TypeError("Odds must be a dictionary")

        # Initialize with at least an empty race dictionary if not present
        if "race" not in self.odds:
            self.odds["race"] = {}

        # Perform validation if requested
        if self.validate:
            self.validate_odds()

        # Auto-complete missing sessions if requested
        if self.auto_complete:
            self._auto_complete_sessions()

        # Update internal state
        self._update_internal_state()

    def validate_odds(self) -> None:
        """
        Validate the odds structure.

        Checks:
        - All sessions are valid
        - All odds are > 1.0
        - Position thresholds are valid
        - At least win odds are provided for race

        Raises
        ------
        ValueError
            If the odds structure is invalid
        """
        # Check that at least race session exists
        if "race" not in self.odds:
            raise ValueError("Odds must include 'race' session")

        # Check all sessions are valid
        for session in self.odds:
            if session not in self.VALID_SESSIONS:
                raise ValueError(
                    f"Invalid session '{session}'. Valid sessions: "
                    f"{self.VALID_SESSIONS}"
                )

        # Get all drivers across all sessions for consistency checking
        all_drivers: Set[str] = set()
        for session, positions in self.odds.items():
            for position, drivers in positions.items():
                all_drivers.update(drivers.keys())

        # Validate each session's odds
        for session, positions in self.odds.items():
            # Check position thresholds
            for position in positions:
                if not isinstance(position, int) or position < 1:
                    raise ValueError(
                        f"Position thresholds must be positive integers, got {position}"
                    )

                # Position shouldn't exceed driver count
                if position > len(all_drivers):
                    raise ValueError(
                        f"Position threshold {position} exceeds driver count "
                        f"{len(all_drivers)}"
                    )

                # Check driver odds
                for driver, odd in positions[position].items():
                    if not isinstance(odd, (int, float)) or odd <= 1.0:
                        raise ValueError(
                            f"Invalid odd {odd} for {driver} in {session} position "
                            f"{position}. "
                            "Odds must be > 1.0"
                        )

        # For race, we require at least win odds (position 1)
        if 1 not in self.odds.get("race", {}):
            raise ValueError("Race odds must include win odds (position 1)")

        self._is_validated = True

    def _auto_complete_sessions(self) -> None:
        """
        Fill in missing sessions with race data.

        If qualifying or sprint sessions are missing, they will be
        filled with the race data.
        """
        if "race" not in self.odds:
            return  # Nothing to complete with

        race_odds = self.odds["race"]

        # Fill in qualifying if missing
        if "qualifying" not in self.odds or not self.odds["qualifying"]:
            self.odds["qualifying"] = race_odds.copy()

        # Fill in sprint if missing
        if "sprint" not in self.odds or not self.odds["sprint"]:
            self.odds["sprint"] = race_odds.copy()

    def _update_internal_state(self) -> None:
        """Update internal state based on the current odds structure."""
        # Get all drivers
        self._drivers = set()
        for session, positions in self.odds.items():
            for position, drivers in positions.items():
                self._drivers.update(drivers.keys())

        # Get all sessions
        self._sessions = list(self.odds.keys())

    @property
    def drivers(self) -> Set[str]:
        """Get the set of all drivers in the odds structure."""
        return self._drivers

    @property
    def sessions(self) -> List[str]:
        """Get the list of all sessions in the odds structure."""
        return self._sessions

    def get_session_odds(self, session: str) -> Dict[int, Dict[str, float]]:
        """
        Get odds for a specific session.

        Parameters
        ----------
        session : str
            Session name ("race", "qualifying", or "sprint")

        Returns
        -------
        Dict[int, Dict[str, float]]
            Dictionary mapping position thresholds to driver odds

        Raises
        ------
        ValueError
            If the session is invalid
        """
        if session not in self.VALID_SESSIONS:
            raise ValueError(
                f"Invalid session '{session}'. Valid sessions: {self.VALID_SESSIONS}"
            )

        return self.odds.get(session, {})

    def get_driver_odds(
        self, driver_id: str, session: str = "race"
    ) -> Dict[int, float]:
        """
        Get odds for a specific driver in a session.

        Parameters
        ----------
        driver_id : str
            Driver ID
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        Dict[int, float]
            Dictionary mapping position thresholds to odds
        """
        session_odds = self.get_session_odds(session)
        driver_odds = {}

        for position, drivers in session_odds.items():
            if driver_id in drivers:
                driver_odds[position] = drivers[driver_id]

        return driver_odds

    def get_position_odds(
        self, position: int, session: str = "race"
    ) -> Dict[str, float]:
        """
        Get odds for a specific position in a session.

        Parameters
        ----------
        position : int
            Position threshold (1 for win, 3 for top-3, etc.)
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        Dict[str, float]
            Dictionary mapping driver IDs to odds

        Raises
        ------
        ValueError
            If the position is not found in the session
        """
        session_odds = self.get_session_odds(session)

        if position not in session_odds:
            raise ValueError(f"Position {position} not found in {session} odds")

        return session_odds[position]

    def get_win_odds(self, session: str = "race") -> Dict[str, float]:
        """
        Get win odds for a session.

        Parameters
        ----------
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        Dict[str, float]
            Dictionary mapping driver IDs to win odds

        Raises
        ------
        ValueError
            If win odds not found for the session
        """
        return self.get_position_odds(1, session)

    def get_win_odds_list(self, session: str = "race") -> Tuple[List[float], List[str]]:
        """
        Get win odds as a list with corresponding driver IDs.

        Parameters
        ----------
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        Tuple[List[float], List[str]]
            Tuple of (odds_list, driver_ids_list)
        """
        win_odds = self.get_win_odds(session)
        driver_ids = list(win_odds.keys())
        odds_list = [win_odds[driver_id] for driver_id in driver_ids]

        return odds_list, driver_ids

    def get_thresholds(self, session: str = "race") -> List[int]:
        """
        Get position thresholds for a session.

        Parameters
        ----------
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        List[int]
            Sorted list of position thresholds
        """
        session_odds = self.get_session_odds(session)
        return sorted(session_odds.keys())

    def get_cumulative_odds(self, session: str = "race") -> Dict[int, Dict[str, float]]:
        """
        Get cumulative odds for a session.

        This is just an alias for get_session_odds for clarity.

        Parameters
        ----------
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        Dict[int, Dict[str, float]]
            Dictionary mapping position thresholds to driver odds
        """
        return self.get_session_odds(session)

    @classmethod
    def from_win_odds(
        cls, win_odds: Dict[str, float], session: str = "race"
    ) -> OddsStructure:
        """
        Create an OddsStructure from win odds only.

        Parameters
        ----------
        win_odds : Dict[str, float]
            Dictionary mapping driver IDs to win odds
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        OddsStructure
            Validated odds structure with only win odds

        Raises
        ------
        ValueError
            If the session is invalid
        """
        if session not in cls.VALID_SESSIONS:
            raise ValueError(
                f"Invalid session '{session}'. Valid sessions: {cls.VALID_SESSIONS}"
            )

        odds = {session: {1: win_odds}}

        # Add win odds to race session if different session specified
        if session != "race":
            odds["race"] = {1: win_odds}

        return cls(odds)

    @classmethod
    def from_win_odds_list(
        cls, odds_list: List[float], driver_ids: List[str], session: str = "race"
    ) -> OddsStructure:
        """
        Create an OddsStructure from a list of win odds and driver IDs.

        Parameters
        ----------
        odds_list : List[float]
            List of win odds
        driver_ids : List[str]
            List of driver IDs corresponding to odds
        session : str, optional
            Session name, by default "race"

        Returns
        -------
        OddsStructure
            Validated odds structure with only win odds

        Raises
        ------
        ValueError
            If lengths of odds_list and driver_ids don't match
        """
        if len(odds_list) != len(driver_ids):
            raise ValueError(
                f"Length of odds_list ({len(odds_list)}) must match "
                f"length of driver_ids ({len(driver_ids)})"
            )

        win_odds = {driver: odd for driver, odd in zip(driver_ids, odds_list)}
        return cls.from_win_odds(win_odds, session)
