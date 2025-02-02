"""
Data structures for managing F1 fantasy league state.

This module provides classes to manage the state of a GridRival F1 fantasy league,
including driver/constructor salaries, performance metrics, and team constraints.
"""

from dataclasses import dataclass

from gridrival_ai.data.reference import VALID_CONSTRUCTOR_IDS, VALID_DRIVER_IDS


@dataclass(frozen=True)
class Salaries:
    """Container for driver and constructor salaries.

    Parameters
    ----------
    drivers : Dict[str, float]
        Mapping of driver IDs to salaries in millions
    constructors : Dict[str, float]
        Mapping of constructor IDs to salaries in millions

    Examples
    --------
    >>> driver_salaries = {"VER": 33.0, "HAM": 26.2}
    >>> constructor_salaries = {"RBR": 30.0, "MER": 27.2}
    >>> salaries = Salaries(driver_salaries, constructor_salaries)
    """

    drivers: dict
    constructors: dict

    def __post_init__(self) -> None:
        """Validate salary data.

        Raises
        ------
        ValueError
            If any IDs are invalid or salaries are negative
        """
        # Validate driver IDs and salaries
        invalid_drivers = set(self.drivers) - VALID_DRIVER_IDS
        if invalid_drivers:
            raise ValueError(f"Invalid driver IDs: {invalid_drivers}")

        negative_drivers = {d: v for d, v in self.drivers.items() if v < 0}
        if negative_drivers:
            raise ValueError(f"Negative driver salaries: {negative_drivers}")

        # Validate constructor IDs and salaries
        invalid_constructors = set(self.constructors) - VALID_CONSTRUCTOR_IDS
        if invalid_constructors:
            raise ValueError(f"Invalid constructor IDs: {invalid_constructors}")

        negative_constructors = {c: v for c, v in self.constructors.items() if v < 0}
        if negative_constructors:
            raise ValueError(f"Negative constructor salaries: {negative_constructors}")


@dataclass(frozen=True)
class RollingAverages:
    """Container for driver performance averages.

    Parameters
    ----------
    values : Dict[str, float]
        Mapping of driver IDs to 8-race rolling average finish positions

    Examples
    --------
    >>> averages = {"VER": 1.5, "HAM": 3.2}
    >>> data = RollingAverages(averages)
    """

    values: dict

    def __post_init__(self) -> None:
        """Validate rolling averages.

        Raises
        ------
        ValueError
            If any IDs are invalid or averages outside valid range
        """
        # Validate driver IDs
        invalid_drivers = set(self.values) - VALID_DRIVER_IDS
        if invalid_drivers:
            raise ValueError(f"Invalid driver IDs: {invalid_drivers}")

        # Validate averages are in valid range (1 to 20)
        invalid_averages = {
            d: v for d, v in self.values.items() if not 1.0 <= v <= 20.0
        }
        if invalid_averages:
            raise ValueError(
                f"Invalid rolling averages (must be between 1-20): {invalid_averages}"
            )


@dataclass(frozen=True)
class TeamConstraints:
    """Container for team selection constraints.

    Parameters
    ----------
    locked_in : Set[str]
        Driver/constructor IDs that must be included in team
    locked_out : Set[str]
        Driver/constructor IDs that cannot be included in team

    Notes
    -----
    An element cannot be both locked in and locked out.
    All IDs must be valid driver or constructor IDs.

    Examples
    --------
    >>> constraints = TeamConstraints(
    ...     locked_in={"HAM", "RBR"},
    ...     locked_out={"VER"}
    ... )
    """

    locked_in: frozenset
    locked_out: frozenset

    def __post_init__(self) -> None:
        """Validate constraint consistency.

        Raises
        ------
        ValueError
            If constraints are invalid or inconsistent
        """
        # Check for overlap between locked_in and locked_out
        overlap = self.locked_in & self.locked_out
        if overlap:
            raise ValueError(f"Elements cannot be both locked in and out: {overlap}")

        # Validate all IDs
        valid_ids = VALID_DRIVER_IDS | VALID_CONSTRUCTOR_IDS
        invalid_locked_in = self.locked_in - valid_ids
        if invalid_locked_in:
            raise ValueError(f"Invalid locked in IDs: {invalid_locked_in}")

        invalid_locked_out = self.locked_out - valid_ids
        if invalid_locked_out:
            raise ValueError(f"Invalid locked out IDs: {invalid_locked_out}")


@dataclass(frozen=True)
class FantasyLeagueData:
    """Complete fantasy league state.

    Parameters
    ----------
    salaries : Salaries
        Current driver and constructor salaries
    averages : RollingAverages
        Driver performance averages
    constraints : TeamConstraints
        Team selection constraints

    Examples
    --------
    >>> # Using factory method
    >>> data = FantasyLeagueData.from_dicts(
    ...     driver_salaries={"VER": 33.0, "HAM": 26.2},
    ...     constructor_salaries={"RBR": 30.0},
    ...     rolling_averages={"VER": 1.5, "HAM": 3.2},
    ...     locked_in={"HAM"},
    ...     locked_out={"VER"}
    ... )
    """

    salaries: Salaries
    averages: RollingAverages
    constraints: TeamConstraints

    @classmethod
    def from_dicts(
        cls,
        driver_salaries: dict,
        constructor_salaries: dict,
        rolling_averages: dict,
        locked_in: frozenset | None = None,
        locked_out: frozenset | None = None,
    ) -> "FantasyLeagueData":
        """Create from dictionary inputs.

        Parameters
        ----------
        driver_salaries : Dict[str, float]
            Mapping of driver IDs to salaries
        constructor_salaries : Dict[str, float]
            Mapping of constructor IDs to salaries
        rolling_averages : Dict[str, float]
            Mapping of driver IDs to rolling averages
        locked_in : Set[str] | None, optional
            IDs that must be included, by default None
        locked_out : Set[str] | None, optional
            IDs that cannot be included, by default None

        Returns
        -------
        FantasyLeagueData
            Initialized fantasy league data

        Examples
        --------
        >>> data = FantasyLeagueData.from_dicts(
        ...     driver_salaries={"VER": 33.0},
        ...     constructor_salaries={"RBR": 30.0},
        ...     rolling_averages={"VER": 1.5},
        ...     locked_in={"HAM"},
        ...     locked_out={"VER"}
        ... )
        """
        return cls(
            salaries=Salaries(driver_salaries, constructor_salaries),
            averages=RollingAverages(rolling_averages),
            constraints=TeamConstraints(
                locked_in=frozenset(locked_in or set()),
                locked_out=frozenset(locked_out or set()),
            ),
        )

    def get_available_drivers(self) -> frozenset:
        """Get IDs of drivers available for selection.

        Returns
        -------
        Set[str]
            IDs of available drivers (not locked out)

        Examples
        --------
        >>> data = FantasyLeagueData.from_dicts(...)
        >>> available = data.get_available_drivers()
        >>> "VER" in available
        True
        """
        return frozenset(self.salaries.drivers) - self.constraints.locked_out

    def get_available_constructors(self) -> frozenset:
        """Get IDs of constructors available for selection.

        Returns
        -------
        Set[str]
            IDs of available constructors (not locked out)

        Examples
        --------
        >>> data = FantasyLeagueData.from_dicts(...)
        >>> available = data.get_available_constructors()
        >>> "RBR" in available
        True
        """
        return frozenset(self.salaries.constructors) - self.constraints.locked_out
