"""
Data structures for managing F1 fantasy league state.

This module provides classes to manage the state of a GridRival F1 fantasy league,
including driver/constructor salaries and performance metrics.
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
class FantasyLeagueData:
    """Fantasy league state containing salaries and performance metrics.

    Parameters
    ----------
    salaries : Salaries
        Current driver and constructor salaries
    averages : RollingAverages
        Driver performance averages

    Examples
    --------
    >>> # Using factory method
    >>> data = FantasyLeagueData.from_dicts(
    ...     driver_salaries={"VER": 33.0, "HAM": 26.2},
    ...     constructor_salaries={"RBR": 30.0},
    ...     rolling_averages={"VER": 1.5, "HAM": 3.2}
    ... )
    """

    salaries: Salaries
    averages: RollingAverages

    @classmethod
    def from_dicts(
        cls,
        driver_salaries: dict,
        constructor_salaries: dict,
        rolling_averages: dict,
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

        Returns
        -------
        FantasyLeagueData
            Initialized fantasy league data

        Examples
        --------
        >>> data = FantasyLeagueData.from_dicts(
        ...     driver_salaries={"VER": 33.0},
        ...     constructor_salaries={"RBR": 30.0},
        ...     rolling_averages={"VER": 1.5}
        ... )
        """
        return cls(
            salaries=Salaries(driver_salaries, constructor_salaries),
            averages=RollingAverages(rolling_averages),
        )

    def get_all_drivers(self) -> frozenset:
        """Get IDs of all available drivers.

        Returns
        -------
        Set[str]
            IDs of all drivers in the league

        Examples
        --------
        >>> data = FantasyLeagueData.from_dicts(...)
        >>> all_drivers = data.get_all_drivers()
        >>> "VER" in all_drivers
        True
        """
        return frozenset(self.salaries.drivers)

    def get_all_constructors(self) -> frozenset:
        """Get IDs of all available constructors.

        Returns
        -------
        Set[str]
            IDs of all constructors in the league

        Examples
        --------
        >>> data = FantasyLeagueData.from_dicts(...)
        >>> all_constructors = data.get_all_constructors()
        >>> "RBR" in all_constructors
        True
        """
        return frozenset(self.salaries.constructors)
