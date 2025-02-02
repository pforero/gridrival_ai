"""Core F1 data models.

This module contains the core data models used to represent F1-related entities.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Driver:
    """F1 driver representation.

    Parameters
    ----------
    driver_id : str
        Three letter abbreviation used as unique identifier
    name : str
        Full name of the driver

    Raises
    ------
    ValueError
        If driver_id is not exactly 3 letters
    """

    driver_id: str
    name: str

    def __post_init__(self) -> None:
        """Validate the driver_id."""
        if len(self.driver_id) != 3:
            raise ValueError("Driver ID must be exactly 3 letters")
        if not self.driver_id.isalpha():
            raise ValueError("Driver ID must contain only letters")
        if not self.driver_id.isupper():
            raise ValueError("Driver ID must be uppercase")


@dataclass(frozen=True)
class Constructor:
    """F1 constructor (team) representation.

    Parameters
    ----------
    constructor_id : str
        Three letter abbreviation used as unique identifier
    name : str
        Full name of the constructor
    drivers : tuple[str, str]
        Tuple of two driver IDs for this constructor

    Raises
    ------
    ValueError
        If constructor_id is not exactly 3 letters or if not exactly 2 drivers
    """

    constructor_id: str
    name: str
    drivers: tuple[str, str]

    def __post_init__(self) -> None:
        """Validate the constructor_id and drivers."""
        if len(self.constructor_id) != 3:
            raise ValueError("Constructor ID must be exactly 3 letters")
        if not self.constructor_id.isalpha():
            raise ValueError("Constructor ID must contain only letters")
        if not self.constructor_id.isupper():
            raise ValueError("Constructor ID must be uppercase")
        if len(self.drivers) != 2:
            raise ValueError("Constructor must have exactly 2 drivers")
        for driver_id in self.drivers:
            if len(driver_id) != 3:
                raise ValueError("Driver ID must be exactly 3 letters")
            if not driver_id.isalpha():
                raise ValueError("Driver ID must contain only letters")
            if not driver_id.isupper():
                raise ValueError("Driver ID must be uppercase")


@dataclass(frozen=True)
class Race:
    """F1 race representation.

    Parameters
    ----------
    name : str
        Name of the grand prix
    is_sprint : bool
        Whether this race includes a sprint race
    """

    name: str
    is_sprint: bool = False

    def __post_init__(self) -> None:
        """Validate the race data."""
        if not self.name:
            raise ValueError("Race name cannot be empty")
