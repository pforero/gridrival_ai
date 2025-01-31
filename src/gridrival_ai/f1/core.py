"""Core classes for F1 data structures."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Pilot:
    """F1 pilot representation.

    Parameters
    ----------
    driver_id : str
        Unique identifier for the driver
    name : str
        Full name of the driver
    abbreviation : str
        Three letter abbreviation used in timing screens

    Raises
    ------
    ValueError
        If abbreviation is not exactly 3 letters
    """

    driver_id: str
    name: str
    abbreviation: str

    def __post_init__(self) -> None:
        """Validate the abbreviation."""
        if len(self.abbreviation) != 3:
            raise ValueError("Abbreviation must be exactly 3 letters")
        if not self.abbreviation.isalpha():
            raise ValueError("Abbreviation must contain only letters")


@dataclass(frozen=True)
class Constructor:
    """F1 constructor (team) representation.

    Parameters
    ----------
    constructor_id : str
        Unique identifier for the constructor
    name : str
        Full name of the constructor
    abbreviation : str
        Three letter abbreviation used in timing screens
    drivers : List[Pilot]
        List of two drivers for this constructor

    Raises
    ------
    ValueError
        If abbreviation is not exactly 3 letters or if not exactly 2 drivers
    """

    constructor_id: str
    name: str
    abbreviation: str
    drivers: List[Pilot]

    def __post_init__(self) -> None:
        """Validate the abbreviation and drivers."""
        if len(self.abbreviation) != 3:
            raise ValueError("Abbreviation must be exactly 3 letters")
        if not self.abbreviation.isalpha():
            raise ValueError("Abbreviation must contain only letters")
        if len(self.drivers) != 2:
            raise ValueError("Constructor must have exactly 2 drivers")


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
    is_sprint: bool

    def __post_init__(self) -> None:
        """Validate the race name."""
        if not self.name:
            raise ValueError("Race name cannot be empty")
