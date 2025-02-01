"""F1 module for managing Formula 1 data structures."""

from typing import Optional

from gridrival_ai.f1.core import Constructor
from gridrival_ai.f1.data import CONSTRUCTORS, DRIVER_TO_CONSTRUCTOR, PILOTS, RACE_LIST


def get_constructor(driver_id: str) -> Optional[Constructor]:
    """Get the constructor (team) for a given driver.

    Parameters
    ----------
    driver_id : str
        Three letter driver identifier

    Returns
    -------
    Optional[Constructor]
        The constructor object if driver exists, None otherwise
    """
    return DRIVER_TO_CONSTRUCTOR.get(driver_id)


def get_teammate(driver_id: str) -> Optional[str]:
    """Get the teammate's ID for a given driver.

    Parameters
    ----------
    driver_id : str
        Three letter driver identifier

    Returns
    -------
    Optional[str]
        The teammate's driver ID if driver exists, None otherwise
    """
    if constructor := DRIVER_TO_CONSTRUCTOR.get(driver_id):
        driver1, driver2 = constructor.drivers
        return driver2 if driver1 == driver_id else driver1
    return None


__all__ = [
    "PILOTS",
    "CONSTRUCTORS",
    "RACE_LIST",
    "get_constructor",
    "get_teammate",
]
