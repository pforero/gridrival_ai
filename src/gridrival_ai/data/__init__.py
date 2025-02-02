"""Data package for F1 and fantasy league data models and utilities."""

from gridrival_ai.data.models import Constructor, Driver, Race
from gridrival_ai.data.reference import (
    CONSTRUCTORS,
    DRIVER_TO_CONSTRUCTOR,
    DRIVERS,
    RACE_LIST,
    VALID_CONSTRUCTOR_IDS,
    VALID_DRIVER_IDS,
    get_constructor,
    get_driver,
    get_teammate,
)

__all__ = [
    # Models
    "Constructor",
    "Driver",
    "Race",
    # Reference data
    "CONSTRUCTORS",
    "DRIVERS",
    "DRIVER_TO_CONSTRUCTOR",
    "RACE_LIST",
    "VALID_CONSTRUCTOR_IDS",
    "VALID_DRIVER_IDS",
    "get_constructor",
    "get_driver",
    "get_teammate",
]
