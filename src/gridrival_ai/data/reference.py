"""Static reference data for F1.

This module contains static data about F1 drivers, constructors, and races.
"""

from typing import Dict, List, Literal, Optional

from gridrival_ai.data.models import Constructor, Driver, Race

# List of all drivers
# Each tuple contains (driver_id, full name)
DRIVER_LIST: List[tuple[str, str]] = [
    ("VER", "Max Verstappen"),
    ("LAW", "Liam Lawson"),
    ("RUS", "George Russell"),
    ("ANT", "Andrea Kimi Antonelli"),
    ("LEC", "Charles Leclerc"),
    ("HAM", "Lewis Hamilton"),
    ("NOR", "Lando Norris"),
    ("PIA", "Oscar Piastri"),
    ("ALO", "Fernando Alonso"),
    ("STR", "Lance Stroll"),
    ("GAS", "Pierre Gasly"),
    ("DOO", "Jack Doohan"),
    ("ALB", "Alexander Albon"),
    ("SAI", "Carlos Sainz Jr."),
    ("TSU", "Yuki Tsunoda"),
    ("HAD", "Isack Hadjar"),
    ("HUL", "Nico Hülkenberg"),
    ("BOR", "Gabriel Bortoleto"),
    ("OCO", "Esteban Ocon"),
    ("BEA", "Oliver Bearman"),
]

# Type alias for driver IDs
DriverId = Literal[
    "VER",
    "LAW",
    "RUS",
    "ANT",
    "LEC",
    "HAM",
    "NOR",
    "PIA",
    "ALO",
    "STR",
    "GAS",
    "DOO",
    "ALB",
    "SAI",
    "TSU",
    "HAD",
    "HUL",
    "BOR",
    "OCO",
    "BEA",
]

# List of all constructors
# Each tuple contains (constructor_id, full name, (driver1_id, driver2_id))
CONSTRUCTOR_LIST: List[tuple[str, str, tuple[str, str]]] = [
    ("RBR", "Red Bull Racing", ("VER", "LAW")),
    ("MER", "Mercedes", ("RUS", "ANT")),
    ("FER", "Ferrari", ("LEC", "HAM")),
    ("MCL", "McLaren", ("NOR", "PIA")),
    ("AST", "Aston Martin", ("ALO", "STR")),
    ("ALP", "Alpine", ("GAS", "DOO")),
    ("WIL", "Williams", ("ALB", "SAI")),
    ("RBU", "Racing Bulls", ("TSU", "HAD")),
    ("SAU", "Kick Sauber", ("HUL", "BOR")),
    ("HAA", "Haas", ("OCO", "BEA")),
]

# List of all races
RACE_LIST = [
    Race("Australia", False),
    Race("China", True),
    Race("Japan", False),
    Race("Bahrain", False),
    Race("Saudi Arabia", False),
    Race("Miami", True),
    Race("Emilia Romagna", False),
    Race("Monaco", False),
    Race("Spain", False),
    Race("Canada", False),
    Race("Austria", False),
    Race("Britain", False),
    Race("Belgium", True),
    Race("Hungary", False),
    Race("Netherlands", False),
    Race("Italy", False),
    Race("Azerbaijan", False),
    Race("Singapore", False),
    Race("United States", True),
    Race("Mexico", False),
    Race("Brazil", True),
    Race("Las Vegas", False),
    Race("Qatar", True),
    Race("Abu Dhabi", False),
]

# Dictionary mappings
DRIVERS: Dict[str, Driver] = {
    driver_id: Driver(driver_id=driver_id, name=name) for driver_id, name in DRIVER_LIST
}

CONSTRUCTORS: Dict[str, Constructor] = {
    constructor_id: Constructor(
        constructor_id=constructor_id,
        name=name,
        drivers=drivers,
    )
    for constructor_id, name, drivers in CONSTRUCTOR_LIST
}

DRIVER_TO_CONSTRUCTOR: Dict[str, Constructor] = {
    driver_id: constructor
    for constructor in CONSTRUCTORS.values()
    for driver_id in constructor.drivers
}

# Validation sets
VALID_DRIVER_IDS: set[str] = {driver[0] for driver in DRIVER_LIST}
VALID_CONSTRUCTOR_IDS: set[str] = {constructor[0] for constructor in CONSTRUCTOR_LIST}


def get_driver(driver_id: str) -> Optional[Driver]:
    """Get a driver by ID.

    Parameters
    ----------
    driver_id : str
        The driver's ID

    Returns
    -------
    Optional[Driver]
        The driver if found, None otherwise
    """
    return DRIVERS.get(driver_id)


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
