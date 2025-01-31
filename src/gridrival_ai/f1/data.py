"""2025 F1 season data."""

from typing import Dict, List, Literal

from gridrival_ai.f1.core import Constructor, Pilot, Race

# List of all drivers for 2025 season
# Each tuple contains (driver_id, full name)
DRIVERS_2025: List[tuple[str, str]] = [
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

# List of all constructors for 2025 season
# Each tuple contains (constructor_id, full name, (driver1_id, driver2_id))
CONSTRUCTORS_LIST_2025: List[tuple[str, str, tuple[str, str]]] = [
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

# Create DriverId type dynamically from the drivers list
DriverId = Literal[tuple(driver[0] for driver in DRIVERS_2025)]  # type: ignore

# Create ConstructorId type dynamically from the constructors list
ConstructorId = Literal[tuple(constructor[0] for constructor in CONSTRUCTORS_LIST_2025)]  # type: ignore

# Create pilots dictionary dynamically
PILOTS_2025: Dict[str, Pilot] = {
    driver_id: Pilot(driver_id=driver_id, name=name) for driver_id, name in DRIVERS_2025
}

# Create constructors dictionary dynamically
CONSTRUCTORS_2025: Dict[str, Constructor] = {
    constructor_id: Constructor(
        constructor_id=constructor_id, name=name, drivers=drivers
    )
    for constructor_id, name, drivers in CONSTRUCTORS_LIST_2025
}

# Create all races
RACES_2025 = [
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

# For runtime validation
VALID_DRIVER_IDS: set[str] = {driver[0] for driver in DRIVERS_2025}
VALID_CONSTRUCTOR_IDS: set[str] = {
    constructor[0] for constructor in CONSTRUCTORS_LIST_2025
}
