"""2025 F1 season data."""

from gridrival_ai.f1.core import Constructor, Pilot, Race

# Create all pilots
PILOTS_2025 = {
    "VER": Pilot("VER", "Max Verstappen", "VER"),
    "LAW": Pilot("LAW", "Liam Lawson", "LAW"),
    "RUS": Pilot("RUS", "George Russell", "RUS"),
    "ANT": Pilot("ANT", "Andrea Kimi Antonelli", "ANT"),
    "LEC": Pilot("LEC", "Charles Leclerc", "LEC"),
    "HAM": Pilot("HAM", "Lewis Hamilton", "HAM"),
    "NOR": Pilot("NOR", "Lando Norris", "NOR"),
    "PIA": Pilot("PIA", "Oscar Piastri", "PIA"),
    "ALO": Pilot("ALO", "Fernando Alonso", "ALO"),
    "STR": Pilot("STR", "Lance Stroll", "STR"),
    "GAS": Pilot("GAS", "Pierre Gasly", "GAS"),
    "DOO": Pilot("DOO", "Jack Doohan", "DOO"),
    "ALB": Pilot("ALB", "Alexander Albon", "ALB"),
    "SAI": Pilot("SAI", "Carlos Sainz Jr.", "SAI"),
    "TSU": Pilot("TSU", "Yuki Tsunoda", "TSU"),
    "HAD": Pilot("HAD", "Isack Hadjar", "HAD"),
    "HUL": Pilot("HUL", "Nico Hülkenberg", "HUL"),
    "BOR": Pilot("BOR", "Gabriel Bortoleto", "BOR"),
    "OCO": Pilot("OCO", "Esteban Ocon", "OCO"),
    "BEA": Pilot("BEA", "Oliver Bearman", "BEA"),
}

# Create all constructors
CONSTRUCTORS_2025 = {
    "RBR": Constructor("RBR", "Red Bull Racing", "RBR",
                      [PILOTS_2025["VER"], PILOTS_2025["LAW"]]),
    "MER": Constructor("MER", "Mercedes", "MER",
                      [PILOTS_2025["RUS"], PILOTS_2025["ANT"]]),
    "FER": Constructor("FER", "Ferrari", "FER",
                      [PILOTS_2025["LEC"], PILOTS_2025["HAM"]]),
    "MCL": Constructor("MCL", "McLaren", "MCL",
                      [PILOTS_2025["NOR"], PILOTS_2025["PIA"]]),
    "AST": Constructor("AST", "Aston Martin", "AST",
                      [PILOTS_2025["ALO"], PILOTS_2025["STR"]]),
    "ALP": Constructor("ALP", "Alpine", "ALP",
                      [PILOTS_2025["GAS"], PILOTS_2025["DOO"]]),
    "WIL": Constructor("WIL", "Williams", "WIL",
                      [PILOTS_2025["ALB"], PILOTS_2025["SAI"]]),
    "RBU": Constructor("RBU", "Racing Bulls", "RBU",
                      [PILOTS_2025["TSU"], PILOTS_2025["HAD"]]),
    "SAU": Constructor("SAU", "Kick Sauber", "SAU",
                      [PILOTS_2025["HUL"], PILOTS_2025["BOR"]]),
    "HAA": Constructor("HAA", "Haas", "HAA",
                      [PILOTS_2025["OCO"], PILOTS_2025["BEA"]]),
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
