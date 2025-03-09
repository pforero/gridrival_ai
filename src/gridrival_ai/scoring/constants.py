"""
Default configurations and constants for F1 fantasy scoring.

This module contains default point configurations and validation rules for the
GridRival F1 fantasy scoring system. Values can be overridden via configuration
files.

Notes
-----
All position-based dictionaries use 1-based indexing to match F1 positions.
"""

# Default qualifying points (P1-P20)
DEFAULT_QUALIFYING_POINTS = {i: 52 - (i * 2) for i in range(1, 21)}

# Default race points (P1-P20)
DEFAULT_RACE_POINTS = {i: 103 - (i * 3) for i in range(1, 21)}

# Default sprint points (P1-P8 only)
DEFAULT_SPRINT_POINTS = {
    1: 8,
    2: 7,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 2,
    8: 1,
}

# Default improvement points for positions gained vs 8-race average
DEFAULT_IMPROVEMENT_POINTS = {
    1: 2,  # 1 position ahead
    2: 4,  # 2 positions ahead
    3: 6,  # 3 positions ahead
    4: 9,  # 4 positions ahead
    5: 12,  # 5 positions ahead
    6: 16,  # 6 positions ahead
    7: 20,  # 7 positions ahead
    8: 25,  # 8 positions ahead
    9: 30,  # 9+ positions ahead
}

# Default points for beating teammate by position margin
DEFAULT_TEAMMATE_POINTS = {
    3: 2,  # 1-3 positions ahead
    7: 5,  # 4-7 positions ahead
    12: 8,  # 8-12 positions ahead
    20: 12,  # >12 positions ahead
}

# Default completion stage thresholds (25%, 50%, 75%, 90%)
DEFAULT_COMPLETION_THRESHOLDS = [0.25, 0.5, 0.75, 0.9]

# Default values for other scoring parameters
DEFAULT_COMPLETION_STAGE_POINTS = 3.0
DEFAULT_OVERTAKE_MULTIPLIER = 3.0
DEFAULT_MINIMUM_POINTS = 650.0

# Default constructor points (per driver, will be summed)
DEFAULT_CONSTRUCTOR_QUALIFYING_POINTS = {i: 31 - i for i in range(1, 21)}
DEFAULT_CONSTRUCTOR_RACE_POINTS = {i: 62 - (i * 2) for i in range(1, 21)}

# Validation constants
MIN_POSITION = 1
MAX_POSITION = 20
MAX_SPRINT_POSITION = 8
MIN_POINTS = 0.0
MAX_POINTS = 1000.0  # Reasonable upper limit for any single component
MIN_MULTIPLIER = 1.0
MAX_MULTIPLIER = 10.0  # Reasonable upper limit for multipliers

# Schema for config JSON validation
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "qualifying_points": {
            "type": "object",
            "patternProperties": {
                "^[1-9][0-9]?$": {"type": "number", "minimum": MIN_POINTS}
            },
            "minProperties": 1,
            "maxProperties": MAX_POSITION,
        },
        "race_points": {
            "type": "object",
            "patternProperties": {
                "^[1-9][0-9]?$": {"type": "number", "minimum": MIN_POINTS}
            },
            "minProperties": 1,
            "maxProperties": MAX_POSITION,
        },
        "sprint_points": {
            "type": "object",
            "patternProperties": {"^[1-8]$": {"type": "number", "minimum": MIN_POINTS}},
            "minProperties": 1,
            "maxProperties": MAX_SPRINT_POSITION,
        },
        "improvement_points": {
            "type": "object",
            "patternProperties": {
                "^[1-9][0-9]?$": {"type": "number", "minimum": MIN_POINTS}
            },
        },
        "teammate_points": {
            "type": "object",
            "patternProperties": {
                "^[1-9][0-9]?$": {"type": "number", "minimum": MIN_POINTS}
            },
        },
        "completion_stage_points": {
            "type": "number",
            "minimum": MIN_POINTS,
            "maximum": MAX_POINTS,
        },
        "completion_thresholds": {
            "type": "array",
            "items": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "minItems": 1,
        },
        "overtake_multiplier": {
            "type": "number",
            "minimum": MIN_MULTIPLIER,
            "maximum": MAX_MULTIPLIER,
        },
        "minimum_points": {
            "type": "number",
            "minimum": MIN_POINTS,
            "maximum": MAX_POINTS,
        },
        "constructor_qualifying_points": {
            "type": "object",
            "patternProperties": {
                "^[1-9][0-9]?$": {"type": "number", "minimum": MIN_POINTS}
            },
            "minProperties": 1,
            "maxProperties": MAX_POSITION,
        },
        "constructor_race_points": {
            "type": "object",
            "patternProperties": {
                "^[1-9][0-9]?$": {"type": "number", "minimum": MIN_POINTS}
            },
            "minProperties": 1,
            "maxProperties": MAX_POSITION,
        },
    },
    "additionalProperties": False,
}
