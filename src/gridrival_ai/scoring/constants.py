"""
Default scoring values for F1 fantasy.

This module contains all constants needed for F1 fantasy scoring calculations
without validation logic, configuration loading, or other complexity.
"""

from typing import Literal

RaceFormat = Literal["STANDARD", "SPRINT"]


# Default qualifying points (P1-P20)
QUALIFYING_POINTS = {i: 52 - (i * 2) for i in range(1, 21)}

# Default race points (P1-P20)
RACE_POINTS = {i: 103 - (i * 3) for i in range(1, 21)}

# Default sprint points (P1-P20)
SPRINT_POINTS = {i: 21 - i for i in range(1, 21)}

# Default constructor qualifying points (P1-P20)
CONSTRUCTOR_QUALIFYING_POINTS = {i: 31 - i for i in range(1, 21)}

# Default constructor race points (P1-P20)
CONSTRUCTOR_RACE_POINTS = {i: 62 - (i * 2) for i in range(1, 21)}

# Default improvement points for positions gained vs 8-race average
IMPROVEMENT_POINTS = {
    1: 0,
    2: 2,
    3: 4,
    4: 6,
    5: 9,
    6: 12,
    7: 16,
    8: 20,
    9: 25,
    10: 30,
    11: 30,
    12: 30,
    13: 30,
    14: 30,
    15: 30,
    16: 30,
    17: 30,
    18: 30,
    19: 30,
    20: 30,
}

# Default points for beating teammate by position margin
TEAMMATE_POINTS = {
    3: 2,  # 1-3 positions ahead
    7: 5,  # 4-7 positions ahead
    12: 8,  # 8-12 positions ahead
    20: 12,  # >12 positions ahead
}

# Default completion stage thresholds (25%, 50%, 75%, 90%)
COMPLETION_THRESHOLDS = [0.25, 0.5, 0.75, 0.9]

# Default values for other scoring parameters
COMPLETION_STAGE_POINTS = 3.0
OVERTAKE_MULTIPLIER = 3.0
MINIMUM_POINTS = 650.0
