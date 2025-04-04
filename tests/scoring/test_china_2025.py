"""Tests for the ScoringCalculator with China 2025 race results."""

import pytest

from gridrival_ai.scoring.calculator import ScoringCalculator


@pytest.fixture
def calculator():
    """Create a ScoringCalculator with default config."""
    return ScoringCalculator()


@pytest.fixture
def mock_teams():
    """Return a mock of the current F1 teams and their drivers."""
    return {
        "RBR": ("VER", "LAW"),  # Red Bull Racing
        "MER": ("RUS", "ANT"),  # Mercedes
        "FER": ("LEC", "HAM"),  # Ferrari
        "MCL": ("NOR", "PIA"),  # McLaren
        "AST": ("ALO", "STR"),  # Aston Martin
        "ALP": ("GAS", "DOO"),  # Alpine
        "WIL": ("ALB", "SAI"),  # Williams
        "RBU": ("TSU", "HAD"),  # Racing Bulls
        "SAU": ("HUL", "BOR"),  # Kick Sauber
        "HAA": ("OCO", "BEA"),  # Haas
    }


@pytest.fixture
def driver_data():
    """Return a dictionary with all driver data for the race."""
    return {
        "PIA": {
            "qualifying_pos": 1,
            "race_pos": 1,
            "rolling_avg": 6,
            "completion_pct": 1.0,
            "sprint_pos": 2,
        },
        "NOR": {
            "qualifying_pos": 3,
            "race_pos": 2,
            "rolling_avg": 2,
            "completion_pct": 1.0,
            "sprint_pos": 8,
        },
        "RUS": {
            "qualifying_pos": 2,
            "race_pos": 3,
            "rolling_avg": 6,
            "completion_pct": 1.0,
            "sprint_pos": 4,
        },
        "VER": {
            "qualifying_pos": 4,
            "race_pos": 4,
            "rolling_avg": 2,
            "completion_pct": 1.0,
            "sprint_pos": 3,
        },
        "OCO": {
            "qualifying_pos": 11,
            "race_pos": 5,
            "rolling_avg": 13,
            "completion_pct": 1.0,
            "sprint_pos": 16,
        },
        "ANT": {
            "qualifying_pos": 8,
            "race_pos": 6,
            "rolling_avg": 8,
            "completion_pct": 1.0,
            "sprint_pos": 7,
        },
        "ALB": {
            "qualifying_pos": 10,
            "race_pos": 7,
            "rolling_avg": 13,
            "completion_pct": 1.0,
            "sprint_pos": 11,
        },
        "BEA": {
            "qualifying_pos": 17,
            "race_pos": 8,
            "rolling_avg": 15,
            "completion_pct": 1.0,
            "sprint_pos": 15,
        },
        "STR": {
            "qualifying_pos": 14,
            "race_pos": 9,
            "rolling_avg": 17,
            "completion_pct": 1.0,
            "sprint_pos": 9,
        },
        "SAI": {
            "qualifying_pos": 15,
            "race_pos": 10,
            "rolling_avg": 13,
            "completion_pct": 1.0,
            "sprint_pos": 17,
        },
        "HAD": {
            "qualifying_pos": 7,
            "race_pos": 11,
            "rolling_avg": 17,
            "completion_pct": 1.0,
            "sprint_pos": 13,
        },
        "LAW": {
            "qualifying_pos": 20,
            "race_pos": 12,
            "rolling_avg": 8,
            "completion_pct": 1.0,
            "sprint_pos": 14,
        },
        "DOO": {
            "qualifying_pos": 18,
            "race_pos": 13,
            "rolling_avg": 19,
            "completion_pct": 1.0,
            "sprint_pos": 20,
        },
        "BOR": {
            "qualifying_pos": 19,
            "race_pos": 14,
            "rolling_avg": 20,
            "completion_pct": 1.0,
            "sprint_pos": 18,
        },
        "HUL": {
            "qualifying_pos": 12,
            "race_pos": 15,
            "rolling_avg": 16,
            "completion_pct": 1.0,
            "sprint_pos": 19,
        },
        "TSU": {
            "qualifying_pos": 9,
            "race_pos": 16,
            "rolling_avg": 11,
            "completion_pct": 1.0,
            "sprint_pos": 6,
        },
        "ALO": {
            "qualifying_pos": 13,
            "race_pos": 17,
            "rolling_avg": 10,
            "completion_pct": 0.0,
            "sprint_pos": 10,
        },
        "LEC": {
            "qualifying_pos": 6,
            "race_pos": 18,
            "rolling_avg": 4,
            "completion_pct": 0.0,
            "sprint_pos": 5,
        },
        "HAM": {
            "qualifying_pos": 5,
            "race_pos": 19,
            "rolling_avg": 5,
            "completion_pct": 0.0,
            "sprint_pos": 1,
        },
        "GAS": {
            "qualifying_pos": 16,
            "race_pos": 20,
            "rolling_avg": 11,
            "completion_pct": 0.0,
            "sprint_pos": 12,
        },
    }


def test_driver_total_points(calculator, driver_data, mock_teams):
    """Test total points calculation for all drivers."""
    driver_points = {}
    for driver, data in driver_data.items():
        # Find teammate from mock teams
        teammate = None
        for team_drivers in mock_teams.values():
            if driver in team_drivers:
                teammate = next((d for d in team_drivers if d != driver), None)
                break

        points = calculator.calculate_driver_points(
            qualifying_pos=data["qualifying_pos"],
            race_pos=data["race_pos"],
            rolling_avg=data["rolling_avg"],
            teammate_pos=driver_data[teammate]["race_pos"],
            completion_pct=data["completion_pct"],
            sprint_pos=data["sprint_pos"],
            race_format="SPRINT",
        )
        driver_points[driver] = points.total

    # Expected total points for each driver
    expected_points = {
        "PIA": 192,  # P1 in race, P2 in sprint, P1 in qualifying
        "NOR": 171,  # P2 in race, P8 in sprint, P3 in qualifying
        "RUS": 177,  # P3 in race, P4 in sprint, P2 in qualifying
        "VER": 173,  # P4 in race, P3 in sprint, P4 in qualifying
        "OCO": 175,  # P5 in race, P16 in sprint, P11 in qualifying
        "ANT": 155,  # P6 in race, P7 in sprint, P8 in qualifying
        "ALB": 159,  # P7 in race, P11 in sprint, P10 in qualifying
        "BEA": 158,  # P8 in race, P15 in sprint, P17 in qualifying
        "STR": 167,  # P9 in race, P9 in sprint, P14 in qualifying
        "SAI": 130,  # P10 in race, P17 in sprint, P15 in qualifying
        "HAD": 145,  # P11 in race, P13 in sprint, P7 in qualifying
        "LAW": 122,  # P12 in race, P14 in sprint, P20 in qualifying
        "DOO": 125,  # P13 in race, P20 in sprint, P18 in qualifying
        "BOR": 119,  # P14 in race, P18 in sprint, P19 in qualifying
        "HUL": 100,  # P15 in race, P19 in sprint, P12 in qualifying
        "TSU": 116,  # P16 in race, P6 in sprint, P9 in qualifying
        "ALO": 89,  # DNF
        "LEC": 56 + 49 + 2,  # DNF
        "HAM": 62 + 46,  # DNF but P1 in sprint
        "GAS": 29 + 43,  # DNF
    }

    # Test all drivers
    for driver, expected in expected_points.items():
        assert driver_points[driver] == expected


def test_constructor_total_points(calculator, driver_data, mock_teams):
    """Test total points calculation for all constructors."""
    constructor_points = {}
    for constructor, drivers in mock_teams.items():
        points = calculator.calculate_constructor_points(
            driver1_qualifying=driver_data[drivers[0]]["qualifying_pos"],
            driver1_race=driver_data[drivers[0]]["race_pos"],
            driver2_qualifying=driver_data[drivers[1]]["qualifying_pos"],
            driver2_race=driver_data[drivers[1]]["race_pos"],
        )
        constructor_points[constructor] = sum(points.values())

    # Expected total points for each constructor
    expected_points = {
        "MCL": 176.0,  # McLaren (PIA + NOR)
        "MER": 158.0,  # Mercedes (RUS + HAM)
        "RBR": 130.0,  # Red Bull (VER)
        "ALP": 64.0 + 22.0,  # Alpine (OCO + GAS)
        "AST": 107.0,  # Aston Martin (ALO + STR)
        "FER": 51.0 + 26.0 + 24.0,  # Ferrari (LEC + SAI)
        "HAA": 132.0,  # Haas (HUL + BEA)
        "RBU": 116.0,  # Racing Bulls (TSU + LAW)
        "SAU": 97.0,  # Sauber (BOR + DOO)
        "WIL": 127.0,  # Williams (ALB + HAD)
    }

    # Test all constructors
    for constructor, expected in expected_points.items():
        assert constructor_points[constructor] == expected
