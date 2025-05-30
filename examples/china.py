"""
Example script for optimizing a GridRival F1 fantasy team for the Chinese Grand Prix.

This script demonstrates how to use the gridrival_ai library to optimize a
fantasy team composition for the Chinese Grand Prix which includes a sprint race.
"""

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.probabilities.distributions import RaceDistribution


def main():
    """Run the optimization for Chinese GP fantasy team."""
    # Step 1: Set up driver and constructor data
    driver_salaries = {
        "VER": 29.1,  # Max Verstappen
        "NOR": 28.5,  # Lando Norris
        "LEC": 25.3,  # Charles Leclerc
        "HAM": 23.8,  # Lewis Hamilton
        "PIA": 23.6,  # Oscar Piastri
        "RUS": 23.0,  # George Russell
        "ANT": 21.2,  # Andrea Kimi Antonelli
        "LAW": 19.6,  # Liam Lawson
        "TSU": 17.4,  # Yuki Tsunoda
        "ALO": 16.8,  # Fernando Alonso
        "GAS": 16.0,  # Pierre Gasly
        "ALB": 13.8,  # Alex Albon
        "OCO": 13.1,  # Esteban Ocon
        "SAI": 12.6,  # Carlos Sainz
        "BEA": 10.5,  # Oliver Bearman
        "HUL": 9.6,   # Nico Hulkenberg
        "STR": 8.2,   # Lance Stroll
        "HAD": 7.0,   # Isack Hadjar
        "BOR": 4.4,   # Gabriel Bortoleto
        "DOO": 4.2,   # Jack Doohan
    }

    constructor_salaries = {
        "MCL": 30.0,  # McLaren
        "FER": 26.5,  # Ferrari
        "RBR": 23.7,  # Red Bull Racing
        "MER": 23.0,  # Mercedes
        "AST": 18.1,  # Aston Martin
        "WIL": 14.6,  # Williams
        "HAA": 13.9,  # Haas
        "RBU": 9.0,   # Racing Bulls
        "ALP": 8.3,   # Alpine
        "SAU": 6.9,   # Kick Sauber
    }

    # Driver's 8-race rolling average finish positions
    rolling_averages = {
        "VER": 2,
        "NOR": 2,
        "LEC": 4,
        "HAM": 5,
        "PIA": 6,
        "RUS": 6,
        "ANT": 8,
        "LAW": 8,
        "TSU": 11,
        "ALO": 10,
        "GAS": 11,
        "ALB": 13,
        "OCO": 13,
        "SAI": 13,
        "BEA": 15,
        "HUL": 16,
        "STR": 17,
        "HAD": 17,
        "BOR": 20,
        "DOO": 19,
    }

    # Locked-in and locked-out drivers
    locked_in_drivers = {"ALB", "SAI", "STR", "HAD"}
    locked_out_drivers = {"LEC"}

    # Locked-in and locked-out constructors
    locked_in_constructors = set()
    locked_out_constructors = {"MCL"}

    # Combined locked sets for the optimizer
    locked_in = locked_in_drivers.union(locked_in_constructors)
    locked_out = locked_out_drivers.union(locked_out_constructors)

    # Betting odds for various race outcomes
    # Sprint race win odds
    sprint_winning_odds = {
        "NOR": 2.00,  # Lando Norris
        "PIA": 4.33,  # Oscar Piastri
        "VER": 5.00,  # Max Verstappen
        "RUS": 17.00,  # George Russell
        "LEC": 19.00,  # Charles Leclerc
        "HAM": 23.00,  # Lewis Hamilton
        "ANT": 51.00,  # Andrea Kimi Antonelli
        "ALB": 151.00,  # Alex Albon
        "SAI": 151.00,  # Carlos Sainz
        "LAW": 151.00,  # Liam Lawson
        "GAS": 151.00,  # Pierre Gasly
        "TSU": 151.00,  # Yuki Tsunoda
        "ALO": 326.00,  # Fernando Alonso
        "STR": 326.00,  # Lance Stroll
        "HUL": 326.00,  # Nico Hulkenberg
        "OCO": 501.00,  # Esteban Ocon
        "BOR": 501.00,  # Gabriel Bortoleto
        "HAD": 501.00,  # Isack Hadjar
        "DOO": 501.00,  # Jack Doohan
        "BEA": 501.00,  # Oliver Bearman
    }

    # Sprint podium (top 3) odds
    sprint_podium_odds = {
        "NOR": 1.28,  # Lando Norris
        "PIA": 1.40,  # Oscar Piastri
        "VER": 1.44,  # Max Verstappen
        "LEC": 2.50,  # Charles Leclerc
        "RUS": 2.75,  # George Russell
        "HAM": 5.00,  # Lewis Hamilton
        "ANT": 8.00,  # Andrea Kimi Antonelli
        "ALB": 26.00,  # Alex Albon
        "SAI": 26.00,  # Carlos Sainz
        "TSU": 34.00,  # Yuki Tsunoda
        "LAW": 51.00,  # Liam Lawson
        "GAS": 51.00,  # Pierre Gasly
        "ALO": 126.00,  # Fernando Alonso
        "STR": 126.00,  # Lance Stroll
        "OCO": 176.00,  # Esteban Ocon
        "HAD": 176.00,  # Isack Hadjar
        "HUL": 176.00,  # Nico Hulkenberg
        "BOR": 326.00,  # Gabriel Bortoleto
        "DOO": 326.00,  # Jack Doohan
        "BEA": 326.00,  # Oliver Bearman
    }

    # Sprint top 6 finish odds
    sprint_top6_odds = {
        "NOR": 1.05,  # Lando Norris
        "VER": 1.05,  # Max Verstappen
        "PIA": 1.08,  # Oscar Piastri
        "LEC": 1.08,  # Charles Leclerc
        "RUS": 1.22,  # George Russell
        "HAM": 1.40,  # Lewis Hamilton
        "ANT": 1.73,  # Andrea Kimi Antonelli
        "ALB": 3.25,  # Alex Albon
        "SAI": 3.25,  # Carlos Sainz
        "LAW": 9.00,  # Liam Lawson
        "GAS": 9.00,  # Pierre Gasly
        "TSU": 9.00,  # Yuki Tsunoda
        "ALO": 15.00,  # Fernando Alonso
        "STR": 34.00,  # Lance Stroll
        "HUL": 34.00,  # Nico Hulkenberg
        "OCO": 101.00,  # Esteban Ocon
        "BOR": 151.00,  # Gabriel Bortoleto
        "HAD": 151.00,  # Isack Hadjar
        "DOO": 151.00,  # Jack Doohan
        "BEA": 151.00,  # Oliver Bearman
    }

    # Sprint top 8 finish odds
    sprint_top8_odds = {
        "VER": 1.02,  # Max Verstappen
        "NOR": 1.02,  # Lando Norris
        "PIA": 1.03,  # Oscar Piastri
        "LEC": 1.05,  # Charles Leclerc
        "RUS": 1.10,  # George Russell
        "HAM": 1.20,  # Lewis Hamilton
        "ANT": 1.25,  # Andrea Kimi Antonelli
        "ALB": 1.61,  # Alex Albon
        "SAI": 1.67,  # Carlos Sainz
        "TSU": 3.50,  # Yuki Tsunoda
        "LAW": 3.50,  # Liam Lawson
        "GAS": 3.75,  # Pierre Gasly
        "ALO": 7.00,  # Fernando Alonso
        "STR": 11.00,  # Lance Stroll
        "HUL": 11.00,  # Nico Hulkenberg
        "DOO": 15.00,  # Jack Doohan
        "HAD": 21.00,  # Isack Hadjar
        "BOR": 29.00,  # Gabriel Bortoleto
        "OCO": 34.00,  # Esteban Ocon
        "BEA": 51.00,  # Oliver Bearman
    }

    # Qualifying win odds
    qualifying_win_odds = {
        "NOR": 1.33,  # Lando Norris
        "PIA": 1.45,  # Oscar Piastri
        "VER": 1.60,  # Max Verstappen
        "LEC": 2.75,  # Charles Leclerc
        "RUS": 3.25,  # George Russell
        "HAM": 5.50,  # Lewis Hamilton
        "ANT": 6.50,  # Kimi Antonelli
        "SAI": 21.00,  # Carlos Sainz
        "ALB": 26.00,  # Alex Albon
        "TSU": 29.00,  # Yuki Tsunoda
        "LAW": 35.00,  # Liam Lawson
        "GAS": 41.00,  # Pierre Gasly
        "ALO": 67.00,  # Fernando Alonso
        "STR": 81.00,  # Lance Stroll
        "HAD": 101.00,  # Isack Hadjar
        "HUL": 201.00,  # Nico Hulkenberg
        "DOO": 251.00,  # Jack Doohan
        "BOR": 301.00,  # Gabriel Bortoleto
        "OCO": 401.00,  # Esteban Ocon
        "BEA": 501.00,  # Oliver Bearman
    }

    # Race win odds
    race_winning_odds = {
        "NOR": 2.50,  # Lando Norris
        "PIA": 4.50,  # Oscar Piastri
        "VER": 5.50,  # Max Verstappen
        "LEC": 17.00,  # Charles Leclerc
        "RUS": 17.00,  # George Russell
        "HAM": 23.00,  # Lewis Hamilton
        "ANT": 41.00,  # Andrea Kimi Antonelli
        "SAI": 101.00,  # Carlos Sainz
        "ALB": 151.00,  # Alex Albon
        "LAW": 151.00,  # Liam Lawson
        "TSU": 201.00,  # Yuki Tsunoda
        "ALO": 251.00,  # Fernando Alonso
        "GAS": 251.00,  # Pierre Gasly
        "STR": 501.00,  # Lance Stroll
        "HAD": 501.00,  # Isack Hadjar
        "DOO": 751.00,  # Jack Doohan
        "HUL": 1001.00,  # Nico Hulkenberg
        "BOR": 1501.00,  # Gabriel Bortoleto
        "OCO": 2001.00,  # Esteban Ocon
        "BEA": 2501.00,  # Oliver Bearman
    }

    # Race podium (top 3) odds
    race_podium_odds = {
        "NOR": 1.33,  # Lando Norris
        "PIA": 1.45,  # Oscar Piastri
        "VER": 1.60,  # Max Verstappen
        "LEC": 2.75,  # Charles Leclerc
        "RUS": 3.25,  # George Russell
        "HAM": 5.50,  # Lewis Hamilton
        "ANT": 6.50,  # Kimi Antonelli
        "SAI": 21.00,  # Carlos Sainz
        "ALB": 26.00,  # Alex Albon
        "TSU": 29.00,  # Yuki Tsunoda
        "LAW": 35.00,  # Liam Lawson
        "GAS": 41.00,  # Pierre Gasly
        "ALO": 67.00,  # Fernando Alonso
        "STR": 81.00,  # Lance Stroll
        "HAD": 101.00,  # Isack Hadjar
        "HUL": 201.00,  # Nico Hulkenberg
        "DOO": 251.00,  # Jack Doohan
        "BOR": 301.00,  # Gabriel Bortoleto
        "OCO": 401.00,  # Esteban Ocon
        "BEA": 501.00,  # Oliver Bearman
    }

    # Race top 6 finish odds
    race_top6_odds = {
        "NOR": 1.20,  # Lando Norris
        "PIA": 1.22,  # Oscar Piastri
        "LEC": 1.25,  # Charles Leclerc
        "VER": 1.30,  # Max Verstappen
        "RUS": 1.55,  # George Russell
        "HAM": 2.25,  # Lewis Hamilton
        "SAI": 2.75,  # Carlos Sainz
        "ALB": 3.00,  # Alex Albon
        "ANT": 3.00,  # Kimi Antonelli
        "GAS": 8.00,  # Pierre Gasly
        "TSU": 8.00,  # Yuki Tsunoda
        "LAW": 10.00,  # Liam Lawson
        "ALO": 17.00,  # Fernando Alonso
        "STR": 26.00,  # Lance Stroll
        "HUL": 26.00,  # Nico Hulkenberg
        "OCO": 51.00,  # Esteban Ocon
        "BOR": 67.00,  # Gabriel Bortoleto
        "HAD": 67.00,  # Isack Hadjar
        "DOO": 67.00,  # Jack Doohan
        "BEA": 67.00,  # Oliver Bearman
    }

    # Race top 10 finish odds
    race_top10_odds = {
        "NOR": 1.10,  # Lando Norris
        "PIA": 1.10,  # Oscar Piastri
        "LEC": 1.11,  # Charles Leclerc
        "VER": 1.11,  # Max Verstappen
        "RUS": 1.12,  # George Russell
        "HAM": 1.26,  # Lewis Hamilton
        "SAI": 1.36,  # Carlos Sainz
        "ALB": 1.52,  # Alex Albon
        "ANT": 1.60,  # Kimi Antonelli
        "TSU": 1.80,  # Yuki Tsunoda
        "LAW": 2.50,  # Liam Lawson
        "GAS": 2.50,  # Pierre Gasly
        "ALO": 3.40,  # Fernando Alonso
        "STR": 5.00,  # Lance Stroll
        "HUL": 5.50,  # Nico Hulkenberg
        "DOO": 6.50,  # Jack Doohan
        "HAD": 8.00,  # Isack Hadjar
        "BOR": 9.00,  # Gabriel Bortoleto
        "OCO": 10.00,  # Esteban Ocon
        "BEA": 15.00,  # Oliver Bearman
    }

    # Step 2: Initialize the fantasy league data with the collected information
    league_data = FantasyLeagueData.from_dicts(
        driver_salaries=driver_salaries,
        constructor_salaries=constructor_salaries,
        rolling_averages=rolling_averages,
    )

    # Step 3: Create race distribution from all betting odds
    race_dist = RaceDistribution.from_structured_odds(
        odds_structure={
            "sprint": {
                1: sprint_winning_odds,  # Win probabilities for sprint race
                3: sprint_podium_odds,   # Podium (top 3) probabilities for sprint race
                6: sprint_top6_odds,     # Top 6 finish probabilities for sprint race
                8: sprint_top8_odds,     # Top 8 finish probabilities for sprint race
            },
            "race": {
                1: race_winning_odds,    # Win probabilities for main race
                3: race_podium_odds,     # Podium probabilities for main race
                6: race_top6_odds,       # Top 6 finish probabilities for main race
                10: race_top10_odds,     # Top 10 finish probabilities for main race
            },
            "qualifying": {
                1: qualifying_win_odds    # Qualifying pole position probabilities
            },
        },
        grid_method="cumulative",
    )

    # Step 5: Create and run the optimizer
    optimizer = TeamOptimizer(
        league_data=league_data,
        race_distribution=race_dist,
        driver_stats=rolling_averages,
        budget=98.1,  # Updated budget is £98.1M
    )

    # Run the optimization for a sprint race format
    result = optimizer.optimize(
        race_format="SPRINT",
        locked_in=locked_in,
        locked_out=locked_out,
    )

    # Step 6: Output the best solution
    if result.best_solution:
        print("\nOPTIMAL TEAM FOUND:")
        print("-----------------")
        print(f"Drivers: {', '.join(sorted(result.best_solution.drivers))}")
        print(f"Constructor: {result.best_solution.constructor}")
        print(f"Talent Driver: {result.best_solution.talent_driver}")
        print(f"Total Cost: £{result.best_solution.total_cost:.2f}M")
        print(f"Expected Points: {result.best_solution.expected_points:.2f}")
        print(f"Remaining Budget: £{result.remaining_budget:.2f}M")

        # Print points breakdown by driver/constructor
        print("\nPOINTS BREAKDOWN:")
        print("-----------------")
        for element_id, points_dict in result.best_solution.points_breakdown.items():
            if isinstance(points_dict, dict):
                total_element_points = sum(points_dict.values())
                print(f"{element_id}: {total_element_points:.2f} points")
            else:
                print(f"{element_id}: {points_dict:.2f} points")
    else:
        print(f"No valid solution found: {result.error_message}")


if __name__ == "__main__":
    main()
