"""
Example script for optimizing a GridRival F1 fantasy team for the Japanese Grand Prix.

This script demonstrates how to use the gridrival_ai library to optimize a fantasy team
composition for the Japanese Grand Prix.
"""

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.probabilities.distributions import RaceDistribution


def main():
    """Run the optimization for Japanese GP fantasy team."""
    # Step 1: Set up driver and constructor data
    driver_salaries = {
        "PIA": 29.8,  # Oscar Piastri
        "NOR": 29.4,  # Lando Norris
        "VER": 28.0,  # Max Verstappen
        "RUS": 27.5,  # George Russell
        "LEC": 24.4,  # Charles Leclerc
        "HAM": 22.2,  # Lewis Hamilton
        "ANT": 21.8,  # Andrea Kimi Antonelli
        "ALB": 18.5,  # Alex Albon
        "TSU": 15.6,  # Yuki Tsunoda
        "SAI": 15.1,  # Carlos Sainz
        "OCO": 14.1,  # Esteban Ocon
        "HAD": 13.7,  # Isack Hadjar
        "GAS": 13.5,  # Pierre Gasly
        "ALO": 12.6,  # Fernando Alonso
        "BEA": 12.6,  # Oliver Bearman
        "LAW": 9.7,   # Liam Lawson
        "HUL": 8.2,   # Nico Hulkenberg
        "STR": 7.7,   # Lance Stroll
        "COL": 6.6,   # Franco Colapinto (new driver replacing Doohan)
        "BOR": 4.0,   # Gabriel Bortoleto
    }

    constructor_salaries = {
        "MCL": 30.0,  # McLaren
        "MER": 25.7,  # Mercedes
        "FER": 23.1,  # Ferrari
        "RBR": 22.8,  # Red Bull Racing
        "WIL": 17.7,  # Williams
        "HAA": 14.1,  # Haas
        "RBU": 13.5,  # Racing Bulls
        "AST": 10.9,  # Aston Martin
        "ALP": 9.7,   # Alpine
        "SAU": 7.1,   # Kick Sauber
    }

    # Driver's 8-race rolling average finish positions
    rolling_averages = {
        "VER": 3,
        "TSU": 11,
        "RUS": 5,
        "ANT": 7,
        "LEC": 7,
        "HAM": 8,
        "NOR": 3,
        "PIA": 4,
        "ALO": 13,
        "STR": 15,
        "LAW": 14,
        "HAD": 14,
        "GAS": 14,
        "COL": 17,
        "SAI": 13,
        "ALB": 10,
        "OCO": 12,
        "BEA": 13,
        "HUL": 16,
        "BOR": 18,
    }

    # Combined locked sets for the optimizer
    locked_in = {"LEC", "TSU", "SAI", "BOR"}
    locked_out = {"PIA", "RBR"}

    # Betting odds for qualifying win
    qualifying_win_odds = {
        "PIA": 2.60,  # Oscar Piastri
        "NOR": 2.75,  # Lando Norris
        "VER": 4.75,  # Max Verstappen
        "RUS": 10.00,  # George Russell
        "ANT": 17.00,  # Kimi Antonelli
        "LEC": 21.00,  # Charles Leclerc
        "HAM": 41.00,  # Lewis Hamilton
        "ALB": 251.00,  # Alex Albon
        "SAI": 251.00,  # Carlos Sainz
        "TSU": 251.00,  # Yuki Tsunoda
        "BOR": 501.00,  # Gabriel Bortoleto
        "HAD": 501.00,  # Isack Hadjar
        "HUL": 501.00,  # Nico Hulkenberg
        "GAS": 501.00,  # Pierre Gasly
        "LAW": 1001.00,  # Liam Lawson
        "OCO": 1501.00,  # Esteban Ocon
        "COL": 2001.00,  # Franco Colapinto
        "ALO": 2501.00,  # Fernando Alonso
        "BEA": 2501.00,  # Oliver Bearman
        "STR": 3001.00,  # Lance Stroll
    }

    # Betting odds for race win
    race_winning_odds = {
        "PIA": 2.40,  # Oscar Piastri
        "NOR": 2.60,  # Lando Norris
        "VER": 6.00,  # Max Verstappen
        "RUS": 15.00,  # George Russell
        "LEC": 34.00,  # Charles Leclerc
        "ANT": 34.00,  # Kimi Antonelli
        "HAM": 55.00,  # Lewis Hamilton
        "ALB": 251.00,  # Alex Albon
        "SAI": 301.00,  # Carlos Sainz
        "TSU": 301.00,  # Yuki Tsunoda
        "GAS": 501.00,  # Pierre Gasly
        "HAD": 1001.00,  # Isack Hadjar
        "LAW": 1500.00,  # Liam Lawson
        "COL": 1501.00,  # Franco Colapinto
        "OCO": 1500.00,  # Esteban Ocon
        "ALO": 1500.00,  # Fernando Alonso
        "BEA": 1700.00,  # Oliver Bearman
        "STR": 2001.00,  # Lance Stroll
        "HUL": 2001.00,  # Nico Hulkenberg
        "BOR": 3001.00,  # Gabriel Bortoleto
    }

    # Betting odds for race podium (top 3) finish
    race_podium_odds = {
        "NOR": 1.22,  # Lando Norris
        "PIA": 1.22,  # Oscar Piastri
        "VER": 1.67,  # Max Verstappen
        "RUS": 2.65,  # George Russell
        "LEC": 4.10,  # Charles Leclerc
        "ANT": 4.50,  # Kimi Antonelli
        "HAM": 5.00,  # Lewis Hamilton
        "ALB": 67.00,  # Alex Albon
        "SAI": 67.00,  # Carlos Sainz
        "TSU": 70.00,  # Yuki Tsunoda
        "GAS": 101.00,  # Pierre Gasly
        "HAD": 151.00,  # Isack Hadjar
        "LAW": 251.00,  # Liam Lawson
        "OCO": 500.00,  # Esteban Ocon
        "BEA": 500.00,  # Oliver Bearman
        "ALO": 501.00,  # Fernando Alonso
        "COL": 501.00,  # Franco Colapinto
        "STR": 950.00,  # Lance Stroll
        "HUL": 1000.00,  # Nico Hulkenberg
        "BOR": 1501.00,  # Gabriel Bortoleto
    }

    # Betting odds for race top 6 finish
    race_top6_odds = {
        "NOR": 1.05,  # Lando Norris
        "PIA": 1.05,  # Oscar Piastri
        "VER": 1.08,  # Max Verstappen
        "LEC": 1.40,  # Charles Leclerc
        "RUS": 1.44,  # George Russell
        "ANT": 2.30,  # Kimi Antonelli
        "HAM": 2.75,  # Lewis Hamilton
        "TSU": 3.50,  # Yuki Tsunoda
        "ALB": 5.00,  # Alex Albon
        "SAI": 5.00,  # Carlos Sainz
        "GAS": 10.00,  # Pierre Gasly
        "HAD": 11.00,  # Isack Hadjar
        "LAW": 13.00,  # Liam Lawson
        "COL": 21.00,  # Franco Colapinto
        "OCO": 34.00,  # Esteban Ocon
        "ALO": 34.00,  # Fernando Alonso
        "STR": 41.00,  # Lance Stroll
        "BEA": 41.00,  # Oliver Bearman
        "HUL": 51.00,  # Nico Hulkenberg
        "BOR": 67.00,  # Gabriel Bortoleto
    }

    # Betting odds for race top 10 finish
    race_top10_odds = {
        "NOR": 1.04,  # Lando Norris
        "RUS": 1.05,  # George Russell
        "PIA": 1.04,  # Oscar Piastri
        "VER": 1.06,  # Max Verstappen
        "LEC": 1.08,  # Charles Leclerc
        "ANT": 1.08,  # Kimi Antonelli
        "HAM": 1.08,  # Lewis Hamilton
        "TSU": 1.22,  # Yuki Tsunoda
        "ALB": 1.50,  # Alex Albon
        "SAI": 1.53,  # Carlos Sainz
        "GAS": 2.20,  # Pierre Gasly
        "HAD": 2.85,  # Isack Hadjar
        "OCO": 6.00,  # Esteban Ocon
        "ALO": 6.00,  # Fernando Alonso
        "COL": 6.00,  # Franco Colapinto
        "BEA": 6.00,  # Oliver Bearman
        "LAW": 6.50,  # Liam Lawson
        "STR": 10.00,  # Lance Stroll
        "HUL": 12.00,  # Nico Hulkenberg
        "BOR": 23.00,  # Gabriel Bortoleto
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
            "race": {
                1: race_winning_odds,  # Win probabilities for main race
                3: race_podium_odds,  # Podium probabilities for main race
                6: race_top6_odds,  # Top 6 finish probabilities for main race
                10: race_top10_odds,  # Top 10 finish probabilities for main race
            },
            "qualifying": {
                1: qualifying_win_odds  # Qualifying pole position probabilities
            },
        },
        grid_method="cumulative",
    )

    # Step 4: Create and run the optimizer
    optimizer = TeamOptimizer(
        league_data=league_data,
        race_distribution=race_dist,
        driver_stats=rolling_averages,
        budget=111.98,  # Available budget
    )

    # Run the optimization for a standard race format
    result = optimizer.optimize(
        race_format="STANDARD",
        locked_in=locked_in,
        locked_out=locked_out,
    )

    # Step 5: Output the best solution
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
