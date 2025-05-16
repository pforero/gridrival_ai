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
        "VER": 29.8,  # Max Verstappen
        "NOR": 28.8,  # Lando Norris
        "PIA": 26.1,  # Oscar Piastri
        "RUS": 24.9,  # George Russell
        "LEC": 24.0,  # Charles Leclerc
        "HAM": 21.7,  # Lewis Hamilton
        "ANT": 21.3,  # Andrea Kimi Antonelli
        "TSU": 17.2,  # Yuki Tsunoda
        "ALB": 16.7,  # Alex Albon
        "ALO": 15.1,  # Fernando Alonso
        "OCO": 13.1,  # Esteban Ocon
        "BEA": 13.8,  # Oliver Bearman
        "GAS": 13.7,  # Pierre Gasly
        "SAI": 12.4,  # Carlos Sainz
        "LAW": 11.2,  # Liam Lawson
        "HAD": 11.0,  # Isack Hadjar
        "HUL": 8.7,  # Nico Hulkenberg
        "STR": 8.2,  # Lance Stroll
        "DOO": 7.3,  # Franco Colapinto
        "BOR": 5.1,  # Gabriel Bortoleto
    }

    constructor_salaries = {
        "MCL": 30.0,  # McLaren
        "MER": 24.8,  # Mercedes
        "FER": 23.7,  # Ferrari
        "RBR": 22.8,  # Red Bull Racing
        "WIL": 16.4,  # Williams
        "HAA": 15.7,  # Haas
        "AST": 14.6,  # Aston Martin
        "RBU": 12.0,  # Racing Bulls
        "ALP": 8.7,  # Alpine
        "SAU": 7.0,  # Kick Sauber
    }

    # Driver's 8-race rolling average finish positions
    rolling_averages = {
        "VER": 2,
        "TSU": 12,
        "RUS": 6,
        "ANT": 7,
        "LEC": 6,
        "HAM": 7,
        "NOR": 2,
        "PIA": 5,
        "ALO": 12,
        "STR": 16,
        "LAW": 10,
        "HAD": 15,
        "GAS": 13,
        "DOO": 18,
        "SAI": 13,
        "ALB": 12,
        "OCO": 13,
        "BEA": 14,
        "HUL": 16,
        "BOR": 19,
    }

    # Combined locked sets for the optimizer
    locked_in = {"LEC", "HAM", "HAD", "PIA", "LAW"}
    locked_out = {"RBU"}

    # Betting odds for qualifying win
    qualifying_win_odds = {
        "NOR": 2.25,  # Lando Norris
        "PIA": 2.30,  # Oscar Piastri
        "VER": 8.00,  # Max Verstappen
        "RUS": 11.00,  # George Russell
        "LEC": 21.00,  # Charles Leclerc
        "HAM": 29.00,  # Lewis Hamilton
        "ANT": 101.00,  # Kimi Antonelli
        "TSU": 401.00,  # Yuki Tsunoda
        "ALB": 501.00,  # Alex Albon
        "SAI": 501.00,  # Carlos Sainz
        "HAD": 501.00,  # Isack Hadjar
        "LAW": 1251.00,  # Liam Lawson
        "ALO": 1501.00,  # Fernando Alonso
        "GAS": 1501.00,  # Pierre Gasly
        "OCO": 2001.00,  # Esteban Ocon
        "BEA": 2001.00,  # Oliver Bearman
        "HUL": 2501.00,  # Nico Hulkenberg
        "BOR": 2501.00,  # Gabriel Bortoleto
        "DOO": 3001.00,  # Franco Colapinto
        "STR": 2001.00,  # Lance Stroll
    }

    # Betting odds for race win
    race_winning_odds = {
        "NOR": 2.20,  # Lando Norris
        "PIA": 2.80,  # Oscar Piastri
        "VER": 8.00,  # Max Verstappen
        "RUS": 19.00,  # George Russell
        "LEC": 26.00,  # Charles Leclerc
        "HAM": 29.00,  # Lewis Hamilton
        "ANT": 100.00,  # Kimi Antonelli
        "TSU": 250.00,  # Yuki Tsunoda
        "HAD": 500.00,  # Isack Hadjar
        "ALB": 501.00,  # Alex Albon
        "SAI": 501.00,  # Carlos Sainz
        "LAW": 1001.00,  # Liam Lawson
        "OCO": 1501.00,  # Esteban Ocon
        "BEA": 1501.00,  # Oliver Bearman
        "ALO": 2001.00,  # Fernando Alonso
        "GAS": 2001.00,  # Pierre Gasly
        "HUL": 2501.00,  # Nico Hulkenberg
        "BOR": 3001.00,  # Gabriel Bortoleto
        "DOO": 3001.00,  # Franco Colapinto
        "STR": 3001.00,  # Lance Stroll
    }

    # Betting odds for race podium (top 3) finish
    race_podium_odds = {
        "NOR": 1.17,  # Lando Norris
        "PIA": 1.20,  # Oscar Piastri
        "VER": 1.80,  # Max Verstappen
        "RUS": 2.50,  # George Russell
        "LEC": 3.75,  # Charles Leclerc
        "HAM": 6.50,  # Lewis Hamilton
        "ANT": 11.00,  # Kimi Antonelli
        "HAD": 61.00,  # Isack Hadjar
        "TSU": 67.00,  # Yuki Tsunoda
        "ALB": 101.00,  # Alex Albon
        "SAI": 101.00,  # Carlos Sainz
        "LAW": 251.00,  # Liam Lawson
        "BEA": 301.00,  # Oliver Bearman
        "OCO": 351.00,  # Esteban Ocon
        "ALO": 501.00,  # Fernando Alonso
        "GAS": 501.00,  # Pierre Gasly
        "BOR": 1001.00,  # Gabriel Bortoleto
        "DOO": 1001.00,  # Franco Colapinto
        "STR": 1001.00,  # Lance Stroll
        "HUL": 1001.00,  # Nico Hulkenberg
    }

    # Betting odds for race top 6 finish
    race_top6_odds = {
        "NOR": 1.06,  # Lando Norris
        "PIA": 1.07,  # Oscar Piastri
        "VER": 1.08,  # Max Verstappen
        "RUS": 1.14,  # George Russell
        "LEC": 1.33,  # Charles Leclerc
        "HAM": 1.72,  # Lewis Hamilton
        "ANT": 2.00,  # Kimi Antonelli
        "HAD": 4.50,  # Isack Hadjar
        "TSU": 4.50,  # Yuki Tsunoda
        "ALB": 7.00,  # Alex Albon
        "SAI": 9.00,  # Carlos Sainz
        "BEA": 13.00,  # Oliver Bearman
        "LAW": 15.00,  # Liam Lawson
        "GAS": 23.00,  # Pierre Gasly
        "OCO": 23.00,  # Esteban Ocon
        "ALO": 41.00,  # Fernando Alonso
        "BOR": 101.00,  # Gabriel Bortoleto
        "DOO": 101.00,  # Franco Colapinto
        "HUL": 101.00,  # Nico Hulkenberg
        "STR": 151.00,  # Lance Stroll
    }

    # Betting odds for race top 10 finish
    race_top10_odds = {
        "NOR": 1.04,  # Lando Norris
        "VER": 1.05,  # Max Verstappen
        "PIA": 1.05,  # Oscar Piastri
        "LEC": 1.06,  # Charles Leclerc
        "RUS": 1.06,  # George Russell
        "HAM": 1.07,  # Lewis Hamilton
        "ANT": 1.11,  # Kimi Antonelli
        "ALB": 1.62,  # Alex Albon
        "HAD": 1.65,  # Isack Hadjar
        "TSU": 1.72,  # Yuki Tsunoda
        "SAI": 2.25,  # Carlos Sainz
        "LAW": 2.50,  # Liam Lawson
        "BEA": 3.25,  # Oliver Bearman
        "OCO": 3.50,  # Esteban Ocon
        "ALO": 3.75,  # Fernando Alonso
        "GAS": 5.50,  # Pierre Gasly
        "BOR": 11.00,  # Gabriel Bortoleto
        "DOO": 11.00,  # Franco Colapinto
        "HUL": 11.00,  # Nico Hulkenberg
        "STR": 17.00,  # Lance Stroll
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
        budget=111.108,  # Available budget
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
