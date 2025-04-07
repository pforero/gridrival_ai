"""
Example script for optimizing a GridRival F1 fantasy team for Australia race.

This script demonstrates how to use the gridrival_ai library to optimize a
fantasy team composition for the Australian Grand Prix.
"""

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.probabilities.distributions import RaceDistribution


def main():
    """Run the optimization for Australian GP fantasy team."""
    # Step 1: Set up driver and constructor data
    driver_salaries = {
        "VER": 30.0,  # Max Verstappen
        "LAW": 21.6,  # Liam Lawson
        "RUS": 23.0,  # George Russell
        "ANT": 20.2,  # Andrea Kimi Antonelli
        "LEC": 27.2,  # Charles Leclerc
        "HAM": 25.8,  # Lewis Hamilton
        "NOR": 28.6,  # Lando Norris
        "PIA": 24.4,  # Oscar Piastri
        "ALO": 18.8,  # Fernando Alonso
        "STR": 6.2,   # Lance Stroll
        "TSU": 17.4,  # Yuki Tsunoda
        "HAD": 9.0,   # Isack Hadjar
        "GAS": 16.0,  # Pierre Gasly
        "DOO": 4.8,   # Jack Doohan
        "SAI": 14.6,  # Carlos Sainz
        "ALB": 11.8,  # Alex Albon
        "OCO": 13.2,  # Esteban Ocon
        "BEA": 10.4,  # Oliver Bearman
        "HUL": 7.6,   # Nico Hulkenberg
        "BOR": 3.4,   # Gabriel Bortoleto
    }

    constructor_salaries = {
        "RBR": 24.4,  # Red Bull Racing
        "MER": 21.6,  # Mercedes
        "FER": 27.2,  # Ferrari
        "MCL": 30.0,  # McLaren
        "AST": 18.8,  # Aston Martin
        "ALP": 7.6,   # Alpine
        "WIL": 13.2,  # Williams
        "RBU": 10.4,  # Racing Bulls
        "SAU": 4.8,   # Kick Sauber
        "HAA": 16.0,  # Haas
    }

    # Driver's 8-race rolling average finish positions
    rolling_averages = {
        "VER": 1,
        "LAW": 7,
        "RUS": 6,
        "ANT": 8,
        "LEC": 3,
        "HAM": 4,
        "NOR": 2,
        "PIA": 5,
        "ALO": 9,
        "STR": 18,
        "TSU": 10,
        "HAD": 16,
        "GAS": 11,
        "DOO": 19,
        "SAI": 12,
        "ALB": 14,
        "OCO": 13,
        "BEA": 15,
        "HUL": 17,
        "BOR": 20,
    }

    # Betting odds for the driver to win the race (in decimal)
    winning_odds = {
        "VER": 5,
        "LAW": 70,
        "RUS": 13,
        "ANT": 51,
        "LEC": 5.5,
        "HAM": 7.75,
        "NOR": 3,
        "PIA": 9,
        "ALO": 251,
        "STR": 501,
        "TSU": 301,
        "HAD": 550,
        "GAS": 251,
        "DOO": 550,
        "SAI": 126,
        "ALB": 151,
        "OCO": 751,
        "BEA": 751,
        "HUL": 1001,
        "BOR": 2001,
    }

    # Step 2: Initialize the fantasy league data with the collected information
    league_data = FantasyLeagueData.from_dicts(
        driver_salaries=driver_salaries,
        constructor_salaries=constructor_salaries,
        rolling_averages=rolling_averages,
    )

    # Step 3: Create race distribution from betting odds
    race_dist = RaceDistribution.from_structured_odds(
        odds_structure={"race": {1: winning_odds}},
        grid_method="harville",
        normalization_method="sinkhorn",
        odds_method="basic",
    )

    # Step 5: Create and run the optimizer
    optimizer = TeamOptimizer(
        league_data=league_data,
        race_distribution=race_dist,
        driver_stats=rolling_averages,
    )

    # Run the optimization for a standard race format
    result = optimizer.optimize(race_format="STANDARD")

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
    else:
        print(f"No valid solution found: {result.error_message}")


if __name__ == "__main__":
    main()
