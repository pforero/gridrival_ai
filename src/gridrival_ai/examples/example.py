from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.points.calculator import PointsCalculator
from gridrival_ai.probabilities.factory import DistributionFactory
from gridrival_ai.probabilities.registry import DistributionRegistry
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat

# Driver salaries (in millions)
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
    "STR": 06.2,  # Lance Stroll
    "TSU": 17.4,  # Yuki Tsunoda
    "HAD": 09.0,  # Isack Hadjar
    "GAS": 16.0,  # Pierre Gasly
    "DOO": 04.8,  # Jack Doohan
    "SAI": 14.6,  # Carlos Sainz
    "ALB": 11.8,  # Alex Albon
    "OCO": 13.2,  # Esteban Ocon
    "BEA": 10.4,  # Oliver Bearman
    "HUL": 07.6,  # Nico Hulkenberg
    "BOR": 03.4,  # Gabriel Bortoleto
}

# Constructor salaries (in millions)
constructor_salaries = {
    "RBR": 24.4,  # Red Bull Racing
    "MER": 21.6,  # Mercedes
    "FER": 27.2,  # Ferrari
    "MCL": 30.0,  # McLaren
    "AST": 18.8,  # Aston Martin
    "ALP": 07.6,  # Alpine
    "WIL": 13.2,  # Williams
    "RBU": 10.4,  # Racing Bulls
    "SAU": 04.8,  # Kick Sauber
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


def main():
    # Step 1: Set up the fantasy league data
    league_data = FantasyLeagueData.from_dicts(
        driver_salaries=driver_salaries,
        constructor_salaries=constructor_salaries,
        rolling_averages=rolling_averages,
    )

    # Step 2: Create distribution registry and populate with race probabilities
    # First create an empty registry
    registry = DistributionRegistry()

    # Then use the factory to populate the registry with distributions
    # based on the winning odds
    DistributionFactory.register_structured_odds(
        registry=registry,
        odds_structure={"race": {1: winning_odds}},
        method="basic",  # Using basic odds conversion method
    )

    # Step 3: Set up scoring calculator
    scorer = ScoringCalculator()

    # Step 4: Create points calculator
    points_calculator = PointsCalculator(
        scorer=scorer, probability_registry=registry, driver_stats=rolling_averages
    )

    # Step 5: Create and run the optimizer
    optimizer = TeamOptimizer(
        league_data=league_data,
        points_calculator=points_calculator,
        probability_registry=registry,
        driver_stats=rolling_averages,
    )

    result = optimizer.optimize(race_format=RaceFormat.STANDARD)

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
