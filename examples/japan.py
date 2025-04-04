from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.points.calculator import PointsCalculator
from gridrival_ai.probabilities.distributions import RaceDistribution
from gridrival_ai.scoring.calculator import ScoringCalculator
from gridrival_ai.scoring.types import RaceFormat

# Driver salaries (in millions)
driver_salaries = {
    "VER": 28.8,  # Max Verstappen
    "NOR": 28.0,  # Lando Norris
    "PIA": 25.6,  # Oscar Piastri
    "RUS": 25.0,  # George Russell
    "LEC": 23.3,  # Charles Leclerc
    "HAM": 21.8,  # Lewis Hamilton
    "ANT": 20.8,  # Andrea Kimi Antonelli
    "TSU": 18.1,  # Yuki Tsunoda
    "ALB": 15.8,  # Alex Albon
    "OCO": 15.1,  # Esteban Ocon
    "ALO": 14.8,  # Fernando Alonso
    "GAS": 14.0,  # Pierre Gasly
    "SAI": 13.4,  # Carlos Sainz
    "LAW": 12.9,  # Liam Lawson
    "BEA": 12.5,  # Oliver Bearman
    "STR": 10.2,  # Lance Stroll
    "HUL": 9.1,  # Nico Hulkenberg
    "HAD": 9.0,  # Isack Hadjar
    "DOO": 6.2,  # Jack Doohan
    "BOR": 6.0,  # Gabriel Bortoleto
}

# Constructor salaries (in millions)
constructor_salaries = {
    "MCL": 30.0,  # McLaren
    "MER": 24.0,  # Mercedes
    "FER": 23.5,  # Ferrari
    "RBR": 23.2,  # Red Bull Racing
    "AST": 16.9,  # Aston Martin
    "HAA": 16.5,  # Haas
    "WIL": 15.6,  # Williams
    "RBU": 10.7,  # Racing Bulls
    "ALP": 8.2,  # Alpine
    "SAU": 7.7,  # Kick Sauber
}

# Driver's 8-race rolling average finish positions
rolling_averages = {
    "VER": 2,
    "TSU": 9,
    "RUS": 6,
    "ANT": 8,
    "LEC": 6,
    "HAM": 7,
    "NOR": 2,
    "PIA": 5,
    "ALO": 11,
    "STR": 16,
    "LAW": 12,
    "HAD": 16,
    "GAS": 13,
    "DOO": 19,
    "SAI": 13,
    "ALB": 12,
    "OCO": 12,
    "BEA": 14,
    "HUL": 16,
    "BOR": 19,
}

# Locked-in and locked-out drivers from screenshots
locked_in_drivers = {
    "ALB",
    "SAI",
    "STR",
    "PIA",
}  # Drivers with red circle with line icon
locked_out_drivers = {"RUS"}  # Driver with lock icon

# Locked-in and locked-out constructors from screenshots
locked_in_constructors = set()  # No constructors locked in
locked_out_constructors = {"WIL"}  # McLaren has lock icon

# Combined locked sets for the optimizer
locked_in = locked_in_drivers.union(locked_in_constructors)
locked_out = locked_out_drivers.union(locked_out_constructors)

# Betting odds for qualifying win (in decimal) - dark blue values from image
qualifying_win_odds = {
    "NOR": 1.80,  # Lando Norris
    "PIA": 3.25,  # Oscar Piastri
    "RUS": 13.00,  # George Russell
    "VER": 13.00,  # Max Verstappen
    "LEC": 15.00,  # Charles Leclerc
    "HAM": 17.00,  # Lewis Hamilton
    "TSU": 67.00,  # Yuki Tsunoda
    "HAD": 126.00,  # Isack Hadjar
    "ANT": 126.00,  # Kimi Antonelli
    "LAW": 251.00,  # Liam Lawson
    "ALB": 301.00,  # Alex Albon
    "SAI": 301.00,  # Carlos Sainz
    "ALO": 501.00,  # Fernando Alonso
    "STR": 1501.00,  # Lance Stroll
    "GAS": 1501.00,  # Pierre Gasly
    "OCO": 2001.00,  # Esteban Ocon
    "HUL": 2001.00,  # Nico Hulkenberg
    "BOR": 2501.00,  # Gabriel Bortoleto
    "DOO": 3001.00,  # Jack Doohan
    "BEA": 3001.00,  # Oliver Bearman
}


# Betting odds for race winning (in decimal)
race_winning_odds = {
    "NOR": 2.25,  # Lando Norris
    "PIA": 3.00,  # Oscar Piastri
    "VER": 9.00,  # Max Verstappen
    "RUS": 12.00,  # George Russell
    "LEC": 15.00,  # Charles Leclerc
    "HAM": 15.00,  # Lewis Hamilton
    "TSU": 41.00,  # Yuki Tsunoda
    "HAD": 101.00,  # Isack Hadjar
    "ANT": 101.00,  # Kimi Antonelli
    "LAW": 151.00,  # Liam Lawson
    "ALO": 251.00,  # Fernando Alonso
    "ALB": 301.00,  # Alex Albon
    "SAI": 301.00,  # Carlos Sainz
    "STR": 501.00,  # Lance Stroll
    "GAS": 501.00,  # Pierre Gasly
    "HUL": 1001.00,  # Nico Hulkenberg
    "BOR": 2001.00,  # Gabriel Bortoleto
    "OCO": 2501.00,  # Esteban Ocon
    "DOO": 3001.00,  # Jack Doohan
    "BEA": 3001.00,  # Oliver Bearman
}

# Betting odds for race podium (top 3) finish (in decimal)
race_podium_odds = {
    "NOR": 1.30,  # Lando Norris
    "PIA": 1.40,  # Oscar Piastri
    "RUS": 2.40,  # George Russell
    "VER": 2.65,  # Max Verstappen
    "LEC": 3.50,  # Charles Leclerc
    "HAM": 3.50,  # Lewis Hamilton
    "TSU": 9.00,  # Yuki Tsunoda
    "HAD": 21.00,  # Isack Hadjar
    "ANT": 21.00,  # Kimi Antonelli
    "LAW": 26.00,  # Liam Lawson
    "ALB": 70.00,  # Alex Albon
    "SAI": 70.00,  # Carlos Sainz
    "ALO": 81.00,  # Fernando Alonso
    "STR": 251.00,  # Lance Stroll
    "GAS": 251.00,  # Pierre Gasly
    "HUL": 301.00,  # Nico Hulkenberg
    "OCO": 501.00,  # Esteban Ocon
    "BOR": 751.00,  # Gabriel Bortoleto
    "BEA": 751.00,  # Oliver Bearman
    "DOO": 1001.00,  # Jack Doohan
}

# Betting odds for race top 6 finish (in decimal)
race_top6_odds = {
    "NOR": 1.05,  # Lando Norris
    "PIA": 1.08,  # Oscar Piastri
    "VER": 1.13,  # Max Verstappen
    "RUS": 1.17,  # George Russell
    "LEC": 1.29,  # Charles Leclerc
    "HAM": 1.33,  # Lewis Hamilton
    "TSU": 2.00,  # Yuki Tsunoda
    "HAD": 3.50,  # Isack Hadjar
    "ANT": 4.00,  # Kimi Antonelli
    "LAW": 6.00,  # Liam Lawson
    "SAI": 11.00,  # Carlos Sainz
    "ALB": 12.00,  # Alex Albon
    "ALO": 13.00,  # Fernando Alonso
    "GAS": 26.00,  # Pierre Gasly
    "STR": 34.00,  # Lance Stroll
    "HUL": 67.00,  # Nico Hulkenberg
    "OCO": 101.00,  # Esteban Ocon
    "BOR": 101.00,  # Gabriel Bortoleto
    "DOO": 151.00,  # Jack Doohan
    "BEA": 151.00,  # Oliver Bearman
}

# Betting odds for race top 10 finish (in decimal)
race_top10_odds = {
    "NOR": 1.06,  # Lando Norris
    "VER": 1.06,  # Max Verstappen
    "PIA": 1.06,  # Oscar Piastri
    "RUS": 1.07,  # George Russell
    "HAM": 1.07,  # Lewis Hamilton
    "LEC": 1.08,  # Charles Leclerc
    "TSU": 1.28,  # Yuki Tsunoda
    "ANT": 1.36,  # Kimi Antonelli
    "HAD": 1.67,  # Isack Hadjar
    "LAW": 1.85,  # Liam Lawson
    "ALB": 2.10,  # Alex Albon
    "ALO": 2.10,  # Fernando Alonso
    "SAI": 2.25,  # Carlos Sainz
    "GAS": 4.00,  # Pierre Gasly
    "STR": 5.00,  # Lance Stroll
    "HUL": 7.00,  # Nico Hulkenberg
    "OCO": 11.00,  # Esteban Ocon
    "BOR": 11.00,  # Gabriel Bortoleto
    "BEA": 11.00,  # Oliver Bearman
    "DOO": 17.00,  # Jack Doohan
}


def main():
    # Step 1: Set up the fantasy league data
    league_data = FantasyLeagueData.from_dicts(
        driver_salaries=driver_salaries,
        constructor_salaries=constructor_salaries,
        rolling_averages=rolling_averages,
        locked_in=locked_in,
        locked_out=locked_out,
    )

    # Step 2: Create distribution registry and populate with race probabilities
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

    # Step 3: Set up scoring calculator
    scorer = ScoringCalculator()

    # Step 4: Create points calculator
    points_calculator = PointsCalculator(
        scorer=scorer, race_distribution=race_dist, driver_stats=rolling_averages
    )

    # Step 5: Create and run the optimizer
    optimizer = TeamOptimizer(
        league_data=league_data,
        points_calculator=points_calculator,
        race_distribution=race_dist,
        driver_stats=rolling_averages,
        budget=109.59,
    )

    # Run the optimization for a sprint race format
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
