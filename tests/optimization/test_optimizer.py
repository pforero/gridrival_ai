"""Tests for the TeamOptimizer class."""

from unittest.mock import Mock, patch

import pytest

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.optimization.types import ConstructorScoring, DriverScoring
from gridrival_ai.probabilities.distributions import PositionDistribution


class TestJointDistribution:
    """Simple test implementation of JointDistribution for testing."""

    def __init__(self):
        """Initialize with a simple distribution."""
        self.distribution = {
            (1, 1): 0.1,  # (qual_pos, race_pos): probability
            (1, 2): 0.1,
            (2, 1): 0.2,
            (2, 2): 0.2,
            (3, 3): 0.4,
        }

    def items(self):
        """Return distribution items."""
        return self.distribution.items()


class TestRaceDistribution:
    """A simplified RaceDistribution for testing."""

    def __init__(self, distributions):
        """Initialize with a simple distribution."""
        self.distributions = distributions

    def get_driver_distribution(self, driver_id, session_type):
        """Return a driver's distribution for a session type."""
        return self.distributions.get(
            (driver_id, session_type), PositionDistribution({1: 1.0})
        )

    def get_qualifying_race_distribution(self, driver_id):
        """Return a joint distribution for qualifying and race."""
        return TestJointDistribution()

    def get_completion_probability(self, driver_id):
        """Return a driver's completion probability."""
        return 0.95


@pytest.fixture
def race_distribution():
    """Create a test implementation of RaceDistribution."""
    # Create a minimal distribution
    driver_ids = [
        "VER",
        "LAW",
        "HAM",
        "RUS",
        "LEC",
        "SAI",
        "NOR",
        "ANT",
        "ALO",
        "OCO",
        "BEA",
    ]
    session_types = ["qualifying", "race", "sprint"]

    # Create a dictionary of position distributions
    distributions = {}
    for driver_id in driver_ids:
        for session_type in session_types:
            distributions[(driver_id, session_type)] = PositionDistribution(
                {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.1, 5: 0.1}
            )

    return TestRaceDistribution(distributions)


@pytest.fixture
def sample_league_data():
    """Create sample league data with realistic values."""
    driver_salaries = {
        "VER": 33.0,  # Too expensive for talent
        "LAW": 19.0,
        "HAM": 24.0,
        "RUS": 21.0,
        "LEC": 28.0,
        "SAI": 22.0,
        "NOR": 18.0,  # Just at talent threshold
        "ANT": 15.0,  # Eligible for talent
        "ALO": 17.0,  # Eligible for talent
        "OCO": 14.0,  # Eligible for talent
        "BEA": 3.0,
    }

    constructor_salaries = {
        "RBR": 22.0,
        "MER": 20.0,
        "FER": 21.0,
        "MCL": 18.0,
        "AST": 16.0,
        "ALP": 15.0,
        "HAA": 3.0,
    }

    # Default empty constraints
    league_data = FantasyLeagueData.from_dicts(
        driver_salaries=driver_salaries,
        constructor_salaries=constructor_salaries,
        rolling_averages={d: 2.0 for d in driver_salaries},
    )

    return league_data


@pytest.fixture
def driver_stats():
    """Create sample driver statistics."""
    return {
        "VER": 1.5,
        "LAW": 3.8,
        "HAM": 2.1,
        "RUS": 3.2,
        "LEC": 2.5,
        "SAI": 3.5,
        "NOR": 3.0,
        "ANT": 5.2,
        "ALO": 4.4,
        "OCO": 6.8,
        "BEA": 20.0,
    }


@pytest.fixture
def optimizer(race_distribution, sample_league_data, driver_stats):
    """Create TeamOptimizer with dependencies."""
    return TeamOptimizer(
        league_data=sample_league_data,
        race_distribution=race_distribution,
        driver_stats=driver_stats,
    )


class TestTeamOptimizer:
    """Test suite for TeamOptimizer class."""

    def test_initialization(
        self,
        race_distribution,
        sample_league_data,
        driver_stats,
    ):
        """Test the initialization of TeamOptimizer."""
        optimizer = TeamOptimizer(
            league_data=sample_league_data,
            race_distribution=race_distribution,
            driver_stats=driver_stats,
            budget=95.0,
        )

        assert optimizer.league_data == sample_league_data
        assert optimizer.race_distribution == race_distribution
        assert optimizer.driver_stats == driver_stats
        assert optimizer.budget == 95.0
        assert hasattr(optimizer, "_scorer")  # Check that _scorer is initialized

    def test_calculate_driver_scores(self, optimizer, sample_league_data):
        """Test calculation of driver scores."""
        # Call the private method directly
        driver_scores = optimizer._calculate_driver_scores(race_format="STANDARD")

        # Check structure and content
        assert isinstance(driver_scores, dict)
        assert all(isinstance(score, DriverScoring) for score in driver_scores.values())

        # Check a specific driver
        ver_score = driver_scores.get("VER")
        assert ver_score is not None
        assert ver_score.salary == 33.0
        assert ver_score.can_be_talent is False  # Salary > 18.0

        # Check a talent-eligible driver
        ant_score = driver_scores.get("ANT")
        assert ant_score is not None
        assert ant_score.salary == 15.0
        assert ant_score.can_be_talent is True  # Salary < 18.0

        # Validate that driver scoring works
        assert ver_score.regular_points > 0
        assert ant_score.regular_points > 0

    @patch("gridrival_ai.optimization.optimizer.CONSTRUCTORS")
    def test_calculate_constructor_scores(
        self, mock_constructors, optimizer, sample_league_data
    ):
        """Test calculation of constructor scores using mocked CONSTRUCTORS."""
        # Mock the CONSTRUCTORS dictionary with our test data
        constructor = Mock()
        constructor.drivers = ("VER", "LAW")  # Sample driver pair for testing
        mock_constructors.get.return_value = constructor

        # Call the private method directly
        constructor_scores = optimizer._calculate_constructor_scores()

        # Check structure and content
        assert isinstance(constructor_scores, dict)
        assert all(
            isinstance(score, ConstructorScoring)
            for score in constructor_scores.values()
        )

        # Check that constructor scores are calculated
        for constructor_id, score in constructor_scores.items():
            if constructor_id in sample_league_data.salaries.constructors:
                assert score.points > 0  # Points should be positive
                assert (
                    score.salary
                    == sample_league_data.salaries.constructors[constructor_id]
                )
                assert "qualifying" in score.points_dict
                assert "race" in score.points_dict

    def test_basic_optimization(self, optimizer):
        """Test basic optimization with no constraints."""
        result = optimizer.optimize(race_format="STANDARD")

        # Should find a valid solution
        assert result.best_solution is not None
        assert len(result.best_solution.drivers) == 5
        assert (
            result.best_solution.constructor
            in optimizer.league_data.salaries.constructors
        )

        # Check budget constraint
        assert result.best_solution.total_cost <= 100.0

        # Check talent driver
        assert result.best_solution.talent_driver in result.best_solution.drivers
        talent_salary = optimizer.league_data.salaries.drivers[
            result.best_solution.talent_driver
        ]
        assert talent_salary <= 18.0  # Talent driver must have salary <= 18.0

    def test_locked_out(self, optimizer):
        """Test optimization with locked-out."""
        # Lock out top drivers
        result = optimizer.optimize(
            race_format="STANDARD", locked_out={"SAI", "LAW", "HAA"}
        )

        # Check locked-out drivers are excluded
        assert result.best_solution is not None
        assert "SAI" not in result.best_solution.drivers
        assert "LAW" not in result.best_solution.drivers
        assert "HAA" not in result.best_solution.constructor

    def test_locked_in(self, optimizer):
        """Test optimization with locked-in constructor."""
        # Lock in RBR constructor
        result = optimizer.optimize(race_format="STANDARD", locked_in={"RBR"})

        # Try to find a non-RBR solution
        non_rbr_solution = None
        for solution in result.all_solutions:
            if solution.constructor != "RBR":
                non_rbr_solution = solution
                break

        if non_rbr_solution:
            assert non_rbr_solution.constructor != "RBR"
            # Get the salaries of all elements in this solution
            rbr_salary = optimizer.league_data.salaries.constructors["RBR"]
            driver_salaries_sum = sum(
                optimizer.league_data.salaries.drivers[driver]
                for driver in non_rbr_solution.drivers
            )
            constructor_salary = optimizer.league_data.salaries.constructors[
                non_rbr_solution.constructor
            ]

            # The raw sum without penalty
            raw_sum = driver_salaries_sum + constructor_salary

            # Check that the actual cost is higher than raw sum due to penalty
            # Penalty should be 3% of the locked-in constructor's salary
            expected_penalty = rbr_salary * 0.03

            # The solution's total cost should include this penalty
            assert non_rbr_solution.total_cost > raw_sum
            assert (
                abs(non_rbr_solution.total_cost - (raw_sum + expected_penalty)) < 0.01
            )

    def test_points_breakdown(self, optimizer):
        """Test points breakdown in solution."""
        result = optimizer.optimize(race_format="STANDARD")

        assert result.best_solution is not None

        # Check points breakdown structure
        assert result.best_solution.constructor in result.best_solution.points_breakdown

        for driver_id in result.best_solution.drivers:
            assert driver_id in result.best_solution.points_breakdown

            # Check component structure for drivers
            driver_points = result.best_solution.points_breakdown[driver_id]
            assert "qualifying" in driver_points
            assert "race" in driver_points

            # Talent driver should have doubled points compared to regular drivers
            if driver_id == result.best_solution.talent_driver:
                # Find a non-talent driver for comparison
                non_talent_drivers = [
                    d for d in result.best_solution.drivers if d != driver_id
                ]
                if non_talent_drivers:
                    non_talent_id = non_talent_drivers[0]
                    non_talent_points = result.best_solution.points_breakdown[
                        non_talent_id
                    ]

                    # Verify that components are doubled (roughly)
                    # We're checking based on the real calculator's values
                    talent_total = sum(driver_points.values())
                    non_talent_total = sum(non_talent_points.values())
                    assert (
                        talent_total > non_talent_total
                    )  # Talent should get more points

    def test_top_n_solutions(self, optimizer):
        """Test getting top N solutions."""
        result = optimizer.optimize(race_format="STANDARD")

        # Get top 5 solutions
        top_5 = result.top_n(5)

        # Should return at most 5 solutions
        assert len(top_5) <= 5

        # Solutions should be sorted by expected points (highest first)
        for i in range(len(top_5) - 1):
            assert top_5[i].expected_points >= top_5[i + 1].expected_points

    def test_remaining_budget(self, optimizer):
        """Test remaining budget calculation."""
        result = optimizer.optimize(race_format="STANDARD")

        # Calculate expected remaining budget
        expected_remaining = optimizer.budget - result.best_solution.total_cost

        # Should match the property value
        assert result.remaining_budget == expected_remaining
