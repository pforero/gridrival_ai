"""Tests for the TeamOptimizer class."""

from unittest.mock import Mock

import pytest

from gridrival_ai.data.fantasy import FantasyLeagueData
from gridrival_ai.optimization.optimizer import TeamOptimizer
from gridrival_ai.optimization.types import ConstructorScoring, DriverScoring
from gridrival_ai.probabilities.distributions import (
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.scoring.types import RaceFormat


@pytest.fixture
def mock_points_calculator():
    """Create mock points calculator with controlled outputs."""
    calculator = Mock()

    # Define behavior for calculate_driver_points
    def calculate_driver_points(driver_id, race_format=None):
        # Points roughly proportional to driver quality for testing
        points_map = {
            "VER": {"qualifying": 50.0, "race": 50.0},  # Total 100
            "LAW": {"qualifying": 40.0, "race": 40.0},  # Total 80
            "HAM": {"qualifying": 45.0, "race": 45.0},  # Total 90
            "RUS": {"qualifying": 42.5, "race": 42.5},  # Total 85
            "LEC": {"qualifying": 47.5, "race": 47.5},  # Total 95
            "SAI": {"qualifying": 45.0, "race": 45.0},  # Total 90
            "NOR": {"qualifying": 46.0, "race": 46.0},  # Total 92
            "ANT": {"qualifying": 35.0, "race": 35.0},  # Total 70
            "ALO": {"qualifying": 41.0, "race": 41.0},  # Total 82
            "OCO": {"qualifying": 37.5, "race": 37.5},  # Total 75
            "BEA": {"qualifying": 10.0, "race": 15.0},  # Total 25
        }
        # Default for drivers not in the map
        default_points = {"qualifying": 30.0, "race": 30.0}  # Total 60
        base_points = points_map.get(driver_id, default_points)

        if race_format == RaceFormat.SPRINT:
            base_points["sprint"] = (
                base_points["qualifying"] * 0.2
            )  # 20% of qualifying points

        return base_points

    calculator.calculate_driver_points = Mock(side_effect=calculate_driver_points)

    # Define behavior for calculate_constructor_points
    def calculate_constructor_points(constructor_id, race_format=None):
        points_map = {
            "RBR": {"qualifying": 75.0, "race": 75.0},  # Total 150
            "MER": {"qualifying": 70.0, "race": 70.0},  # Total 140
            "FER": {"qualifying": 72.5, "race": 72.5},  # Total 145
            "MCL": {"qualifying": 67.5, "race": 67.5},  # Total 135
            "AST": {"qualifying": 60.0, "race": 60.0},  # Total 120
            "ALP": {"qualifying": 55.0, "race": 55.0},  # Total 110
            "HAA": {"qualifying": 25.0, "race": 25.0},  # Total 50
        }
        # Default for constructors not in the map
        default_points = {"qualifying": 50.0, "race": 50.0}  # Total 100

        return points_map.get(constructor_id, default_points)

    calculator.calculate_constructor_points = Mock(
        side_effect=calculate_constructor_points
    )

    return calculator


@pytest.fixture
def mock_race_distribution():
    """Create mock race distribution."""
    race_dist = Mock(spec=RaceDistribution)

    # Mock session distributions
    race_session = Mock(spec=SessionDistribution)
    qualifying_session = Mock(spec=SessionDistribution)
    sprint_session = Mock(spec=SessionDistribution)

    race_dist.race = race_session
    race_dist.qualifying = qualifying_session
    race_dist.sprint = sprint_session

    # Define behavior for get_driver_distribution method
    def get_driver_distribution(driver_id, session_type):
        # For simplicity, just return a mock position distribution
        # In real tests, you might want to configure more complex behavior
        return Mock()

    race_dist.get_driver_distribution = Mock(side_effect=get_driver_distribution)

    # Define behavior for get_session method
    def get_session(session_type):
        if session_type == "race":
            return race_session
        elif session_type == "qualifying":
            return qualifying_session
        elif session_type == "sprint":
            return sprint_session
        else:
            raise ValueError(f"Invalid session type: {session_type}")

    race_dist.get_session = Mock(side_effect=get_session)

    return race_dist


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
        locked_in=set(),
        locked_out=set(),
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
def optimizer(
    mock_points_calculator, mock_race_distribution, sample_league_data, driver_stats
):
    """Create TeamOptimizer with mocked dependencies."""
    return TeamOptimizer(
        league_data=sample_league_data,
        points_calculator=mock_points_calculator,
        race_distribution=mock_race_distribution,
        driver_stats=driver_stats,
    )


class TestTeamOptimizer:
    """Test suite for TeamOptimizer class."""

    def test_initialization(
        self,
        mock_points_calculator,
        mock_race_distribution,
        sample_league_data,
        driver_stats,
    ):
        """Test the initialization of TeamOptimizer."""
        optimizer = TeamOptimizer(
            league_data=sample_league_data,
            points_calculator=mock_points_calculator,
            race_distribution=mock_race_distribution,
            driver_stats=driver_stats,
            budget=95.0,
        )

        assert optimizer.league_data == sample_league_data
        assert optimizer.points_calculator == mock_points_calculator
        assert optimizer.race_distribution == mock_race_distribution
        assert optimizer.driver_stats == driver_stats
        assert optimizer.budget == 95.0

    def test_calculate_driver_scores(self, optimizer, sample_league_data):
        """Test calculation of driver scores."""
        # Call the private method directly
        driver_scores = optimizer._calculate_driver_scores(
            race_format=RaceFormat.STANDARD, locked_out=set()
        )

        # Check structure and content
        assert isinstance(driver_scores, dict)
        assert all(isinstance(score, DriverScoring) for score in driver_scores.values())

        # Check a specific driver
        ver_score = driver_scores.get("VER")
        assert ver_score is not None
        assert ver_score.regular_points == 100.0  # 50 + 50
        assert ver_score.salary == 33.0
        assert ver_score.can_be_talent is False  # Salary > 18.0

        # Check a talent-eligible driver
        ant_score = driver_scores.get("ANT")
        assert ant_score is not None
        assert ant_score.regular_points == 70.0  # 35 + 35
        assert ant_score.salary == 15.0
        assert ant_score.can_be_talent is True  # Salary < 18.0

    def test_calculate_constructor_scores(self, optimizer, sample_league_data):
        """Test calculation of constructor scores."""
        # Call the private method directly
        constructor_scores = optimizer._calculate_constructor_scores(
            race_format=RaceFormat.STANDARD, locked_out=set()
        )

        # Check structure and content
        assert isinstance(constructor_scores, dict)
        assert all(
            isinstance(score, ConstructorScoring)
            for score in constructor_scores.values()
        )

        # Check a specific constructor
        rbr_score = constructor_scores.get("RBR")
        assert rbr_score is not None
        assert rbr_score.points == 150.0  # 75 + 75
        assert rbr_score.salary == 22.0

        # Check points_dict structure
        assert "qualifying" in rbr_score.points_dict
        assert "race" in rbr_score.points_dict

    def test_basic_optimization(self, optimizer):
        """Test basic optimization with no constraints."""
        result = optimizer.optimize(race_format=RaceFormat.STANDARD)

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

    def test_locked_in_drivers(self, optimizer):
        """Test optimization with locked-in drivers."""
        # Lock in ALO and ANT
        result = optimizer.optimize(
            race_format=RaceFormat.STANDARD, locked_in={"ALO", "ANT"}
        )

        # Check locked-in drivers are included
        assert result.best_solution is not None
        assert "ALO" in result.best_solution.drivers
        assert "ANT" in result.best_solution.drivers

    def test_locked_out_drivers(self, optimizer):
        """Test optimization with locked-out drivers."""
        # Lock out top drivers
        result = optimizer.optimize(
            race_format=RaceFormat.STANDARD, locked_out={"VER", "LEC", "HAM"}
        )

        # Check locked-out drivers are excluded
        assert result.best_solution is not None
        assert "VER" not in result.best_solution.drivers
        assert "LEC" not in result.best_solution.drivers
        assert "HAM" not in result.best_solution.drivers

    def test_tight_budget_constraint(self, optimizer):
        """Test optimization with tight budget constraint."""
        # Create optimizer with reduced budget
        tight_optimizer = TeamOptimizer(
            league_data=optimizer.league_data,
            points_calculator=optimizer.points_calculator,
            race_distribution=optimizer.race_distribution,
            driver_stats=optimizer.driver_stats,
            budget=85.0,  # Reduced budget
        )

        result = tight_optimizer.optimize(race_format=RaceFormat.STANDARD)

        # Should still find a valid solution
        assert result.best_solution is not None
        # Should be within budget
        assert result.best_solution.total_cost <= 85.0

    def test_impossible_budget(self, optimizer):
        """Test optimization with impossible budget constraint."""
        # Create optimizer with extremely low budget
        impossible_optimizer = TeamOptimizer(
            league_data=optimizer.league_data,
            points_calculator=optimizer.points_calculator,
            race_distribution=optimizer.race_distribution,
            driver_stats=optimizer.driver_stats,
            budget=50.0,  # Too low to create a valid team
        )

        result = impossible_optimizer.optimize(race_format=RaceFormat.STANDARD)

        # Should not find a solution
        assert result.best_solution is None
        assert "No valid team composition" in result.error_message

    def test_locked_in_constructor(self, optimizer):
        """Test optimization with locked-in constructor."""
        # Lock in RBR constructor
        result = optimizer.optimize(race_format=RaceFormat.STANDARD, locked_in={"RBR"})

        # Check constructor is included
        assert result.best_solution is not None
        assert result.best_solution.constructor == "RBR"

    def test_sprint_race_format(self, optimizer):
        """Test optimization with sprint race format."""
        # Optimize for sprint race
        result = optimizer.optimize(race_format=RaceFormat.SPRINT)

        # Should find a valid solution
        assert result.best_solution is not None

        # Check that points were calculated correctly
        for driver_id in result.best_solution.drivers:
            # The points breakdown for this driver should include sprint points
            driver_points = result.best_solution.points_breakdown[driver_id]
            assert "sprint" in driver_points

    def test_no_talent_drivers(self, optimizer):
        """Test handling when no talent drivers are available."""
        # Create a scenario where no drivers are eligible for talent
        # by updating all driver salaries to be above the threshold
        high_salary_drivers = {
            driver_id: max(19.0, salary)  # Ensure all salaries are above threshold
            for driver_id, salary in optimizer.league_data.salaries.drivers.items()
        }

        league_data = FantasyLeagueData.from_dicts(
            driver_salaries=high_salary_drivers,
            constructor_salaries=optimizer.league_data.salaries.constructors,
            rolling_averages=optimizer.driver_stats,
        )

        no_talent_optimizer = TeamOptimizer(
            league_data=league_data,
            points_calculator=optimizer.points_calculator,
            race_distribution=optimizer.race_distribution,
            driver_stats=optimizer.driver_stats,
        )

        result = no_talent_optimizer.optimize(race_format=RaceFormat.STANDARD)

        # Should still find a valid solution
        assert result.best_solution is not None
        # But no talent driver should be selected
        assert result.best_solution.talent_driver == ""

    def test_talent_driver_selection(self, optimizer):
        """Test talent driver selection logic."""
        # Test that the best talent driver is chosen
        result = optimizer.optimize(race_format=RaceFormat.STANDARD)

        assert result.best_solution is not None
        talent_driver = result.best_solution.talent_driver

        # Should be eligible for talent
        assert optimizer.league_data.salaries.drivers[talent_driver] <= 18.0

        # Among eligible drivers, should select the one with highest points
        eligible_drivers = [
            d
            for d in result.best_solution.drivers
            if optimizer.league_data.salaries.drivers[d] <= 18.0
        ]

        if eligible_drivers:
            # Get points for all eligible drivers
            eligible_points = {
                d: sum(optimizer.points_calculator.calculate_driver_points(d).values())
                for d in eligible_drivers
            }

            # Talent driver should be the one with highest points
            best_driver = max(eligible_points, key=eligible_points.get)
            assert talent_driver == best_driver

    def test_too_many_locked_in_drivers(self, optimizer):
        """Test handling when too many drivers are locked in."""
        # Lock in 6 drivers (more than allowed)
        result = optimizer.optimize(
            race_format=RaceFormat.STANDARD,
            locked_in={"VER", "HAM", "LEC", "SAI", "NOR", "ANT"},
        )

        # Should not find a valid solution
        assert result.best_solution is None
        assert result.error_message is not None

    def test_points_breakdown(self, optimizer):
        """Test points breakdown in solution."""
        result = optimizer.optimize(race_format=RaceFormat.STANDARD)

        assert result.best_solution is not None

        # Check points breakdown structure
        assert result.best_solution.constructor in result.best_solution.points_breakdown

        for driver_id in result.best_solution.drivers:
            assert driver_id in result.best_solution.points_breakdown

            # Check component structure for drivers
            driver_points = result.best_solution.points_breakdown[driver_id]
            assert "qualifying" in driver_points
            assert "race" in driver_points

            # Talent driver should have doubled points
            if driver_id == result.best_solution.talent_driver:
                # The points should be doubled from the underlying value
                base_points = optimizer.points_calculator.calculate_driver_points(
                    driver_id
                )
                for component, value in driver_points.items():
                    if component in base_points:
                        assert value == base_points[component] * 2
