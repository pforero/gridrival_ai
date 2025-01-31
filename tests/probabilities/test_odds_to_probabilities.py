"""Tests for the odds to position probability converter."""

from typing import Dict

import numpy as np
import pytest

from gridrival_ai.probabilities.odds_to_probabilities import (
    OddsToPositionProbabilityConverter,
)

# Mark all tests in this module as slow by default
pytestmark = pytest.mark.slow


@pytest.fixture
def sample_converter():
    """Create sample OddsToPositionProbabilityConverter for testing with 20 drivers."""
    # Create base odds with implied probabilities within target ranges
    odds = []

    # Top driver (high chance of winning)
    odds.append(
        {
            "driver_id": "D01",
            "1": 2.5,  # 40% chance
            "3": 1.2,  # 83% chance
            "6": 1.05,  # 95% chance
            "10": 1.01,  # 99% chance
        }
    )

    # Second tier drivers (moderate chance)
    for i in range(2, 6):
        odds.append(
            {
                "driver_id": f"D{i:02d}",
                "1": 8.0,  # 12.5% each
                "3": 2.2,  # 45% each
                "6": 1.3,  # 77% each
                "10": 1.05,  # 95% each
            }
        )

    # Mid-field drivers
    for i in range(6, 16):
        odds.append(
            {
                "driver_id": f"D{i:02d}",
                "1": 50.0,  # 2% each
                "3": 15.0,  # 6.7% each
                "6": 4.0,  # 25% each
                "10": 1.5,  # 67% each
            }
        )

    # Back markers
    for i in range(16, 21):
        odds.append(
            {
                "driver_id": f"D{i:02d}",
                "1": 250.0,  # 0.4% each
                "3": 80.0,  # 1.25% each
                "6": 25.0,  # 4% each
                "10": 8.0,  # 12.5% each
            }
        )

    return OddsToPositionProbabilityConverter(odds)


@pytest.fixture
def minimal_converter():
    """Create minimal OddsToPositionProbabilityConverter with only winner market for 20
    drivers.
    """
    odds = []

    # Top driver
    odds.append({"driver_id": "D01", "1": 2.5})  # 40% chance

    # Second tier
    for i in range(2, 6):
        odds.append({"driver_id": f"D{i:02d}", "1": 8.0})  # 12.5% each

    # Mid-field
    for i in range(6, 16):
        odds.append({"driver_id": f"D{i:02d}", "1": 50.0})  # 2% each

    # Back markers
    for i in range(16, 21):
        odds.append({"driver_id": f"D{i:02d}", "1": 250.0})  # 0.4% each

    return OddsToPositionProbabilityConverter(odds)


@pytest.fixture
def small_converter():
    """Create small OddsToPositionProbabilityConverter for testing with 3 drivers.

    Creates a converter with clear probability preferences:
    - D1: Strong favorite (~67% win probability)
    - D2: Second favorite (~33% win probability)
    - D3: Underdog (~17% win probability)
    """
    return OddsToPositionProbabilityConverter(
        [
            {"driver_id": "D1", "1": 1.5},  # ~67% win probability
            {"driver_id": "D2", "1": 3.0},  # ~33% win probability
            {"driver_id": "D3", "1": 6.0},  # ~17% win probability
        ]
    )


def test_initialization(sample_converter, minimal_converter):
    """Test proper initialization of OddsToPositionProbabilityConverter.

    Tests:
    1. Drivers list contains all drivers from odds
    2. Number of drivers is correct
    3. Markets are correctly identified and sorted
    4. Initialization fails without winner odds
    """
    expected_drivers = {f"D{i:02d}" for i in range(1, 21)}

    # Test full markets case
    assert set(sample_converter.drivers) == expected_drivers
    assert sample_converter.num_drivers == 20
    assert sample_converter.markets == [1, 3, 6, 10]

    # Test minimal markets case
    assert set(minimal_converter.drivers) == expected_drivers
    assert minimal_converter.num_drivers == 20
    assert minimal_converter.markets == [1]

    # Test initialization fails without winner odds
    with pytest.raises(ValueError, match="Winner odds.*required"):
        odds = [
            {"driver_id": "D1", "3": 1.2},
            {"driver_id": "D2", "3": 1.5},
        ]
        OddsToPositionProbabilityConverter(odds)


def test_calculate_expected_positions(minimal_converter):
    """Test calculation of expected positions with and without ties.

    Tests:
    1. Simple case with distinct probabilities
    2. Case with ties between some drivers
    3. Case with multiple groups of ties
    4. Complex case with multiple groups
    """
    # Test 1: Simple case with distinct probabilities
    win_probs = np.array([0.5, 0.3, 0.2])
    expected = minimal_converter._calculate_expected_positions(win_probs)
    assert np.allclose(expected, [1.0, 2.0, 3.0])

    # Test 2: Case with ties
    win_probs = np.array([0.4, 0.3, 0.3])  # Tie for 2nd/3rd
    expected = minimal_converter._calculate_expected_positions(win_probs)
    assert np.allclose(expected, [1.0, 2.5, 2.5])

    # Test 3: Multiple groups with ties
    win_probs = np.array([0.25, 0.25, 0.25, 0.25])  # All tied
    expected = minimal_converter._calculate_expected_positions(win_probs)
    assert np.allclose(expected, [2.5, 2.5, 2.5, 2.5])

    # Test 4: Complex case with multiple groups
    win_probs = np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # Two pairs of ties
    expected = minimal_converter._calculate_expected_positions(win_probs)
    assert np.allclose(expected, [1.5, 1.5, 3.0, 4.5, 4.5])


def test_compute_adjusted_probabilities(sample_converter):
    """Test probability adjustment using power method."""
    adjusted = sample_converter._compute_adjusted_probabilities()

    # Test 1: Sum to target for all markets
    for market in [1, 3, 6, 10]:
        market_sum = adjusted[market].sum()
        assert np.isclose(
            market_sum, float(market)
        ), f"Market {market} probabilities sum to {market_sum}, expected {market}"

    # Test 2: Probabilities are valid
    for market in [1, 3, 6, 10]:
        assert all(
            0 <= p <= 1 for p in adjusted[market]
        ), f"Market {market} has invalid probabilities"


def test_initialize_prob_matrix(sample_converter, minimal_converter):
    """Test initial probability matrix respects market constraints.

    Tests:
    1. Basic matrix properties (dimensions and probability axioms)
    2. Winner probabilities match adjusted probabilities
    3. Uniform distribution between market positions
    4. Uniform distribution after last market
    """

    def check_basic_properties(matrix: np.ndarray, converter) -> None:
        """Verify matrix dimensions and probability axioms."""
        # Check dimensions and probability axioms
        assert matrix.shape == (20, 20), "Matrix should match number of drivers"
        assert np.allclose(
            matrix.sum(axis=1), 1.0
        ), "Driver probabilities should sum to 1"
        assert np.allclose(
            matrix.sum(axis=0), 1.0
        ), "Position probabilities should sum to 1"

        # Check winner probabilities
        assert np.allclose(
            matrix[:, 0], converter.adjusted_probs[1]
        ), "Winner probabilities should match adjusted probabilities"

    def check_uniform_distribution(
        matrix: np.ndarray, adjusted: Dict[int, np.ndarray], markets: list[int]
    ) -> None:
        """Verify uniform distribution between market positions.

        For each interval between consecutive markets, verifies that:
        1. Probabilities are uniformly distributed
        2. Values match the expected differences
        3. Remaining positions after last market are uniform
        """
        # Check intervals between markets
        for i in range(len(markets) - 1):
            start, end = markets[i], markets[i + 1]
            interval_size = end - start

            # Calculate expected uniform values for this interval
            prob_diff = (adjusted[end] - adjusted[start])[
                :, np.newaxis
            ]  # Add dimension for broadcasting
            expected_probs = prob_diff / interval_size

            # Check each position in the interval has the expected probabilities
            actual_probs = matrix[:, start:end]
            assert np.allclose(
                actual_probs, expected_probs, rtol=1e-9
            ), f"Non-uniform distribution between P{start+1}-P{end}"

        # Check uniform distribution after last market
        if markets[-1] < matrix.shape[1]:
            last_market = markets[-1]
            remaining_size = matrix.shape[1] - last_market

            # Calculate expected uniform values for remaining positions
            remaining_prob = (1.0 - adjusted[last_market])[
                :, np.newaxis
            ] / remaining_size
            actual_probs = matrix[:, last_market:]

            assert np.allclose(
                actual_probs, remaining_prob, rtol=1e-9
            ), "Non-uniform distribution after last market"

    # Test both converters
    for converter in [sample_converter, minimal_converter]:
        matrix = converter._initialize_prob_matrix()
        check_basic_properties(matrix, converter)
        check_uniform_distribution(matrix, converter.adjusted_probs, converter.markets)


def test_shape_penalty(small_converter):
    """Test shape penalty behavior for position distance and smoothness."""
    # Test 1: Distance from expected position
    matrix1 = np.array(
        [
            [0.5, 0.3, 0.2],  # D1: gradually decreasing from pos 1
            [0.3, 0.4, 0.3],  # D2: peaked at pos 2
            [0.2, 0.3, 0.5],  # D3: gradually increasing to pos 3
        ]
    )

    matrix2 = np.array(
        [
            [0.2, 0.3, 0.5],  # D1: opposite of expected (should be highest at pos 1)
            [0.5, 0.3, 0.2],  # D2: opposite of expected (should peak at pos 2)
            [0.5, 0.3, 0.2],  # D3: opposite of expected (should be highest at pos 3)
        ]
    )

    penalty1 = small_converter._shape_penalty(matrix1)
    penalty2 = small_converter._shape_penalty(matrix2)
    assert (
        penalty1 < penalty2
    ), "Penalty should increase with distance from expected position"

    # Test 2: Smoothness (testing only one driver)
    matrix3 = np.array(
        [
            [0.5, 0.3, 0.2],  # D1: smooth decrease
            [0.3, 0.4, 0.3],  # D2: keep constant
            [0.2, 0.3, 0.5],  # D3: keep constant
        ]
    )

    matrix4 = np.array(
        [
            [0.5, 0.1, 0.4],  # D1: jumpy distribution
            [0.3, 0.4, 0.3],  # D2: same as matrix3
            [0.2, 0.3, 0.5],  # D3: same as matrix3
        ]
    )

    penalty3 = small_converter._shape_penalty(matrix3)
    penalty4 = small_converter._shape_penalty(matrix4)
    assert penalty3 < penalty4, "Penalty should increase with less smooth distributions"


def test_regularized_objective(small_converter):
    """Test that regularized objective properly incorporates entropy.

    The objective should be higher (worse) for distributions with lower entropy
    when shape penalties are similar.
    """
    # Matrix with moderate concentration (medium entropy)
    matrix1 = np.array(
        [
            [0.5, 0.3, 0.2],  # D1: highest at pos 1
            [0.3, 0.4, 0.3],  # D2: highest at pos 2
            [0.2, 0.3, 0.5],  # D3: highest at pos 3
        ]
    )

    # Matrix with high concentration (low entropy)
    matrix2 = np.array(
        [
            [0.7, 0.2, 0.1],  # D1: highest at pos 1
            [0.2, 0.6, 0.2],  # D2: highest at pos 2
            [0.1, 0.2, 0.7],  # D3: highest at pos 3
        ]
    )

    # Both matrices follow expected positions but with different concentration
    # So shape penalties should be similar, but entropy should differ

    objective1 = small_converter._regularized_objective(matrix1.flatten())
    objective2 = small_converter._regularized_objective(matrix2.flatten())

    # Matrix2 has lower entropy, so should have higher (worse) objective
    assert objective1 < objective2, (
        "Regularized objective should penalize lower entropy "
        "when shape penalties are similar"
    )


def test_relaxed_constraints(small_converter):
    """Test that relaxed constraints are properly formatted and reasonable.

    Tests:
    1. Constraints are properly formatted for scipy.optimize
    2. Each market has a constraint
    3. Constraints have reasonable tolerance
    """
    constraints = small_converter._get_relaxed_constraints()

    # Test basic constraint properties
    assert len(constraints) == len(small_converter.markets)

    for constraint in constraints:
        # Check constraint format
        assert constraint["type"] == "eq"
        assert "fun" in constraint
        assert "tol" in constraint
        assert constraint["tol"] > 0

        # Test constraint function with a simple probability matrix
        test_matrix = np.ones((3, 3)) / 3  # Uniform distribution
        result = constraint["fun"](test_matrix.flatten())
        assert len(result) == small_converter.num_drivers


def test_fallback_solution(small_converter):
    """Test that fallback solution produces valid probabilities.

    Tests:
    1. Row sums equal 1 (each driver's probabilities sum to 1)
    2. Column sums equal 1 (each position's probabilities sum to 1)
    3. All probabilities are non-negative
    """
    # Create an invalid probability matrix
    small_converter.position_probs = np.array(
        [
            [0.8, 0.8, 0.8],  # Sums > 1
            [0.3, 0.3, 0.3],  # Sums < 1
            [0.5, 0.5, 0.5],  # Sums > 1
        ]
    )

    # Apply fallback solution
    small_converter._apply_fallback_solution()

    # Check results
    assert np.allclose(small_converter.position_probs.sum(axis=1), 1.0)
    assert np.allclose(small_converter.position_probs.sum(axis=0), 1.0)
    assert np.all(small_converter.position_probs >= 0)


def test_calculate_position_probabilities(small_converter):
    """Test the format and validity of final probability output.

    Tests:
    1. Output format is correct
    2. All drivers are included
    3. All positions are included
    4. Probabilities are valid
    """
    probs = small_converter.calculate_position_probabilities()

    # Check format and completeness
    assert len(probs) == small_converter.num_drivers
    assert all(driver in probs for driver in small_converter.drivers)

    for driver_probs in probs.values():
        # Check all positions are present
        assert len(driver_probs) == small_converter.num_drivers
        assert all(
            pos in driver_probs for pos in range(1, small_converter.num_drivers + 1)
        )

        # Check probability validity
        assert np.isclose(sum(driver_probs.values()), 1.0)
        assert all(0 <= p <= 1 for p in driver_probs.values())

        # Check type consistency
        assert all(isinstance(pos, int) for pos in driver_probs.keys())
        assert all(isinstance(p, float) for p in driver_probs.values())
