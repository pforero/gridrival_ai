"""Basic tests for probability conversion methods."""

import math

import numpy as np
import pytest

from gridrival_ai.utils.odds_conversion import (
    basic_method,
    harville_method,
    odds_ratio_method,
    power_method,
    shin_method,
)

TOLERANCE = 1e-4


@pytest.fixture
def simple_odds():
    """Simple odds scenario."""
    return [2.0, 4.0, 8.0]  # Clear favorite, midfield, longshot


@pytest.fixture
def f1_odds():
    """Realistic F1 qualifying odds."""
    return [
        1.50,  # VER
        1.50,  # PER (same odds as teammate)
        4.50,  # LEC
        8.00,  # SAI
        15.0,  # Multiple backmarkers
        15.0,  # with same odds
    ]


@pytest.fixture
def simple_driver_odds():
    """Two drivers with clear favorite."""
    return [
        {"driver_id": "VER", "1": 1.5},  # Clear favorite
        {"driver_id": "HAM", "1": 3.0},  # Underdog
    ]


@pytest.fixture
def f1_driver_odds():
    """Realistic F1 scenario with four drivers."""
    return [
        {"driver_id": "VER", "1": 1.5},  # Clear favorite
        {"driver_id": "HAM", "1": 3.0},  # Strong midfield
        {"driver_id": "LEC", "1": 4.5},  # Weaker midfield
        {"driver_id": "SAI", "1": 8.0},  # Backmarker
    ]


def test_methods_simple_case(simple_odds):
    """Test all methods with simple odds scenario."""
    methods = [
        basic_method,
        lambda x: odds_ratio_method(x)[0],
        lambda x: shin_method(x)[0],
        lambda x: power_method(x)[0],
    ]

    for method in methods:
        probs = method(simple_odds)

        # Basic probability properties
        assert abs(np.sum(probs) - 1.0) < TOLERANCE
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

        # Order preservation (higher odds -> lower probability)
        assert probs[0] > probs[1] > probs[2]


def test_methods_f1_case(f1_odds):
    """Test all methods with realistic F1 odds."""
    methods = [
        basic_method,
        lambda x: odds_ratio_method(x)[0],
        lambda x: shin_method(x)[0],
        lambda x: power_method(x)[0],
    ]

    for method in methods:
        probs = method(f1_odds)

        # Sum to 1
        assert abs(np.sum(probs) - 1.0) < TOLERANCE

        # Equal odds should give equal probabilities
        assert abs(probs[0] - probs[1]) < TOLERANCE  # teammates
        assert abs(probs[4] - probs[5]) < TOLERANCE  # backmarkers

        # Order preservation for different odds
        assert probs[0] > probs[2]  # 1.50 vs 4.50
        assert probs[2] > probs[3]  # 4.50 vs 8.00
        assert probs[3] > probs[4]  # 8.00 vs 15.0


def test_top3_finish(f1_odds):
    """Test methods with top-3 finish target probability."""
    target = 3.0
    methods = [
        lambda x: basic_method(x, target_probability=target),
        lambda x: odds_ratio_method(x, target_probability=target)[0],
        lambda x: shin_method(x, target_probability=target)[0],
        lambda x: power_method(x, target_probability=target)[0],
    ]

    for method in methods:
        probs = method(f1_odds)

        # Sum to target
        assert abs(np.sum(probs) - target) < TOLERANCE

        # Probabilities should be higher but maintain order
        assert probs[0] > probs[2]  # favorite still higher than midfield
        assert probs[2] > probs[4]  # midfield still higher than backmarker


def test_harville_simple_case():
    """Test Harville method with two drivers (win market)."""
    simple_driver_odds = [
        {"driver_id": "VER", "1": 1.5},
        {"driver_id": "HAM", "1": 3.0},
    ]
    result = harville_method(simple_driver_odds, target_market=1.0)

    # For each finishing position, the probabilities over drivers should sum to 1.
    for pos in [1, 2]:
        pos_sum = sum(result[d["driver_id"]][pos] for d in simple_driver_odds)
        assert math.isclose(
            pos_sum, 1.0, abs_tol=TOLERANCE
        ), f"Position {pos} sums to {pos_sum} instead of 1.0"
        for d in simple_driver_odds:
            assert result[d["driver_id"]][pos] >= 0, "Found negative probability"

    # For each driver, the probabilities over positions should sum to 1.
    for d in simple_driver_odds:
        driver_sum = sum(result[d["driver_id"]].values())
        assert math.isclose(
            driver_sum, 1.0, abs_tol=TOLERANCE
        ), f"Driver {d['driver_id']} sums to {driver_sum} instead of 1.0"

    # Verify that the stronger driver (VER) has a higher chance of finishing 1st.
    assert (
        result["VER"][1] > result["HAM"][1]
    ), "Relative strength not maintained for position 1."


def test_harville_relative_strength():
    """Test that the Harville method respects the ordering of driver strength."""
    f1_driver_odds = [
        {"driver_id": "VER", "1": 1.5},
        {"driver_id": "HAM", "1": 2.0},
        {"driver_id": "LEC", "1": 3.0},
        {"driver_id": "SAI", "1": 4.0},
    ]
    result = harville_method(f1_driver_odds, target_market=1.0)

    # Extract P1 probabilities.
    p1_probs = {d["driver_id"]: result[d["driver_id"]][1] for d in f1_driver_odds}
    # Lower odds (i.e. higher implied probability) should yield a higher chance of finishing 1st.
    assert (
        p1_probs["VER"] > p1_probs["HAM"] > p1_probs["LEC"] > p1_probs["SAI"]
    ), "P1 ordering is not as expected."

    # For each finishing position, the probabilities over drivers should sum to 1.
    n = len(f1_driver_odds)
    for pos in range(1, n + 1):
        pos_sum = sum(result[d["driver_id"]][pos] for d in f1_driver_odds)
        assert math.isclose(
            pos_sum, 1.0, abs_tol=TOLERANCE
        ), f"Position {pos} sums to {pos_sum} instead of 1.0"


def test_harville_equal_odds():
    """Test Harville method with two drivers having equal odds."""
    equal_odds = [
        {"driver_id": "VER", "1": 3.0},
        {"driver_id": "PER", "1": 3.0},
    ]
    result = harville_method(equal_odds, target_market=1.0)

    # Both drivers should have the same probability to finish 1st.
    assert math.isclose(
        result["VER"][1], result["PER"][1], abs_tol=TOLERANCE
    ), "Equal odds not resulting in equal P1."

    # For each driver, the sum over positions should equal 1.
    for driver in result:
        driver_sum = sum(result[driver].values())
        assert math.isclose(
            driver_sum, 1.0, abs_tol=TOLERANCE
        ), f"Driver {driver} has total probability {driver_sum} (expected 1.0)."
    # For two drivers there are exactly two finishing positions.
    for driver in result:
        assert (
            len(result[driver]) == 2
        ), f"Driver {driver} should have probabilities for 2 positions."


def test_harville_numerical_stability():
    """Test Harville method with very close odds to check numerical stability."""
    close_odds = [
        {"driver_id": "VER", "1": 2.0},
        {"driver_id": "PER", "1": 2.0001},  # extremely close odds
    ]
    result = harville_method(close_odds, target_market=1.0)

    for pos in [1, 2]:
        pos_sum = sum(result[d["driver_id"]][pos] for d in close_odds)
        assert math.isclose(
            pos_sum, 1.0, abs_tol=TOLERANCE
        ), f"Position {pos} sum {pos_sum} differs from 1.0."


def test_harville_market_six():
    """Test Harville method using a market other than win (e.g. finishing 6th or better)."""
    # (For simplicity we use two drivers; in practice F1 has 20 drivers.)
    market_six_odds = [
        {"driver_id": "VER", "6": 1.5},
        {"driver_id": "HAM", "6": 3.0},
    ]
    # Now target_market is 6.0 and we expect the relative ordering to be the same.
    result = harville_method(market_six_odds, target_market=6.0)

    # Even though the odds were for “top 6” (and strengths sum to 6), the DP still produces a grid (here only 2 positions).
    assert (
        result["VER"][1] > result["HAM"][1]
    ), "Relative strength not maintained for market 6."

    # Check that probabilities for each finishing position sum to 1.
    for pos in [1, 2]:
        pos_sum = sum(result[d["driver_id"]][pos] for d in market_six_odds)
        assert math.isclose(
            pos_sum, 1.0, abs_tol=TOLERANCE
        ), f"Market 6: Position {pos} sums to {pos_sum} (expected 1.0)."

    # Each driver’s grid probabilities sum to 1.
    for driver in result:
        driver_sum = sum(result[driver].values())
        assert math.isclose(
            driver_sum, 1.0, abs_tol=TOLERANCE
        ), f"Driver {driver} total probability {driver_sum} != 1.0."
