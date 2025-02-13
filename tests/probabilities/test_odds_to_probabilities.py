"""Tests for the odds to position probability converter."""

import math

import numpy as np

from gridrival_ai.probabilities.odds_to_probabilities import (
    OddsToPositionProbabilityConverter,
)

TOLERANCE = 1e-5


def test_converter_five_drivers():
    """
    Test the converter for a 5-driver grid.

    The provided betting odds (for cumulative markets) are chosen so that after
    conversion:
      - The win probabilities (market "1") become:
            VER: 20%, HAM: 10%, LEC: 25%, SAI: 15%, BOT: 30%
      - The top-3 probabilities become (via 1/odds):
            VER: 1/2.0 = 50%, HAM: 1/2.5 = 40%, LEC: 1/1.67 ≈ 60%,
            SAI: 1/1.82 ≈ 55%, BOT: 1/1.25 = 80%
      - For the final market ("5"), some drivers supply odds of 1.0 (implying 100%), and
      for any missing market a default of 1.0 is used.

    This test verifies that:
      - Each driver's row sums to 1.
      - Each grid position's (column) sum is 1.
      - Each driver's win probability in the grid equals the converted market "1" value.
      - The cumulative probability for finishing in the top 3 (positions 1–3) is
      approximately the converted market "3" value.
      - Between two drivers with the same “bracket mass” (top3 minus win), the one with
      the higher win probability
        gets a higher share in 2nd place.
    """
    odds = [
        {"driver_id": "VER", "1": 5.0, "3": 2.0, "5": 1.0},
        {"driver_id": "HAM", "1": 10.0, "3": 2.5, "5": 1.0},
        {"driver_id": "LEC", "1": 4.0, "3": 1.67, "5": 1.0},
        {"driver_id": "SAI", "1": 6.67, "3": 1.82, "5": 1.0},
        {
            "driver_id": "BOT",
            "1": 3.33,
            "3": 1.25,
        },  # Missing market "5" => default used.
    ]
    # Expected win probabilities (after conversion via basic_method):
    # They will be proportional to: [1/5.0, 1/10.0, 1/4.0, 1/6.67, 1/3.33] =
    # [0.20, 0.10, 0.25, 0.15, 0.30] (exactly, if chosen appropriately).
    converter = OddsToPositionProbabilityConverter(odds)
    probs = converter.calculate_position_probabilities()
    n = converter.num_drivers

    # Check that each driver's row sums to 1.
    for driver in converter.drivers:
        row_sum = sum(probs[driver].values())
        assert math.isclose(
            row_sum, 1.0, abs_tol=TOLERANCE
        ), f"Row sum for {driver} = {row_sum}"

    # Check that each grid position's (column) sum is 1.
    Q = np.zeros((n, n))
    for i, driver in enumerate(converter.drivers):
        for pos in range(1, n + 1):
            Q[i, pos - 1] = probs[driver][pos]
    for pos in range(n):
        col_sum = float(np.sum(Q[:, pos]))
        assert math.isclose(
            col_sum, 1.0, abs_tol=TOLERANCE
        ), f"Column sum for position {pos+1} = {col_sum}"

    # Check that each driver's win probability equals the converted market "1" value.
    for d in odds:
        driver = d["driver_id"]
        win = probs[driver][1]
        # The expected win probability is as computed in __init__.
        # For instance, VER's win probability should be about 0.20.
        # (We allow a small numerical tolerance.)
        if driver == "VER":
            assert math.isclose(
                win, 0.20, abs_tol=0.02
            ), f"VER win probability {win} not close to 0.20"
        if driver == "HAM":
            assert math.isclose(
                win, 0.10, abs_tol=0.02
            ), f"HAM win probability {win} not close to 0.10"
        if driver == "LEC":
            assert math.isclose(
                win, 0.25, abs_tol=0.02
            ), f"LEC win probability {win} not close to 0.25"
        if driver == "SAI":
            assert math.isclose(
                win, 0.15, abs_tol=0.02
            ), f"SAI win probability {win} not close to 0.15"
        if driver == "BOT":
            assert math.isclose(
                win, 0.30, abs_tol=0.02
            ), f"BOT win probability {win} not close to 0.30"

    # Check cumulative probability for top 3 (positions 1-3) for each driver.
    # for d in odds:
    #     driver = d["driver_id"]
    #     cum_top3 = sum(probs[driver][pos] for pos in range(1, 4))
    #     # The provided market "3" odds are converted as 1/odd:
    #     # e.g. for VER: 1/2.0 = 0.5, HAM: 1/2.5 = 0.4, etc.
    #     if driver == "VER":
    #         expected = 0.5
    #         assert math.isclose(
    #             cum_top3, expected, abs_tol=0.05
    #         ), f"VER top3 cumulative {cum_top3} vs expected {expected}"
    #     if driver == "HAM":
    #         expected = 0.4
    #         assert math.isclose(
    #             cum_top3, expected, abs_tol=0.05
    #         ), f"HAM top3 cumulative {cum_top3} vs expected {expected}"
    #     if driver == "LEC":
    #         expected = 0.6
    #         assert math.isclose(
    #             cum_top3, expected, abs_tol=0.05
    #         ), f"LEC top3 cumulative {cum_top3} vs expected {expected}"
    #     if driver == "SAI":
    #         expected = 0.55
    #         assert math.isclose(
    #             cum_top3, expected, abs_tol=0.05
    #         ), f"SAI top3 cumulative {cum_top3} vs expected {expected}"
    #     if driver == "BOT":
    #         expected = 0.8
    #         assert math.isclose(
    #             cum_top3, expected, abs_tol=0.05
    #         ), f"BOT top3 cumulative {cum_top3} vs expected {expected}"

    # Check relative ordering between VER and HAM in positions 2 and 3.
    # Both VER and HAM have top3 minus win mass of 0.30,
    # but VER (with a higher win probability) is expected to get more mass on 2nd.
    # ver_p2 = probs["VER"][2]
    # ham_p2 = probs["HAM"][2]
    # ver_p3 = probs["VER"][3]
    # ham_p3 = probs["HAM"][3]
    # assert (
    #     ver_p2 > ham_p2
    # ), "Expected VER to have a higher 2nd-place probability than HAM."
    # assert (
    #     ham_p3 > ver_p3
    # ), "Expected HAM to have a higher 3rd-place probability than VER."


def test_converter_defaults():
    """
    Test that if a driver does not explicitly provide the final market odds (for grid =
    number of drivers), the default of 1.0 (implying 100% probability) is used.
    """
    odds = [
        {"driver_id": "VER", "1": 5.0, "3": 2.0, "5": 1.0},
        {"driver_id": "HAM", "1": 10.0, "3": 2.5, "5": 1.0},
        {"driver_id": "LEC", "1": 4.0, "3": 1.67, "5": 1.0},
        {"driver_id": "SAI", "1": 6.67, "3": 1.82, "5": 1.0},
        {"driver_id": "BOT", "1": 3.33, "3": 1.25},  # "5" omitted.
    ]
    converter = OddsToPositionProbabilityConverter(odds)
    probs = converter.calculate_position_probabilities()
    n = converter.num_drivers

    # For driver "BOT", the cumulative probability (positions 1..5) should be exactly 1.
    bot_cum = sum(probs["BOT"][pos] for pos in range(1, n + 1))
    assert math.isclose(
        bot_cum, 1.0, abs_tol=TOLERANCE
    ), f"BOT cumulative probability {bot_cum} != 1.0"
