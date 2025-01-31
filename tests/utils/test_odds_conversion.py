"""Basic tests for probability conversion methods."""

import numpy as np
import pytest

from gridrival_ai.utils.odds_conversion import (
    basic_method,
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
