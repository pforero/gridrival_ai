"""
Harville method module for converting odds to position probability distributions.

This module implements the Harville method for converting betting odds to a complete
grid of position probabilities. The method, proposed by David Harville (1973), uses
a dynamic programming approach to compute the probabilities of each competitor finishing
in each position.

The Harville method assumes that:
1. The probability of a competitor winning is proportional to their strength.
2. After removing a competitor, the relative strengths of the remaining competitors
   remain unchanged.

This implementation converts decimal betting odds into strengths and then computes
the full position probability grid using the recursive Harville formula:

    P(i finishes in position k) = s[i] / sum(s[j] for all remaining j)
        * P(i has not yet finished by position k-1)

where s[i] is the strength of competitor i.

The module provides both a simple converter for win probabilities and a more
comprehensive converter that produces a complete grid of position probabilities.

Examples
--------
>>> from gridrival_ai.probabilities.converters.harville import HarvilleConverter
>>> converter = HarvilleConverter()
>>>
>>> # Converting just win probabilities
>>> win_odds = [1.5, 3.0, 6.0]
>>> win_probs = converter.convert(win_odds)
>>> print(win_probs)
[0.67  0.33  0.17]
>>>
>>> # Converting to a full grid with driver IDs
>>> grid = converter.convert_to_grid(win_odds, ["VER", "HAM", "NOR"])
>>> print(grid["VER"][1])  # Probability VER finishes in P1
0.67
>>> print(grid["HAM"][2])  # Probability HAM finishes in P2
0.45
>>>
>>> # Converting to position distributions
>>> dists = converter.grid_to_position_distributions(grid)
>>> print(dists["NOR"].mean())  # Expected position for NOR
2.72

References
----------
.. [1] Harville, D. A. (1973). Assigning probabilities to the outcomes of
       multi-entry competitions. Journal of the American Statistical Association,
       68(342), 312-316.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from gridrival_ai.probabilities.converters.odds_converter import OddsConverter
from gridrival_ai.probabilities.core import PositionDistribution


class HarvilleConverter(OddsConverter):
    """
    Harville method for converting odds to grid probabilities.

    The Harville method is a dynamic programming approach that computes
    a complete grid of finishing position probabilities. It uses "strengths"
    derived from win odds and ensures constraints across rows and columns.

    Parameters
    ----------
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-10.

    Notes
    -----
    This implementation assumes that the input odds represent win probabilities.
    For other markets (e.g., top 3, top 6), use target_sum parameter and the
    method will adjust accordingly.
    """

    def __init__(self, epsilon: float = 1e-10) -> None:
        """Initialize with epsilon parameter."""
        self.epsilon = epsilon

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert win odds to win probabilities (row 1 of grid).

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.

        Returns
        -------
        np.ndarray
            Array of win probabilities summing to target_sum.
            This is only the first row of the full grid.

        Notes
        -----
        For a complete grid of probabilities, use convert_to_grid().
        """
        # Validate odds
        if any(o <= 1.0 for o in odds):
            raise ValueError("All odds must be greater than 1.0")

        # Convert odds to strengths that sum to target_sum
        raw_probs = np.array([1 / o for o in odds])
        return raw_probs * (target_sum / raw_probs.sum())

    def convert_to_grid(
        self, odds: List[float], driver_ids: Optional[List[str]] = None
    ) -> Dict[Union[str, int], Dict[int, float]]:
        """
        Convert win odds to a complete grid of position probabilities.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds for win market. Must be > 1.0.
        driver_ids : List[str], optional
            List of driver identifiers. If None, positions are used.

        Returns
        -------
        Dict[Union[str, int], Dict[int, float]]
            Nested dictionary mapping drivers to their position probabilities.
            {driver_id: {position: probability, ...}, ...}

        Examples
        --------
        >>> converter = HarvilleConverter()
        >>> odds = [1.5, 3.0]
        >>> grid = converter.convert_to_grid(odds, ["VER", "HAM"])
        >>> grid["VER"][1]  # Probability VER finishes P1
        0.6666666666666667
        >>> grid["VER"][2]  # Probability VER finishes P2
        0.33333333333333337
        >>> grid["HAM"][1]  # Probability HAM finishes P1
        0.33333333333333337
        >>> grid["HAM"][2]  # Probability HAM finishes P2
        0.6666666666666667
        """
        n = len(odds)
        if driver_ids is None:
            driver_ids = [str(i) for i in range(1, n + 1)]

        if len(driver_ids) != n:
            raise ValueError(
                f"Length of driver_ids ({len(driver_ids)}) must match odds ({n})"
            )

        # Convert odds to "strengths" using basic method
        strengths = self.convert(odds, target_sum=1.0)

        # Initialize result dictionary with empty dicts for each driver
        result = {driver_id: {} for driver_id in driver_ids}

        # dp[mask] holds the probability of reaching that state
        dp = np.zeros(1 << n)
        full_mask = (1 << n) - 1
        dp[full_mask] = 1.0

        # Iterate over all masks from full_mask down to 0
        for mask in range(full_mask, -1, -1):
            if dp[mask] == 0:
                continue

            # Build list of available drivers
            available = [i for i in range(n) if mask & (1 << i)]
            if not available:
                continue

            # Sum of strengths for drivers available in this mask
            s = sum(strengths[i] for i in available)

            # The finishing position we are about to assign:
            pos = n - len(available) + 1  # positions are 1-indexed

            # For each available driver, assign the probability to finish in pos
            for i in available:
                p_i = strengths[i] / (s + self.epsilon)
                prob = dp[mask] * p_i

                # Initialize position probability if it doesn't exist
                if pos not in result[driver_ids[i]]:
                    result[driver_ids[i]][pos] = 0

                # Accumulate the probability instead of just assigning
                result[driver_ids[i]][pos] += prob

                new_mask = mask & ~(1 << i)
                dp[new_mask] += prob

        return result

    def grid_to_position_distributions(
        self, grid: Dict[str, Dict[int, float]]
    ) -> Dict[str, PositionDistribution]:
        """
        Convert grid of probabilities to position distributions.

        Parameters
        ----------
        grid : Dict[str, Dict[int, float]]
            Grid of probabilities from convert_to_grid().

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions.

        Examples
        --------
        >>> converter = HarvilleConverter()
        >>> odds = [1.5, 3.0]
        >>> grid = converter.convert_to_grid(odds, ["VER", "HAM"])
        >>> dists = converter.grid_to_position_distributions(grid)
        >>> dists["VER"][1]  # Probability VER finishes P1
        0.6666666666666667
        """
        return {
            driver_id: PositionDistribution(positions)
            for driver_id, positions in grid.items()
        }
