"""
Harville grid creator implementation.

This module implements the Harville method for creating position
probability distributions from win odds.
"""

from typing import Dict, List, Union

import numpy as np

from gridrival_ai.probabilities.core import PositionDistribution
from gridrival_ai.probabilities.grid_creators.base import GridCreator
from gridrival_ai.probabilities.odds_structure import OddsStructure


class HarvilleGridCreator(GridCreator):
    """
    Grid creator using the Harville method.

    The Harville method creates position distributions using only win odds.
    It assumes that the probability of a driver winning is proportional to
    their "strength", and after removing a driver, the relative strengths
    of the remaining drivers remain unchanged.

    This implementation uses dynamic programming to efficiently compute the
    probabilities of each driver finishing in each position.

    References
    ----------
    .. [1] Harville, D. A. (1973). Assigning probabilities to the outcomes of
           multi-entry competitions. Journal of the American Statistical Association,
           68(342), 312-316.
    """

    def create_position_distributions(
        self, odds_input: Union[OddsStructure, Dict], session: str = "race", **kwargs
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions using the Harville method.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
            OddsStructure instance or raw odds dictionary
        session : str, optional
            Session to use for odds, by default "race"
        **kwargs
            Additional parameters (not used)

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        # Ensure input is an OddsStructure
        odds_structure = self._ensure_odds_structure(odds_input)

        # Get win odds and driver IDs
        win_probs, driver_ids = self.convert_win_odds_to_probabilities(
            odds_structure, session
        )

        # Apply Harville dynamic programming algorithm
        grid = self._harville_dp(win_probs, driver_ids)

        # Convert grid to PositionDistribution objects
        distributions = {}
        for driver_id, positions in grid.items():
            try:
                distributions[driver_id] = PositionDistribution(positions)
            except Exception:
                # If validation fails, try without validation and normalize
                distributions[driver_id] = PositionDistribution(
                    positions, _validate=False
                ).normalize()

        # Ensure grid constraints are met (column normalization)
        return self._normalize_grid_distributions(distributions)

    def _harville_dp(
        self, win_probs: np.ndarray, driver_ids: List[str]
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute grid probabilities using Harville dynamic programming.

        Parameters
        ----------
        win_probs : np.ndarray
            Array of win probabilities
        driver_ids : List[str]
            List of driver IDs

        Returns
        -------
        Dict[str, Dict[int, float]]
            Grid of position probabilities
        """
        n = len(win_probs)

        # Initialize DP table
        # dp[mask] holds the probability of reaching that state
        dp = np.zeros(1 << n)
        full_mask = (1 << n) - 1
        dp[full_mask] = 1.0  # Initially all drivers available

        # Initialize result grid
        grid = {driver_id: {} for driver_id in driver_ids}

        # Iterate over all subsets (masks) from full set down to empty
        for mask in range(full_mask, -1, -1):
            if dp[mask] == 0:
                continue  # Skip states with zero probability

            # Get available drivers in this state
            available = [i for i in range(n) if mask & (1 << i)]
            if not available:
                continue

            # Current position to assign (1-indexed)
            pos = n - len(available) + 1

            # Sum of strengths of available drivers
            s = sum(win_probs[i] for i in available)

            # Assign probabilities for this position
            for i in available:
                p_i = win_probs[i] / (s + 1e-10)  # Avoid division by zero
                prob = dp[mask] * p_i

                # Update grid
                grid[driver_ids[i]][pos] = grid[driver_ids[i]].get(pos, 0.0) + prob

                # Update DP table for next state (driver i is removed)
                new_mask = mask & ~(1 << i)
                dp[new_mask] += prob

        return grid
