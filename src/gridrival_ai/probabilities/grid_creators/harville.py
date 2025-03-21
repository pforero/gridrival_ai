"""
Harville grid creator implementation.

This module implements the Harville method for creating position
probability distributions from win odds.
"""

import numpy as np

from gridrival_ai.probabilities.distributions import (
    PositionDistribution,
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.probabilities.grid_creators.base import GridCreator
from gridrival_ai.probabilities.normalizers import get_grid_normalizer
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

    def create_session_distribution(
        self, odds_input: OddsStructure | dict, session_type: str = "race", **kwargs
    ) -> SessionDistribution:
        """
        Create a session distribution from odds using the Harville method.

        Parameters
        ----------
        odds_input : OddsStructure | dict
            OddsStructure instance or raw odds dictionary
        session_type : str, optional
            Session type to use for odds, by default "race"
        **kwargs
            Additional parameters for normalization:
            - method : str, optional
                Method to use for normalization, by default "sinkhorn"
            - Additional parameters for the chosen normalizer

        Returns
        -------
        SessionDistribution
            Session distribution for all drivers
        """
        # Ensure input is an OddsStructure
        odds_structure = self._ensure_odds_structure(odds_input)

        # Get win odds and driver IDs
        win_probs, driver_ids = self.convert_win_odds_to_probabilities(
            odds_structure, session_type
        )

        # Apply Harville dynamic programming algorithm
        driver_position_dict = self._harville_dp(win_probs, driver_ids)

        # Get all positions (assuming all drivers have the same positions available)
        all_positions = set()
        for positions in driver_position_dict.values():
            all_positions.update(positions.keys())
        all_positions = sorted(all_positions)

        # Create matrix representation
        matrix = np.zeros((len(driver_ids), len(all_positions)))
        pos_to_idx = {pos: idx for idx, pos in enumerate(all_positions)}

        for i, driver_id in enumerate(driver_ids):
            for pos, prob in driver_position_dict[driver_id].items():
                j = pos_to_idx[pos]
                matrix[i, j] = prob

        # Verify that all matrix values are between 0 and 1
        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError(
                "Harville algorithm produced probabilities outside [0, 1] range"
            )

        # Normalize the matrix using the specified method
        normalizer = get_grid_normalizer(
            method=kwargs.get("method", "sinkhorn"),
            **{k: v for k, v in kwargs.items() if k != "method"},
        )
        normalized_matrix = normalizer.normalize(matrix)

        # Convert matrix back to PositionDistribution objects
        distributions = {}
        for i, driver_id in enumerate(driver_ids):
            position_probs = {}
            for j, pos in enumerate(all_positions):
                if normalized_matrix[i, j] > 0:
                    position_probs[pos] = normalized_matrix[i, j]

            # Create PositionDistribution with validation turned off
            # since the values might be very close to 1.0 but not exactly
            distributions[driver_id] = PositionDistribution(
                position_probs, _validate=False
            )

        # Create and return session distribution with pre-normalized distributions
        return SessionDistribution(distributions, session_type, _validate=False)

    def create_race_distribution(
        self,
        odds_input: OddsStructure | dict,
        **kwargs,
    ) -> RaceDistribution:
        """
        Create a complete race distribution from odds.

        Parameters
        ----------
        odds_input : OddsStructure | dict
            OddsStructure instance or raw odds dictionary
        **kwargs
            Additional parameters for distribution creation including normalization:
            - method : str, optional
                Method to use for normalization, by default "sinkhorn"

        Returns
        -------
        RaceDistribution
            Race distribution containing all sessions
        """
        # Ensure input is an OddsStructure
        odds_structure = self._ensure_odds_structure(odds_input)

        # Create race session distribution
        race_dist = self.create_session_distribution(
            odds_input, session_type="race", **kwargs
        )

        # Create qualifying and sprint distributions if available in odds structure
        quali_dist = None
        if "qualifying" in odds_structure.sessions:
            quali_dist = self.create_session_distribution(
                odds_input, session_type="qualifying", **kwargs
            )

        sprint_dist = None
        if "sprint" in odds_structure.sessions:
            sprint_dist = self.create_session_distribution(
                odds_input, session_type="sprint", **kwargs
            )

        # Create and return RaceDistribution - it will handle missing sessions
        return RaceDistribution(
            race=race_dist, qualifying=quali_dist, sprint=sprint_dist
        )

    def convert_win_odds_to_probabilities(self, odds_structure, session_type):
        """
        Convert win odds to probabilities.

        Parameters
        ----------
        odds_structure : OddsStructure
            Odds structure to convert
        session_type : str
            Session type to use for odds

        Returns
        -------
        tuple
            Tuple of (win probabilities array, driver IDs list)
        """
        # Get win odds for session
        win_odds = odds_structure.get_win_odds(session_type)

        # Convert the win_odds dictionary values to a list of odds
        odds_list = list(win_odds.values())
        # Convert odds to probabilities using the new API
        probs_array = self.odds_converter.convert(odds_list)
        # Create a dictionary mapping driver IDs to their probabilities
        win_probs_dict = {
            driver_id: prob for driver_id, prob in zip(win_odds.keys(), probs_array)
        }

        # Get driver IDs and corresponding win probabilities
        driver_ids = list(win_probs_dict.keys())
        win_probs = np.array([win_probs_dict[driver_id] for driver_id in driver_ids])

        return win_probs, driver_ids

    def _harville_dp(
        self, win_probs: np.ndarray, driver_ids: list[str]
    ) -> dict[str, dict[int, float]]:
        """
        Compute grid probabilities using Harville dynamic programming.

        Parameters
        ----------
        win_probs : np.ndarray
            Array of win probabilities
        driver_ids : list[str]
            List of driver IDs

        Returns
        -------
        dict[str, dict[int, float]]
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
