"""
Cumulative market grid creator for GridRival AI.

This module implements a grid creator that uses cumulative market odds
(win, top-3, top-6, etc.) to create more accurate position distributions.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np

from gridrival_ai.probabilities.distributions import (
    PositionDistribution,
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.probabilities.grid_creators.base import GridCreator
from gridrival_ai.probabilities.normalizers import get_grid_normalizer
from gridrival_ai.probabilities.odds_structure import OddsStructure


class CumulativeGridCreator(GridCreator):
    """
    Grid creator using cumulative market odds.

    This class creates position distributions from cumulative market odds
    (win, top-3, top-6, etc.), offering more accurate position probabilities
    compared to using win odds alone.

    Parameters
    ----------
    max_position : int, optional
        Maximum finishing position to consider, by default 20
    baseline_method : str, optional
        Method for baseline weights within segments, by default "exponential"
    baseline_params : Dict, optional
        Parameters for the baseline method, by default None

    Notes
    -----
    The algorithm works by:
    1. Removing bookmaker margins from implied probabilities
    2. Segmenting positions based on cumulative markets
    3. Using baseline weights to distribute within segments
    4. Scaling weights to match the cumulative market constraints
    5. Enforcing column constraints while preserving win probabilities
    """

    def __init__(
        self,
        max_position: int = 20,
        baseline_method: str = "exponential",
        baseline_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the cumulative grid creator.

        Parameters
        ----------
        max_position : int, optional
            Maximum finishing position to consider, by default 20
        baseline_method : str, optional
            Method for baseline weights within segments, by default "exponential"
        baseline_params : Dict, optional
            Parameters for the baseline method, by default None
        **kwargs
            Additional parameters for the parent GridCreator
        """
        super().__init__(**kwargs)
        self.max_position = max_position
        self.baseline_method = baseline_method
        self.baseline_params = baseline_params or {}

    def create_session_distribution(
        self,
        odds_input: Union[OddsStructure, Dict],
        session_type: str = "race",
        **kwargs,
    ) -> SessionDistribution:
        """
        Create a session distribution from cumulative market odds.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
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

        # Check if we have multiple markets or just win odds
        session_odds = odds_structure.get_session_odds(session_type)
        position_thresholds = list(session_odds.keys())

        # If we only have win odds, fall back to Harville method
        if len(position_thresholds) == 1 and position_thresholds[0] == 1:
            # Get win odds and driver IDs
            win_odds = session_odds[1]
            odds_list = list(win_odds.values())
            driver_ids = list(win_odds.keys())

            # Convert to probabilities
            win_probs = self.odds_converter.convert(odds_list)

            position_distributions = self._harville_fallback(win_probs, driver_ids)
        else:
            # Process cumulative markets
            cumulative_probs = {}
            for threshold in position_thresholds:
                driver_odds = session_odds[threshold]
                # Convert odds to probabilities
                odds_list = list(driver_odds.values())
                driver_ids = list(driver_odds.keys())

                # Convert to probabilities with appropriate target sum
                # Target sum for threshold n should be n (e.g., top-3 sums to 3.0)
                probs_array = self.odds_converter.convert(
                    odds_list, target_sum=float(threshold)
                )

                # Create probability dictionary
                cumulative_probs[threshold] = {
                    driver_id: prob for driver_id, prob in zip(driver_ids, probs_array)
                }

            # Get all driver IDs across all thresholds
            all_driver_ids = set()
            for threshold_probs in cumulative_probs.values():
                all_driver_ids.update(threshold_probs.keys())

            position_distributions = self._convert_cumulative_to_position_distributions(
                cumulative_probs, list(all_driver_ids)
            )

        # Create and return session distribution
        return SessionDistribution(
            position_distributions, session_type=session_type, _validate=False
        )

    def create_race_distribution(
        self,
        odds_input: Union[OddsStructure, Dict],
        **kwargs,
    ) -> RaceDistribution:
        """
        Create a complete race distribution from odds.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
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

    def _harville_fallback(
        self, win_probs: np.ndarray, driver_ids: list[str]
    ) -> Dict[str, PositionDistribution]:
        """
        Fall back to Harville method for win-only probabilities.

        Parameters
        ----------
        win_probs : np.ndarray
            Array of win probabilities
        driver_ids : list[str]
            List of driver IDs

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        n = len(win_probs)
        if n == 0:
            return {}

        # Initialize result grid
        grid = {driver_id: {} for driver_id in driver_ids}

        # For a single driver case
        if n == 1:
            grid[driver_ids[0]] = {1: 1.0}
            return {driver_ids[0]: PositionDistribution(grid[driver_ids[0]])}

        # Initialize DP table (similar to HarvilleGridCreator._harville_dp)
        dp = np.zeros(1 << n)
        full_mask = (1 << n) - 1
        dp[full_mask] = 1.0  # Initially all drivers available

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

        # Convert grid to PositionDistribution objects
        return {
            driver_id: PositionDistribution(positions)
            for driver_id, positions in grid.items()
        }

    def _convert_cumulative_to_position_distributions(
        self, cumulative_probs: Dict[int, Dict[str, float]], driver_ids: List[str]
    ) -> Dict[str, PositionDistribution]:
        """
        Convert cumulative market probabilities to position distributions.

        Parameters
        ----------
        cumulative_probs : Dict[int, Dict[str, float]]
            Nested dictionary with format:
            {
                1: {"VER": 0.4, "HAM": 0.3, ...},  # Win probabilities
                3: {"VER": 0.7, "HAM": 0.6, ...},  # Top-3 probabilities
                6: {"VER": 0.9, "HAM": 0.8, ...},  # Top-6 probabilities
                ...
            }
        driver_ids : List[str]
            List of all driver IDs to include in the distributions

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        # Dictionary to store resulting distributions
        distributions = {}

        # Process each driver
        for driver_id in driver_ids:
            # Get cumulative probabilities for this driver
            driver_cum_probs, driver_quality = self._extract_driver_probs(
                driver_id, cumulative_probs
            )

            # Convert to position-specific probabilities
            position_probs = self._convert_driver_probs(
                driver_cum_probs, driver_quality
            )

            # Create position distribution
            try:
                distributions[driver_id] = PositionDistribution(position_probs)
            except Exception as e:
                # If validation fails, try to normalize and retry
                warnings.warn(
                    f"Initial distribution invalid for {driver_id}: {e}. "
                    f"Attempting normalization."
                )
                normalized = self._normalize_probs(position_probs)
                distributions[driver_id] = PositionDistribution(
                    normalized, _validate=False
                ).normalize()

        # Ensure column constraints
        distributions = self._ensure_column_constraints(distributions)

        return distributions

    def _extract_driver_probs(
        self,
        driver_id: str,
        cumulative_probs: Dict[int, Dict[str, float]],
    ) -> tuple[Dict[int, float], float]:
        """
        Extract cumulative probabilities for a driver with validation.

        Parameters
        ----------
        driver_id : str
            Driver ID
        cumulative_probs : Dict[int, Dict[str, float]]
            Nested dictionary of cumulative probabilities

        Returns
        -------
        tuple[Dict[int, float], float]
            Dictionary mapping thresholds to probabilities and driver quality estimate
        """
        thresholds = sorted(cumulative_probs.keys())
        driver_probs = {}

        # Extract available probabilities
        for threshold in thresholds:
            if driver_id in cumulative_probs.get(threshold, {}):
                driver_probs[threshold] = cumulative_probs[threshold][driver_id]

        # Ensure we have at least one probability
        if not driver_probs:
            # If no probabilities available, set a small default win probability
            # and full probability for max position
            driver_probs[1] = 0.01  # Small default win probability
            driver_probs[self.max_position] = 1.0
            warnings.warn(
                f"No probabilities found for driver {driver_id}. Using defaults."
            )

        # Estimate driver quality based on win probability or early cumulative markets
        driver_quality = 0.5  # Default to mid-tier
        if 1 in driver_probs:
            driver_quality = driver_probs[
                1
            ]  # Win probability is best quality indicator
        elif 3 in driver_probs:
            driver_quality = driver_probs[3] / 3  # Estimate from top-3 probability
        elif thresholds:
            # Use earliest available threshold
            first_threshold = min(driver_probs.keys())
            driver_quality = driver_probs[first_threshold] / first_threshold

        # Add implicit threshold for full grid if not present
        if self.max_position not in driver_probs:
            driver_probs[self.max_position] = 1.0

        # Ensure monotonically increasing probabilities
        prev_threshold = 0
        prev_prob = 0.0
        ordered_probs = {}

        for threshold in sorted(driver_probs.keys()):
            prob = max(driver_probs[threshold], prev_prob)
            ordered_probs[threshold] = prob
            prev_threshold = threshold
            prev_prob = prob

        return ordered_probs, driver_quality

    def _convert_driver_probs(
        self, driver_cum_probs: Dict[int, float], driver_quality: float
    ) -> Dict[int, float]:
        """
        Convert cumulative probabilities to position-specific probabilities.

        Parameters
        ----------
        driver_cum_probs : Dict[int, float]
            Dictionary mapping thresholds to cumulative probabilities
        driver_quality : float
            Measure of driver quality (0 to 1)

        Returns
        -------
        Dict[int, float]
            Dictionary mapping positions to probabilities
        """
        # Sort thresholds
        sorted_thresholds = sorted(driver_cum_probs.keys())
        position_probs = {}

        # Process each segment defined by thresholds
        prev_threshold = 0
        prev_cum_prob = 0.0

        for threshold in sorted_thresholds:
            cum_prob = driver_cum_probs[threshold]

            # Probability mass for this segment
            segment_prob = cum_prob - prev_cum_prob

            # Skip if no probability in this segment
            if segment_prob <= 0:
                warnings.warn(
                    f"Non-increasing probabilities detected for thresholds "
                    f"{prev_threshold} to {threshold}. Skipping segment."
                )
                prev_threshold = threshold
                prev_cum_prob = cum_prob
                continue

            # Define segment range
            start_pos = prev_threshold + 1
            end_pos = threshold

            # Get baseline weights for this segment
            # Regular segments use normal weighting
            if threshold != self.max_position or prev_threshold == 0:
                baseline_weights = self._get_baseline_weights(
                    start_pos, end_pos, prev_cum_prob, driver_cum_probs
                )
            else:
                # Special case for the tail segment (after last known threshold)
                baseline_weights = self._get_tail_weights(
                    start_pos, end_pos, driver_quality
                )

            # Scale weights to match segment probability
            total_weight = sum(baseline_weights.values())
            if total_weight > 0:
                for pos in range(start_pos, end_pos + 1):
                    if pos in baseline_weights:
                        position_probs[pos] = (
                            segment_prob * baseline_weights[pos] / total_weight
                        )

            prev_threshold = threshold
            prev_cum_prob = cum_prob

        # Ensure all positions have a probability (even if very small)
        for pos in range(1, self.max_position + 1):
            if pos not in position_probs:
                position_probs[pos] = 1e-10  # Very small but non-zero probability

        return position_probs

    def _get_baseline_weights(
        self,
        start_pos: int,
        end_pos: int,
        prev_cum_prob: float,
        driver_cum_probs: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """
        Get baseline weights for positions within a segment.

        Different methods can be used to assign weights:
        - "exponential": Exponential decay from start to end positions
        - "linear": Linear decay from start to end
        - "uniform": Equal weight for all positions

        Parameters
        ----------
        start_pos : int
            Starting position of the segment
        end_pos : int
            Ending position of the segment
        prev_cum_prob : float
            Previous cumulative probability (used for quality estimation)
        driver_cum_probs : Optional[Dict[int, float]], optional
            Complete driver cumulative probabilities for richer quality estimation

        Returns
        -------
        Dict[int, float]
            Dictionary mapping positions to baseline weights
        """
        weights = {}

        if start_pos > end_pos:
            return weights

        # Estimate driver quality based on win probability and cumulative probabilities
        driver_quality = prev_cum_prob
        if driver_cum_probs and 1 in driver_cum_probs:
            # Win probability is a better indicator of quality
            driver_quality = max(driver_quality, driver_cum_probs[1])

        if self.baseline_method == "exponential":
            # Get decay parameter (higher values = faster decay)
            # Better drivers (higher quality) should have faster decay
            base_decay = self.baseline_params.get("decay", 0.5)
            quality_factor = 1.0 + driver_quality  # Adjust decay by quality
            decay = base_decay * quality_factor

            # Create exponentially decaying weights
            for pos in range(start_pos, end_pos + 1):
                weights[pos] = np.exp(-decay * (pos - start_pos))

        elif self.baseline_method == "linear":
            # Linear decay from start to end
            # Adjust steepness based on driver quality
            quality_factor = 1.0 + driver_quality
            for pos in range(start_pos, end_pos + 1):
                base_weight = end_pos + 1 - pos
                weights[pos] = base_weight * quality_factor

        elif self.baseline_method == "uniform":
            # Equal weights
            for pos in range(start_pos, end_pos + 1):
                weights[pos] = 1.0

        else:
            # Default to exponential
            for pos in range(start_pos, end_pos + 1):
                weights[pos] = np.exp(-0.5 * (pos - start_pos))

        # Special case for first (winning) position - adjust weight based on quality
        if start_pos == 1 and 1 in weights:
            # Driver "quality" approximated by previous cumulative probability
            quality_factor = 1.0 + (driver_quality * 2)  # Higher for favorites
            weights[1] *= quality_factor

        return weights

    def _get_tail_weights(
        self,
        start_pos: int,  # Position after last threshold
        end_pos: int,  # Maximum grid position (e.g., 20)
        driver_quality: float,  # Quality measure (e.g., win probability)
    ) -> Dict[int, float]:
        """
        Get weights for positions beyond the last threshold.

        Parameters
        ----------
        start_pos : int
            Starting position (after the last threshold)
        end_pos : int
            Maximum grid position
        driver_quality : float
            Measure of driver quality (0 to 1)

        Returns
        -------
        Dict[int, float]
            Weights for each position in the tail
        """
        weights = {}
        num_positions = end_pos - start_pos + 1

        if num_positions <= 0:
            return weights

        # Different patterns based on driver quality
        if driver_quality > 0.6:  # Strong driver
            # Decreasing exponential: higher prob at start_pos, lower at end_pos
            decay = 2.0  # Steeper decay for stronger drivers
            for pos in range(start_pos, end_pos + 1):
                weights[pos] = np.exp(-decay * (pos - start_pos) / num_positions)
        elif driver_quality < 0.3:  # Weak driver
            # Increasing linear: lower prob at start_pos, higher at end_pos
            for pos in range(start_pos, end_pos + 1):
                weights[pos] = 1.0 + (pos - start_pos) / (num_positions / 2)
        else:  # Mid-tier driver
            # Slightly decreasing: fairly flat but bit higher at start_pos
            for pos in range(start_pos, end_pos + 1):
                weights[pos] = 1.0 - 0.5 * (pos - start_pos) / num_positions

        return weights

    def _normalize_probs(self, probs: Dict[int, float]) -> Dict[int, float]:
        """
        Normalize probabilities to sum to 1.0.

        Parameters
        ----------
        probs : Dict[int, float]
            Dictionary mapping positions to probabilities

        Returns
        -------
        Dict[int, float]
            Normalized probabilities
        """
        total = sum(probs.values())
        if total <= 0:
            # If no valid probabilities, create uniform distribution
            return {i: 1.0 / self.max_position for i in range(1, self.max_position + 1)}
        return {k: v / total for k, v in probs.items()}
    def _ensure_column_constraints(
        self, distributions: Dict[str, PositionDistribution]
    ) -> Dict[str, PositionDistribution]:
        """
        Ensure each position has probability sum = 1.0 across all drivers.

        This step ensures the joint distribution is fully consistent across
        both rows (drivers) and columns (positions), while preserving the
        original win probabilities.

        Parameters
        ----------
        distributions : Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions

        Returns
        -------
        Dict[str, PositionDistribution]
            Adjusted distributions
        """
        drivers = list(distributions.keys())
        n_drivers = len(drivers)
        n_positions = self.max_position

        if n_drivers == 0:
            return distributions

        # Create probability matrix [driver, position]
        P = np.zeros((n_drivers, n_positions))
        for i, driver in enumerate(drivers):
            for pos in range(1, n_positions + 1):
                P[i, pos - 1] = distributions[driver].get(pos)

        # Save original win probabilities
        original_win_probs = P[:, 0].copy()

        # Check if win probabilities sum to 1.0
        win_sum = original_win_probs.sum()
        if not np.isclose(win_sum, 1.0) and win_sum > 0:
            # Scale win probabilities to sum to 1.0
            original_win_probs = original_win_probs / win_sum

        # Iterative proportional fitting (preserving win probabilities)
        max_iterations = 10
        for _ in range(max_iterations):
            # Row normalization
            row_sums = P.sum(axis=1, keepdims=True)
            P = P / np.where(row_sums > 0, row_sums, 1.0)

            # Restore original win probabilities
            P[:, 0] = original_win_probs

            # Column normalization (except position 1)
            for pos in range(1, n_positions):
                col_sum = P[:, pos].sum()
                if col_sum > 0:
                    P[:, pos] = P[:, pos] / col_sum

            # Re-normalize each driver's remaining probabilities to ensure row sums to 1
            for i in range(n_drivers):
                win_prob = P[i, 0]
                remaining_prob = 1.0 - win_prob
                row_sum = P[i, 1:].sum()
                if row_sum > 0:
                    P[i, 1:] = P[i, 1:] * (remaining_prob / row_sum)

        # Convert back to PositionDistribution objects
        result = {}
        for i, driver in enumerate(drivers):
            position_probs = {
                pos + 1: P[i, pos] for pos in range(n_positions) if P[i, pos] > 0
            }
            result[driver] = PositionDistribution(
                position_probs, _validate=False
            ).normalize()

        return result

