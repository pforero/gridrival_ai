"""
Cumulative market odds conversion for position distributions.

This module extends the conversion functionality with support for cumulative
market odds (win, top-3, top-6, etc.) conversion to full position distributions.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from gridrival_ai.probabilities.converters.odds_converter import OddsConverter
from gridrival_ai.probabilities.core import PositionDistribution


class CumulativeMarketConverter(OddsConverter):
    """
    Convert cumulative market odds to complete position distributions.

    This class implements an algorithm for converting cumulative market odds
    (win, top-3, top-6, etc.) into full probability distributions across all
    finishing positions. It handles the constraints that each cumulative market
    places on the sum of probabilities for a range of positions.

    Parameters
    ----------
    max_position : int, optional
        Maximum finishing position to consider, by default 20
    baseline_method : str, optional
        Method for baseline weights within segments, by default "exponential"
    baseline_params : Dict, optional
        Parameters for the baseline method, by default None
    fallback_converter : Optional[OddsConverter], optional
        Converter to use for fallback cases, by default None
    enforce_column_constraints : bool, optional
        Whether to enforce that each position sums to 1.0 across drivers, by default True

    Notes
    -----
    The algorithm works by:
    1. Removing bookmaker margins from implied probabilities
    2. Segmenting positions based on cumulative markets
    3. Using baseline weights to distribute within segments
    4. Scaling weights to match the cumulative market constraints
    5. Optionally enforcing column constraints while preserving win probabilities

    Example
    -------
    >>> converter = CumulativeMarketConverter()
    >>> cumulative_probs = {
    ...     1: {"VER": 0.4, "HAM": 0.3, "NOR": 0.2},  # Win probabilities
    ...     3: {"VER": 0.7, "HAM": 0.6, "NOR": 0.5},  # Top-3 probabilities
    ...     6: {"VER": 0.9, "HAM": 0.8, "NOR": 0.7},  # Top-6 probabilities
    ... }
    >>> distributions = converter.convert_all(cumulative_probs)
    >>> distributions["VER"][1]  # Probability of VER finishing P1
    0.4
    """

    def __init__(
        self,
        max_position: int = 20,
        baseline_method: str = "exponential",
        baseline_params: Optional[Dict] = None,
        fallback_converter: Optional[OddsConverter] = None,
        enforce_column_constraints: bool = True,
    ) -> None:
        """Initialize converter with parameters."""
        self.max_position = max_position
        self.baseline_method = baseline_method
        self.baseline_params = baseline_params or {}
        self.fallback_converter = fallback_converter
        self.enforce_column_constraints = enforce_column_constraints

    def convert(self, odds: List[float], target_sum: float = 1.0) -> np.ndarray:
        """
        Convert odds to probabilities to comply with OddsConverter interface.

        This is implemented primarily for API compatibility.
        For cumulative market conversion, use convert_all method instead.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds
        target_sum : float, optional
            Target sum for the probabilities, by default 1.0

        Returns
        -------
        np.ndarray
            Array of probabilities
        """
        if not odds:
            return np.array([])

        raw_probs = np.array([1.0 / o if o > 0 else 0.0 for o in odds])
        total = raw_probs.sum()
        if total <= 0:
            return np.zeros_like(raw_probs)
        return raw_probs * (target_sum / total)

    def convert_to_dict(
        self, odds: List[float], target_sum: float = 1.0
    ) -> Dict[int, float]:
        """
        Convert odds to probability dictionary.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds
        target_sum : float, optional
            Target sum for the probabilities, by default 1.0

        Returns
        -------
        Dict[int, float]
            Dictionary mapping positions to probabilities
        """
        probs = self.convert(odds, target_sum)
        return {i + 1: float(p) for i, p in enumerate(probs)}

    def to_position_distribution(
        self, odds: List[float], target_sum: float = 1.0, validate: bool = True
    ) -> PositionDistribution:
        """
        Convert odds to position distribution.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds
        target_sum : float, optional
            Target sum for the probabilities, by default 1.0
        validate : bool, optional
            Whether to validate the distribution, by default True

        Returns
        -------
        PositionDistribution
            Position distribution object
        """
        probs_dict = self.convert_to_dict(odds, target_sum)
        return PositionDistribution(probs_dict, _validate=validate)

    def convert_all(
        self, cumulative_probs: Dict[int, Dict[str, float]]
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

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions

        Notes
        -----
        All input probabilities should be de-margined (house edge removed).
        If only win probabilities are provided, falls back to Harville method.

        Raises
        ------
        ValueError
            If no cumulative probabilities are provided or no drivers are found
        """
        # Validate input
        if not cumulative_probs:
            raise ValueError("Empty cumulative probabilities provided")

        # Get all driver IDs and thresholds
        driver_ids = set()
        for thresh_probs in cumulative_probs.values():
            driver_ids.update(thresh_probs.keys())

        if not driver_ids:
            raise ValueError("No drivers found in cumulative probabilities")

        thresholds = sorted(cumulative_probs.keys())

        # Check if we only have win probabilities
        if len(thresholds) == 1 and thresholds[0] == 1:
            # Fall back to Harville method if available
            if self.fallback_converter:
                win_probs = cumulative_probs[1]
                return self._fallback_to_harville(win_probs, list(driver_ids))
            else:
                warnings.warn(
                    "Only win probabilities provided. Results may be suboptimal without "
                    "a fallback converter for the Harville method."
                )

        # Dictionary to store resulting distributions
        distributions = {}

        # Process each driver
        for driver_id in driver_ids:
            # Get cumulative probabilities for this driver
            driver_cum_probs, driver_quality = self._extract_driver_probs(
                driver_id, cumulative_probs, thresholds
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

        # Check column constraints and adjust if necessary
        if self.enforce_column_constraints:
            distributions = self._ensure_column_constraints(distributions)

        return distributions

    def _extract_driver_probs(
        self,
        driver_id: str,
        cumulative_probs: Dict[int, Dict[str, float]],
        thresholds: List[int],
    ) -> Tuple[Dict[int, float], float]:
        """
        Extract cumulative probabilities for a driver with validation.

        Parameters
        ----------
        driver_id : str
            Driver ID
        cumulative_probs : Dict[int, Dict[str, float]]
            Nested dictionary of cumulative probabilities
        thresholds : List[int]
            List of thresholds to extract

        Returns
        -------
        Tuple[Dict[int, float], float]
            Dictionary mapping thresholds to probabilities and driver quality estimate
        """
        driver_probs = {}

        # Extract available probabilities
        for threshold in thresholds:
            if driver_id in cumulative_probs.get(threshold, {}):
                driver_probs[threshold] = cumulative_probs[threshold][driver_id]

        # Ensure we have at least one probability
        if not driver_probs:
            raise ValueError(f"No probabilities found for driver {driver_id}")

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

            # Re-normalize each driver's remaining probabilities to ensure row sums to 1.0
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

    def _fallback_to_harville(
        self, win_probs: Dict[str, float], driver_ids: List[str]
    ) -> Dict[str, PositionDistribution]:
        """
        Fall back to Harville method for win-only probabilities.

        Parameters
        ----------
        win_probs : Dict[str, float]
            Dictionary mapping driver IDs to win probabilities
        driver_ids : List[str]
            List of all driver IDs

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        if not win_probs:
            return {}

        # Ensure win probabilities sum to 1.0
        total_win_prob = sum(win_probs.values())
        if not np.isclose(total_win_prob, 1.0) and total_win_prob > 0:
            win_probs = {d: p / total_win_prob for d, p in win_probs.items()}

        if self.fallback_converter is None:
            # Create default distributions if no fallback
            distributions = {}
            for driver_id in driver_ids:
                win_prob = win_probs.get(driver_id, 0.0)
                # Create simple distribution with high probability for win position
                # and the rest distributed across remaining positions
                probs = {1: win_prob}
                remaining = 1.0 - win_prob
                for pos in range(2, self.max_position + 1):
                    probs[pos] = remaining / (self.max_position - 1)
                distributions[driver_id] = PositionDistribution(probs)
            return distributions

        # Use the fallback converter for Harville method
        sorted_drivers = sorted(driver_ids)
        odds_list = [1.0 / win_probs.get(d, 0.001) for d in sorted_drivers]
        grid_probs = self.fallback_converter.convert_to_grid(odds_list, sorted_drivers)

        # Convert to PositionDistribution objects
        distributions = {}
        for driver_id, probs in grid_probs.items():
            distributions[driver_id] = PositionDistribution(probs)

        return distributions

    @staticmethod
    def remove_margin(
        odds: Dict[str, float], method: str = "proportional", target_sum: float = 1.0
    ) -> Dict[str, float]:
        """
        Remove bookmaker margin from raw betting odds.

        Parameters
        ----------
        odds : Dict[str, float]
            Dictionary mapping driver IDs to decimal odds
        method : str, optional
            Method for margin removal, options:
            - "proportional": Scale probabilities proportionally
            - "equal": Reduce all probabilities by equal amount
            - "weighted": Reduce longshots more than favorites
            By default "proportional"
        target_sum : float, optional
            Target sum for probabilities, by default 1.0

        Returns
        -------
        Dict[str, float]
            Dictionary mapping driver IDs to de-margined probabilities
        """
        # Convert to implied probabilities
        implied_probs = {d: 1.0 / o if o > 0 else 0.0 for d, o in odds.items()}
        total = sum(implied_probs.values())

        if total <= 0:
            return {d: 0.0 for d in odds}

        # Calculate overround
        overround = total / target_sum

        if method == "proportional":
            # Scale probabilities proportionally
            return {d: p / total * target_sum for d, p in implied_probs.items()}

        elif method == "equal":
            # Reduce all probabilities by equal amount
            excess = total - target_sum
            reduction_per_driver = excess / len(odds)
            demargin_probs = {}

            for d, p in implied_probs.items():
                # Ensure we don't reduce below zero
                reduction = min(reduction_per_driver, 0.9 * p)
                demargin_probs[d] = p - reduction

            # Ensure exact sum
            demargin_total = sum(demargin_probs.values())
            if demargin_total > 0:
                scale = target_sum / demargin_total
                return {d: p * scale for d, p in demargin_probs.items()}
            else:
                return {d: target_sum / len(odds) for d in odds}

        elif method == "weighted":
            # Weight by reciprocal of probability (reduces longshots more)
            weights = {d: 1.0 / p if p > 0 else 0.0 for d, p in implied_probs.items()}
            total_weight = sum(weights.values())

            if total_weight <= 0:
                return {d: target_sum / len(odds) for d in odds}

            excess = total - target_sum
            demargin_probs = {}

            for d, p in implied_probs.items():
                weight = weights[d] / total_weight
                reduction = excess * weight
                # Ensure we don't reduce below zero
                reduction = min(reduction, 0.9 * p)
                demargin_probs[d] = p - reduction

            # Ensure exact sum
            demargin_total = sum(demargin_probs.values())
            if demargin_total > 0:
                scale = target_sum / demargin_total
                return {d: p * scale for d, p in demargin_probs.items()}
            else:
                return {d: target_sum / len(odds) for d in odds}

        else:
            # Default to proportional
            return {d: p / total * target_sum for d, p in implied_probs.items()}

    @classmethod
    def from_decimal_odds(
        cls,
        decimal_odds: Dict[int, Dict[str, float]],
        margin_removal_method: str = "proportional",
        **kwargs,
    ) -> Dict[str, PositionDistribution]:
        """
        Create distributions directly from decimal betting odds.

        This is a convenience method that handles margin removal and conversion
        in one step.

        Parameters
        ----------
        decimal_odds : Dict[int, Dict[str, float]]
            Nested dictionary with decimal odds format:
            {
                1: {"VER": 2.2, "HAM": 4.0, ...},  # Win odds
                3: {"VER": 1.2, "HAM": 1.5, ...},  # Top-3 odds
                6: {"VER": 1.05, "HAM": 1.15, ...},  # Top-6 odds
                ...
            }
        margin_removal_method : str, optional
            Method for margin removal, by default "proportional"
        **kwargs
            Additional arguments for the converter

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        converter = cls(**kwargs)

        # Convert decimal odds to probabilities and remove margin
        cumulative_probs = {}
        for market, odds in decimal_odds.items():
            # Remove margin while preserving target sum = market size
            cumulative_probs[market] = converter.remove_margin(
                odds, method=margin_removal_method, target_sum=float(market)
            )

        # Convert to distributions
        return converter.convert_all(cumulative_probs)
