"""
Module for converting betting odds to position probabilities using robust constrained
optimization.

This module provides functionality to convert decimal betting odds into valid
probability distributions for race finishing positions. It uses advanced optimization
techniques to ensure the resulting probabilities are consistent with market odds while
maintaining mathematical properties like proper probability distributions.

The main class OddsToPositionProbabilityConverter handles:
- Bookmaker margin removal
- Favorite-longshot bias adjustment
- Market constraint satisfaction
- Probability distribution smoothing
"""

import functools
from typing import Dict, List

import numpy as np
from scipy.optimize import Bounds, minimize

from gridrival_ai.utils.odds_conversion import power_method

SMOOTHNESS_ALPHA = 5.0
SHAPE_ALPHA = 7.0


class OddsToPositionProbabilityConverter:
    """Convert decimal odds to valid probability distributions using optimization.

    This class implements a robust method to convert decimal betting odds into proper
    probability distributions for race finishing positions. It uses advanced
    optimization techniques with multiple stability enhancements to ensure reliable
    results.

    The conversion process involves:
    1. Smart initialization using market constraints
    2. Constraint relaxation for numerical stability
    3. Multiple optimization strategies with fallbacks
    4. Shape and smoothness regularization
    5. Bookmaker margin removal

    Parameters
    ----------
    driver_odds : List[Dict[str, float]]
        List of dictionaries containing odds for each driver.
        Required keys:
            - driver_id: str
                Unique identifier for the driver
            - 1: float
                Decimal odds for finishing 1st (winner)
        Optional keys:
            - N: float
                Decimal odds for finishing in positions 1 to N
                (e.g., 3 for top3, 6 for top6, etc.)

    Attributes
    ----------
    drivers : List[str]
        List of driver IDs in the same order as input odds
    num_drivers : int
        Total number of drivers
    markets : List[int]
        Available market positions (e.g., [1, 3, 6] for winner, podium, top6)
    adjusted_probs : Dict[int, np.ndarray]
        Adjusted probabilities for each market after removing bookmaker margin
    position_probs : np.ndarray
        Matrix of position probabilities (drivers × positions)

    Notes
    -----
    The number of possible positions is determined by the number of drivers.
    Position keys in odds must be integers and represent cumulative finishes.
    For example:
        {"driver_id": "VER", "1": 1.5, "3": 1.1, "6": 1.05}
        means odds for:
        - Finishing 1st: 1.5
        - Finishing in positions 1-3: 1.1
        - Finishing in positions 1-6: 1.05

    Examples
    --------
    >>> odds = [
    ...     {"driver_id": "VER", "1": 1.5, "3": 1.1},
    ...     {"driver_id": "HAM", "1": 3.0, "3": 1.5},
    ...     {"driver_id": "LEC", "1": 4.0, "3": 1.8}
    ... ]
    >>> converter = OddsToPositionProbabilityConverter(odds)
    >>> probs = converter.calculate_position_probabilities()
    >>> print(probs["VER"][1])  # Probability of VER finishing 1st
    0.55
    """

    def __init__(self, driver_odds: List[Dict[str, float]]) -> None:
        """Initialize converter with driver odds.

        Parameters
        ----------
        driver_odds : List[Dict[str, float]]
            List of dictionaries containing odds for each driver.
            See class docstring for format details.

        Raises
        ------
        ValueError
            If winner odds (position 1) are missing for any driver
        """
        self.driver_odds = driver_odds
        self.drivers = [d["driver_id"] for d in driver_odds]
        self.num_drivers = len(self.drivers)

        # Process markets and compute adjusted probabilities
        self.markets = self._get_markets()
        self.adjusted_probs = self._compute_adjusted_probabilities()

        # Initialize and optimize probabilities
        self.position_probs = self._initialize_prob_matrix()
        self._optimize_probabilities()

    def _get_markets(self) -> List[int]:
        """Get available markets from driver odds.

        Markets are positions for which we have odds (e.g., 1 for winner, 3 for podium).
        Winner odds (1) must be present for all drivers.

        Returns
        -------
        List[int]
            Sorted list of available markets in ascending order.

        Raises
        ------
        ValueError
            If winner odds (position 1) are missing for any driver.
        """
        # Get all numeric keys from odds that aren't 'driver_id'
        all_markets = set()
        for odds in self.driver_odds:
            markets = {
                int(k)
                for k in odds.keys()
                if isinstance(k, (int, str)) and str(k).isdigit()
            }
            all_markets.update(markets)

        if 1 not in all_markets:
            raise ValueError("Winner odds (1) are required for all drivers")

        # Sort markets in ascending order
        return sorted(list(all_markets))

    def _compute_adjusted_probabilities(self) -> Dict[int, np.ndarray]:
        """Adjust raw odds to remove bookmaker margin using the power method.

        This method processes each market position and converts the decimal odds to
        probabilities using Shin's power method. The method handles:
        - Bookmaker margin removal
        - Favorite-longshot bias adjustment
        - Target probability scaling based on position
        - Invalid odds validation and fallback

        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping position numbers to adjusted probabilities.
            Key 1 is for winner, key N is for positions 1 to N.
            Each value is a numpy array of probabilities summing to N.

        Notes
        -----
        The power method is used to transform raw probabilities while maintaining
        proper probability axioms and addressing market biases. For each market:
        - Target probability equals position number (e.g., 3 for top-3)
        - Invalid odds (≤ 1.0) result in zero probabilities
        - The method optimizes a power parameter to achieve target probability sum
        """

        def get_market_odds(pos: int) -> List[float]:
            """Extract valid odds for a given market position."""
            return [float(d.get(str(pos)) or d.get(pos) or 0) for d in self.driver_odds]

        def process_market(pos: int, odds: List[float]) -> np.ndarray:
            """Process single market position odds into probabilities."""
            target = float(min(pos, self.num_drivers))
            if all(o > 1.0 for o in odds):
                probs, _ = power_method(odds, target_probability=target)
                return probs
            return np.zeros(len(odds))

        return {
            pos: process_market(pos, odds)
            for pos in self.markets
            if (odds := get_market_odds(pos)) and any(odds)
        }

    def _initialize_prob_matrix(self) -> np.ndarray:
        """Create smart initial guess respecting market constraints.

        For each market interval (between consecutive market positions),
        distribute probabilities uniformly. For example:
        - If P(top1) = 0.1 and P(top3) = 0.3:
            P(pos1) = 0.1
            P(pos2) = (0.3 - 0.1) / 2 = 0.1
            P(pos3) = (0.3 - 0.1) / 2 = 0.1

        Returns
        -------
        np.ndarray
            Initial probability matrix respecting market constraints.
            Shape is (num_drivers, num_drivers).
        """
        matrix = np.zeros((self.num_drivers, self.num_drivers))

        # Process each market interval
        prev_market = 0
        prev_probs = np.zeros(self.num_drivers)

        for market in self.markets:
            current_probs = self.adjusted_probs[market]
            positions_in_interval = market - prev_market

            if positions_in_interval > 0:
                # Distribute probability difference uniformly across positions
                prob_diff = current_probs - prev_probs
                prob_per_position = prob_diff / positions_in_interval

                # Assign probabilities for each position in interval
                for pos in range(prev_market, market):
                    matrix[:, pos] = prob_per_position

            prev_market = market
            prev_probs = current_probs

        # Handle remaining positions if any
        if prev_market < self.num_drivers:
            remaining_positions = self.num_drivers - prev_market
            remaining_probs = 1.0 - prev_probs
            prob_per_position = remaining_probs / remaining_positions

            for pos in range(prev_market, self.num_drivers):
                matrix[:, pos] = prob_per_position

        return matrix

    def _optimize_probabilities(self) -> None:
        """Robust optimization with multiple fallback strategies.

        Attempts optimization using different solvers in order of preference:
        1. SLSQP with conservative settings
        2. trust-constr with conservative settings
        3. Fallback to iterative proportional fitting if both fail

        The optimization process minimizes a regularized objective function while
        satisfying probability constraints and market odds constraints.
        """
        try:
            # Try SLSQP first with more conservative settings
            self._run_optimization(method="SLSQP", max_iter=1000)
        except RuntimeError:
            try:
                # Try trust-constr with more conservative settings
                self._run_optimization(method="trust-constr", max_iter=2000)
            except Exception as e:
                # If both fail, use fallback
                self._apply_fallback_solution()
                print(f"Optimization failed, using approximate solution: {str(e)}")

    def _run_optimization(self, method: str, max_iter: int) -> None:
        """Core optimization routine with enhanced settings.

        Parameters
        ----------
        method : str
            Optimization method to use ("SLSQP" or "trust-constr")
        max_iter : int
            Maximum number of iterations for the optimizer

        Raises
        ------
        RuntimeError
            If optimization fails to converge
        """
        x0 = self.position_probs.flatten()

        # More conservative bounds to avoid numerical issues
        bounds = Bounds(1e-6, 1.0)

        constraints = [
            # Row sums must equal 1
            {
                "type": "eq",
                "fun": lambda x: x.reshape((self.num_drivers, self.num_drivers)).sum(
                    axis=1
                )
                - 1,
            },
            # Column sums must equal 1
            {
                "type": "eq",
                "fun": lambda x: x.reshape((self.num_drivers, self.num_drivers)).sum(
                    axis=0
                )
                - 1,
            },
        ]

        constraints += self._get_relaxed_constraints()

        if method == "SLSQP":
            options = {
                "maxiter": max_iter,
                "ftol": 1e-4,  # Less strict tolerance
                "eps": 1e-6,  # Less strict step size
                "disp": True,
            }
        else:  # trust-constr
            options = {
                "maxiter": max_iter,
                "xtol": 1e-4,  # Less strict tolerance
                "gtol": 1e-4,  # Less strict gradient tolerance
                "disp": True,
            }

        res = minimize(
            self._regularized_objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.position_probs = res.x.reshape((self.num_drivers, self.num_drivers))

    def _calculate_expected_positions(self, win_probs: np.ndarray) -> np.ndarray:
        """Calculate expected positions accounting for ties.

        For drivers with equal win probabilities, assigns their average position.
        E.g., if drivers 2-5 all have 12.5% win probability, they all get
        expected position (2+3+4+5)/4 = 3.5

        Parameters
        ----------
        win_probs : np.ndarray
            Array of win probabilities for each driver

        Returns
        -------
        np.ndarray
            Array of expected positions for each driver
        """
        n = len(win_probs)
        sorted_idx = np.argsort(-win_probs)
        sorted_probs = win_probs[sorted_idx]

        expected_pos = np.zeros(n)
        i = 0
        while i < n:
            # Find all drivers with same probability
            prob = sorted_probs[i]
            count = 1
            while i + count < n and sorted_probs[i + count] == prob:
                count += 1

            # Calculate average position for this group
            pos = range(i + 1, i + count + 1)
            avg_pos = sum(pos) / count

            # Assign to all drivers in group
            for j in range(count):
                expected_pos[sorted_idx[i + j]] = avg_pos

            i += count

        return expected_pos

    @functools.lru_cache(maxsize=1)
    def _compute_expected_positions(self) -> np.ndarray:
        win_probs = self.adjusted_probs[1]
        return self._calculate_expected_positions(win_probs)

    def _shape_penalty(self, matrix: np.ndarray) -> float:
        """Calculate shape and smoothness penalty for probability distribution.

        Combines two penalty terms:
        1. Shape penalty: deviation from ideal shape centered at expected position
        2. Smoothness penalty: penalizes large jumps between adjacent positions

        Parameters
        ----------
        matrix : np.ndarray
            Probability matrix to evaluate, shape (num_drivers, num_drivers)

        Returns
        -------
        float
            Combined shape and smoothness penalty value
        """
        expected_positions = self._compute_expected_positions()
        positions = np.arange(1, self.num_drivers + 1)

        # Calculate position differences for each driver and position
        # Shape: (num_positions, num_drivers)
        positions_diff = np.abs(positions[:, None] - expected_positions[None, :])

        # Calculate ideal shapes based on win probabilities
        # Shape: (num_drivers, num_positions)
        ideal_shapes = self.adjusted_probs[1][:, None] ** (
            positions_diff.T / SHAPE_ALPHA
        )
        ideal_shapes /= ideal_shapes.sum(axis=1)[:, None]

        # Base shape penalty
        base_penalty = np.sum((matrix - ideal_shapes) ** 2)

        # Smoothness penalty
        # Shape: (num_drivers, num_positions-1)
        diffs = np.diff(matrix, axis=1)

        # Shape: (num_positions-1, num_drivers)
        position_weights = 1.0 / (
            1.0 + np.abs(positions[:-1, None] - expected_positions[None, :])
        )

        # Transpose position_weights to match diffs shape
        smoothness_penalty = np.sum(position_weights.T * diffs**2)

        return base_penalty + SMOOTHNESS_ALPHA * smoothness_penalty

    def _regularized_objective(self, x: np.ndarray) -> float:
        """Objective function balancing entropy and shape penalties.

        Combines two terms:
        1. Negative entropy (to maximize entropy) with small weight
        2. Shape penalty (to minimize) with larger weight

        Parameters
        ----------
        x : np.ndarray
            Flattened probability matrix

        Returns
        -------
        float
            Combined objective value to minimize
        """
        matrix = x.reshape((self.num_drivers, self.num_drivers))

        # Entropy term - positive weight because we want to maximize it
        entropy = -np.sum(matrix * np.log(matrix + 1e-10))

        # Shape penalty - minimize this
        shape = self._shape_penalty(matrix)

        # Note: positive weight for entropy because we want to maximize it
        # Negative weight for shape because we want to minimize it

        return -0.2 * entropy + 0.8 * shape

    def _get_relaxed_constraints(self) -> list:
        """Create constraints with tolerance windows for all markets.

        Each market constraint ensures that the cumulative probabilities up to position
        N match the adjusted market probabilities. For example:
        - P1 market: P(pos1) matches market probability
        - P3 market: P(pos1) + P(pos2) + P(pos3) matches market probability

        Returns
        -------
        list
            List of constraint dictionaries for scipy.optimize.minimize
        """
        constraints = []

        # Add constraints for each market
        for market_pos in self.markets:

            def cumulative_constraint(x, pos=market_pos):
                matrix = x.reshape((self.num_drivers, self.num_drivers))
                cumulative_probs = matrix[:, :pos].sum(axis=1)
                return cumulative_probs - self.adjusted_probs[pos]

            constraints.append(
                {
                    "type": "eq",
                    "fun": cumulative_constraint,
                    "tol": 1e-3,  # Allow small deviations to help convergence
                }
            )

        return constraints

    def _apply_fallback_solution(self) -> None:
        """Fallback to iterative proportional fitting when optimization fails.

        Uses a simple row/column normalization approach to ensure valid probabilities
        when the main optimization methods fail. This is less accurate but more
        robust than the primary optimization methods.
        """
        # Simple row/column normalization
        for _ in range(10):
            # Normalize rows
            row_sums = self.position_probs.sum(axis=1)
            self.position_probs /= row_sums[:, None]

            # Normalize columns
            col_sums = self.position_probs.sum(axis=0)
            self.position_probs /= col_sums[None, :]

    def calculate_position_probabilities(self) -> Dict[str, Dict[int, float]]:
        """Calculate and return validated position probabilities.

        Returns
        -------
        Dict[str, Dict[int, float]]
            Nested dictionary mapping driver IDs to their position probabilities.
            First level key: driver ID (str)
            Second level key: position (int, 1-based)
            Values: probability of driver finishing in that position (float)

        Examples
        --------
        >>> probs = converter.calculate_position_probabilities()
        >>> print(probs["VER"])  # Get all probabilities for VER
        {1: 0.55, 2: 0.25, 3: 0.15, ...}
        >>> print(probs["VER"][1])  # Get probability of VER finishing 1st
        0.55
        """
        return {
            driver: {
                pos + 1: float(self.position_probs[i, pos])
                for pos in range(self.num_drivers)
            }
            for i, driver in enumerate(self.drivers)
        }
