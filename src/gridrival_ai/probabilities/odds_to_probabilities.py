import math
from typing import Dict, List

import numpy as np

TOLERANCE = 1e-6


def basic_method(odds: List[float], target_probability: float = 1.0) -> np.ndarray:
    """
    Convert betting odds to probabilities by taking reciprocals and removing the margin.

    For a list of odds [o1, o2, ...], the raw implied probabilities are
        r[i] = 1 / o[i]
    and then they are normalized so that they sum to target_probability.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds (> 1.0).
    target_probability : float, optional
        The sum of the probabilities after normalization.
        For mutually exclusive outcomes (e.g. win) target_probability=1.

    Returns
    -------
    np.ndarray
        Array of probabilities summing to target_probability.

    Examples
    --------
    >>> odds = [1.9, 3.5, 4.2]
    >>> basic_method(odds)  # approximately [0.526, 0.286, 0.188]
    """
    raw_probs = np.array([1 / o for o in odds])
    return raw_probs * (target_probability / raw_probs.sum())


def _dp_baseline_from_win_probs(
    driver_odds: List[Dict[str, float]], drivers: List[str]
) -> np.ndarray:
    """
    Compute a baseline grid probability matrix using only the win probabilities.

    Assumes that each driver's win probability is stored under key "1"
    (and that these values sum to 1 across drivers).

    The method uses a dynamic–programming (Harville–style) algorithm.

    Returns
    -------
    np.ndarray
        An (n x n) array B where B[i, j] is the probability that driver i finishes in grid position j+1.
    """
    n = len(drivers)
    # Here, driver_odds already hold win probabilities (after conversion).
    strengths = np.array([d["1"] for d in driver_odds])
    dp = np.zeros(1 << n)
    full_mask = (1 << n) - 1
    dp[full_mask] = 1.0

    # B[i, pos] accumulates the probability that driver i finishes in position pos+1.
    B = np.zeros((n, n))

    # For each subset (mask), assign the finishing position (0-indexed) as:
    # pos = n - (number of drivers in mask)
    for mask in range(full_mask, -1, -1):
        if dp[mask] == 0:
            continue
        available = [i for i in range(n) if mask & (1 << i)]
        if not available:
            continue
        s = sum(strengths[i] for i in available)
        pos = n - len(available)  # 0-indexed: pos=0 means 1st place
        for i in available:
            p_i = strengths[i] / (s + 1e-10)
            prob = dp[mask] * p_i
            B[i, pos] += prob
            new_mask = mask & ~(1 << i)
            dp[new_mask] += prob
    return B


class OddsToPositionProbabilityConverter:
    """
    Convert betting odds (in decimal format) from various markets into a grid probability matrix.

    Each driver is provided with betting odds for one or more cumulative markets.
    For example, in a 5–driver race a driver might be given:

         {"driver_id": "VER", "1": 5.0, "3": 2.0, "5": 1.0}

    meaning that:
      - The win odds are 5.0 (implying a ~20% chance after fair conversion),
      - The top–3 odds are 2.0 (implying a 50% chance to finish in the top 3),
      - The full grid (top–5) odds are 1.0 (i.e. a certainty, so if missing it is defaulted to 1.0).

    The converter works as follows:

      1. In __init__, the provided odds are first converted into implied probabilities.
         For the win market ("1") the conversion is done collectively using basic_method
         so that win probabilities sum to 1. For any other market (e.g. "3"),
         each driver's implied probability is computed as 1 / (odds).
         For the final market (equal to the number of drivers), if a driver omits it,
         a default value of 1.0 is used.

      2. A baseline grid probability matrix is computed using the (converted) win probabilities
         via a dynamic–programming (Harville) algorithm.

      3. For each driver and grid position, an adjustment factor is computed so that the cumulative
         sums of grid probabilities (positions 1 to m) match the (converted) cumulative probabilities.
         (For grid position j, the factor is the geometric mean of the ratios
            R(m) = (given cumulative probability for market m) / (baseline cumulative probability for positions 1..m)
         taken over all provided markets m with m >= j.)

      4. Finally, iterative proportional fitting (IPF) rebalances the adjusted matrix so that
         each driver's probabilities sum to 1 and each finishing position's probabilities sum to 1.
    """

    def __init__(self, driver_odds: List[Dict[str, float]]) -> None:
        """
        Initialize the converter with betting odds.

        Parameters
        ----------
        driver_odds : List[Dict[str, float]]
            Each dictionary must contain at least:
              - "driver_id": str
              - "1": float, the win odds (decimal odds > 1)
            Optionally, additional keys (e.g. "3", "5", etc.) are cumulative market odds.
            For the final market (equal to the number of drivers) if omitted, a default of 1.0 is used.
        """
        self.driver_odds = driver_odds
        self.drivers = [d["driver_id"] for d in driver_odds]
        self.num_drivers = len(self.drivers)
        self.markets = self._get_markets()

        # --- Convert odds to implied probabilities ---
        # For market "1": convert collectively so that win probabilities sum to 1.
        win_odds = [d["1"] for d in self.driver_odds]
        win_probs = basic_method(win_odds, target_probability=1.0)
        for i, d in enumerate(self.driver_odds):
            d["1"] = win_probs[i]
        # For every other market, convert by taking the reciprocal.
        # (We assume that the provided odds are margin–free or that the margin is acceptable.)
        for d in self.driver_odds:
            for k in d.keys():
                if k.isdigit() and k != "1":
                    d[k] = 1 / float(d[k])
        # Ensure that every driver has an entry for the final market (equal to num_drivers).
        final_key = str(self.num_drivers)
        for d in self.driver_odds:
            if final_key not in d:
                d[final_key] = 1.0  # 100% chance to finish somewhere

    def _get_markets(self) -> List[int]:
        """
        Extract and return a sorted list of market keys (as integers) from the input odds.

        Raises
        ------
        ValueError
            If any driver is missing win odds (market "1").
        """
        all_markets = set()
        for d in self.driver_odds:
            for k in d:
                if k.isdigit():
                    all_markets.add(int(k))
            if "1" not in d:
                raise ValueError(
                    f"Driver {d.get('driver_id', 'unknown')} is missing win odds (market '1')."
                )
        return sorted(all_markets)

    def _compute_baseline(self) -> np.ndarray:
        """
        Compute the baseline grid probability matrix from win probabilities.

        Returns
        -------
        np.ndarray
            An (n x n) matrix computed via the Harville DP algorithm.
        """
        return _dp_baseline_from_win_probs(self.driver_odds, self.drivers)

    def _adjust_by_markets(self, B: np.ndarray) -> np.ndarray:
        """
        Adjust the baseline matrix B using the additional cumulative probabilities.

        For each driver i and each grid position j (1-indexed), an adjustment factor is computed.
        For each provided market m (with m >= j) the ratio is:
            R(i, m) = (given cumulative probability for market m) / (baseline cumulative probability for positions 1..m)
        The adjustment factor for grid position j is the geometric mean of these ratios.
        Then, the baseline probability B[i, j] is multiplied by the factor.

        Returns
        -------
        np.ndarray
            The adjusted matrix.
        """
        n = self.num_drivers
        P = np.copy(B)
        for i, d in enumerate(self.driver_odds):
            # Get the provided markets for driver i.
            provided_markets = sorted([int(k) for k in d.keys() if k.isdigit()])
            if n not in provided_markets:
                provided_markets.append(n)
            # Compute ratios R for each provided market m.
            R = {}
            for m in provided_markets:
                # Baseline cumulative probability for positions 1..m.
                baseline_cum = float(
                    np.sum(B[i, :m])
                )  # positions 0..m-1 in 0-indexing.
                given_cum = d.get(str(m), 1.0)
                R[m] = given_cum / (baseline_cum + 1e-10)
            # Adjust each grid position j (1-indexed).
            for j in range(1, n + 1):
                factors = [R[m] for m in provided_markets if m >= j]
                if factors:
                    log_mean = np.mean([math.log(f) for f in factors])
                    factor = math.exp(log_mean)
                else:
                    factor = 1.0
                P[i, j - 1] = B[i, j - 1] * factor
        return P

    def _iterative_proportional_fitting(
        self, M: np.ndarray, tol: float = 1e-8, max_iter: int = 1000
    ) -> np.ndarray:
        """
        Rebalance the matrix M to be doubly–stochastic (each row and column sums to 1)
        using iterative proportional fitting.

        Returns
        -------
        np.ndarray
            The balanced matrix.
        """
        Q = np.copy(M)
        for _ in range(max_iter):
            row_sums = Q.sum(axis=1)
            Q = Q / (row_sums[:, None] + 1e-10)
            col_sums = Q.sum(axis=0)
            Q = Q / (col_sums[None, :] + 1e-10)
            if np.allclose(Q.sum(axis=1), 1, atol=tol) and np.allclose(
                Q.sum(axis=0), 1, atol=tol
            ):
                break
        return Q

    def calculate_position_probabilities(self) -> Dict[str, Dict[int, float]]:
        """
        Calculate and return the final grid finishing position probabilities.

        Returns
        -------
        Dict[str, Dict[int, float]]
            A nested dictionary mapping driver IDs to dictionaries mapping grid positions
            (1-indexed) to probabilities.
        """
        B = self._compute_baseline()
        P = self._adjust_by_markets(B)
        Q = self._iterative_proportional_fitting(P)
        result = {}
        for i, driver in enumerate(self.drivers):
            result[driver] = {}
            for j in range(self.num_drivers):
                result[driver][j + 1] = Q[i, j]
        return result
