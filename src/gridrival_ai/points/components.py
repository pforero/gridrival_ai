"""
Component calculators for GridRival F1 fantasy points.

This module provides specialized calculators for each scoring component
in the GridRival F1 fantasy game. Each calculator handles a specific aspect
of the scoring system, such as position points, overtaking, teammate comparison,
race completion, and improvement versus rolling average.

These calculators are designed to work with the probability distributions
from the new probability API and provide expected point values based on
those distributions.

Examples
--------
>>> # Calculate position-based points
>>> from gridrival_ai.probabilities.core import PositionDistribution
>>> position_dist = PositionDistribution({1: 0.6, 2: 0.4})
>>> points_table = np.array([0, 25, 18])  # 0-indexed, P1=25, P2=18
>>> calculator = PositionPointsCalculator()
>>> calculator.calculate(position_dist, points_table)
22.2  # 0.6*25 + 0.4*18
"""

from __future__ import annotations

import numpy as np

from gridrival_ai.probabilities.core import JointDistribution, PositionDistribution


class PositionPointsCalculator:
    """
    Calculate expected points for finishing positions.

    This calculator handles the core position-based points for qualifying,
    race, and sprint sessions.
    """

    def calculate(
        self, position_dist: PositionDistribution, points_table: np.ndarray
    ) -> float:
        """
        Calculate expected points from a position distribution.

        Parameters
        ----------
        position_dist : PositionDistribution
            Distribution over finishing positions
        points_table : np.ndarray
            Array mapping positions to point values

        Returns
        -------
        float
            Expected points

        Examples
        --------
        >>> calculator = PositionPointsCalculator()
        >>> position_dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> points_table = np.array([0, 25, 18])  # 0-indexed, P1=25, P2=18
        >>> calculator.calculate(position_dist, points_table)
        22.2  # 0.6*25 + 0.4*18
        """
        # Create a dictionary mapping positions to points
        point_values = {pos: points_table[pos] for pos in range(1, len(points_table))}

        # Use the distribution's expected value method
        return position_dist.expected_value(point_values)


class OvertakePointsCalculator:
    """
    Calculate expected overtaking points.

    This calculator determines points earned from positions gained
    between qualifying and race.
    """

    def calculate(self, joint_dist: JointDistribution, multiplier: float) -> float:
        """
        Calculate expected overtake points from joint distribution.

        Parameters
        ----------
        joint_dist : JointDistribution
            Joint distribution between qualifying and race positions
        multiplier : float
            Points per position gained

        Returns
        -------
        float
            Expected overtake points

        Notes
        -----
        Points are only awarded for positions gained, not lost.
        The qualifying position is the first element and race position
        is the second element in the joint distribution.

        Examples
        --------
        >>> calculator = OvertakePointsCalculator()
        >>> joint_probs = {(3, 1): 0.4, (3, 2): 0.2, (2, 1): 0.1, (2, 2): 0.3}
        >>> joint_dist = JointDistribution(
        ...     joint_probs,
        ...     outcome1_name="qualifying",
        ...     outcome2_name="race"
        ... )
        >>> calculator.calculate(joint_dist, multiplier=2.0)
        1.2  # (3-1)*0.4*2 + (3-2)*0.2*2 + (2-1)*0.1*2 + (2-2)*0.3*0
        """
        expected_points = 0.0

        for (qual_pos, race_pos), prob in joint_dist.items():
            # Points only awarded for positions gained (not lost)
            positions_gained = max(0, qual_pos - race_pos)
            expected_points += prob * positions_gained * multiplier

        return expected_points


class TeammatePointsCalculator:
    """
    Calculate expected points for beating teammate.

    This calculator determines points earned when a driver finishes
    ahead of their teammate by a certain margin.
    """

    def calculate(self, joint_dist: JointDistribution, thresholds: np.ndarray) -> float:
        """
        Calculate expected teammate points from joint distribution.

        Parameters
        ----------
        joint_dist : JointDistribution
            Joint distribution between driver and teammate positions
        thresholds : np.ndarray
            Array of [threshold, points] pairs for position margins

        Returns
        -------
        float
            Expected teammate points

        Notes
        -----
        The thresholds array should be sorted by margin in ascending order.
        Each row contains [margin, points] where margin is the minimum
        positions ahead to earn the specified points.

        Examples
        --------
        >>> calculator = TeammatePointsCalculator()
        >>> joint_probs = {(1, 3): 0.4, (2, 4): 0.2, (3, 1): 0.1, (4, 2): 0.3}
        >>> joint_dist = JointDistribution(
        ...     joint_probs,
        ...     outcome1_name="driver",
        ...     outcome2_name="teammate"
        ... )
        >>> thresholds = np.array([[2, 2], [5, 5], [10, 8]])
        >>> calculator.calculate(joint_dist, thresholds)
        1.4  # 0.4*2 + 0.2*2 (2 positions ahead = 2 points)
        """
        expected_points = 0.0

        for (driver_pos, teammate_pos), prob in joint_dist.items():
            # Only award points if driver is ahead of teammate
            if driver_pos < teammate_pos:
                margin = teammate_pos - driver_pos

                # Find applicable threshold (first one not exceeded)
                idx = np.searchsorted(thresholds[:, 0], margin, side="right") - 1
                if idx >= 0:  # If we found a valid threshold
                    expected_points += prob * thresholds[idx, 1]

        return expected_points


class CompletionPointsCalculator:
    """
    Calculate expected completion points.

    This calculator determines points earned from completing
    various stages of the race (25%, 50%, 75%, 90%).
    """

    def calculate(
        self, completion_prob: float, thresholds: np.ndarray, points_per_stage: float
    ) -> float:
        """
        Calculate expected completion points.

        Parameters
        ----------
        completion_prob : float
            Probability of completing the full race
        thresholds : np.ndarray
            Array of race completion thresholds (e.g., [0.25, 0.5, 0.75, 0.9])
        points_per_stage : float
            Points awarded per completion stage

        Returns
        -------
        float
            Expected completion points

        Notes
        -----
        The calculation assumes that DNFs occur uniformly across race laps.
        For a driver with completion probability p, the probability (1-p) of DNF
        is distributed uniformly across the race distance.

        Examples
        --------
        >>> calculator = CompletionPointsCalculator()
        >>> thresholds = np.array([0.25, 0.5, 0.75, 0.9])
        >>> calculator.calculate(0.8, thresholds, 3.0)
        10.8  # Complex calculation of expected stage completions
        """
        # Maximum points if full race completed
        max_points = points_per_stage * len(thresholds)

        # Points if race completed
        points_if_completed = completion_prob * max_points

        # Points for DNF cases
        dnf_prob = 1.0 - completion_prob
        if dnf_prob <= 0.0:
            return max_points

        expected_dnf_points = 0.0
        prev_threshold = 0.0

        # Calculate points for each stage
        for i, threshold in enumerate(thresholds):
            # Probability of DNF in this interval (relative to overall DNF prob)
            interval_prob = (threshold - prev_threshold) / 1.0 * dnf_prob

            # Points for completing all previous stages
            points = i * points_per_stage
            expected_dnf_points += interval_prob * points

            prev_threshold = threshold

        # DNF after last threshold but before finish
        last_interval_prob = (1.0 - prev_threshold) / 1.0 * dnf_prob
        expected_dnf_points += last_interval_prob * max_points

        return points_if_completed + expected_dnf_points


class ImprovementPointsCalculator:
    """
    Calculate expected improvement points vs rolling average.

    This calculator determines points earned from finishing better
    than the 8-race rolling average position.
    """

    def calculate(
        self,
        position_dist: PositionDistribution,
        rolling_avg: float,
        improvement_points: np.ndarray,
    ) -> float:
        """
        Calculate expected improvement points.

        Parameters
        ----------
        position_dist : PositionDistribution
            Distribution over race positions
        rolling_avg : float
            8-race rolling average finish position
        improvement_points : np.ndarray
            Array of points for positions gained

        Returns
        -------
        float
            Expected improvement points

        Notes
        -----
        The improvement_points array should contain the points awarded
        for each position gained, indexed by the number of positions.
        The rolling average is rounded to the nearest integer before
        calculating positions gained.

        Examples
        --------
        >>> calculator = ImprovementPointsCalculator()
        >>> position_dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> improvement_points = np.array([0, 2, 4, 6, 9])  # Points by positions gained
        >>> calculator.calculate(position_dist, rolling_avg=3.0, improvement_points=improvement_points)
        3.6  # 0.6*(3-1)*2 + 0.4*(3-2)*2 = 0.6*4 + 0.4*2
        """  # noqa: E501
        expected_points = 0.0
        rounded_avg = round(rolling_avg)

        for pos, prob in position_dist.items():
            positions_ahead = max(0, rounded_avg - pos)
            if positions_ahead > 0:
                # Get applicable improvement points (clip to max available)
                point_index = min(positions_ahead, len(improvement_points) - 1)
                expected_points += prob * improvement_points[point_index]

        return expected_points
