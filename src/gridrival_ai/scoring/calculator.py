"""
High-level calculator for F1 fantasy points.

This module provides a user-friendly interface for calculating F1 fantasy points
based on driver and constructor performance. It offers direct methods for common
scoring scenarios and integrates with probability distributions for expected
point calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from gridrival_ai.probabilities.core import JointDistribution, PositionDistribution
from gridrival_ai.scoring.config import ScoringConfig
from gridrival_ai.scoring.engine import ScoringEngine
from gridrival_ai.scoring.types import (
    ConstructorPositions,
    ConstructorWeekendData,
    DriverPointsBreakdown,
    DriverPositions,
    DriverWeekendData,
    RaceFormat,
)


@dataclass
class ScoringCalculator:
    """
    User-friendly calculator for F1 fantasy points.

    This class provides an intuitive interface for calculating fantasy points
    for drivers and constructors under various race scenarios. It supports both
    direct calculations with specific positions and expected value calculations
    with probability distributions.

    Parameters
    ----------
    config : ScoringConfig, optional
        Configuration for scoring rules, by default None (uses default config).

    Attributes
    ----------
    config : ScoringConfig
        Scoring configuration used for calculations.
    engine : ScoringEngine
        Low-level calculation engine for optimized computations.

    Examples
    --------
    >>> # Calculate points for a specific driver scenario
    >>> calculator = ScoringCalculator()
    >>> points = calculator.calculate_driver_points(
    ...     qualifying_pos=1,
    ...     race_pos=1,
    ...     rolling_avg=2.5,
    ...     teammate_pos=3,
    ...     completion_pct=1.0
    ... )
    >>> print(f"Total points: {points.total}")
    Total points: 166.0

    >>> # Calculate with probability distributions
    >>> from gridrival_ai.probabilities.core import PositionDistribution
    >>> qual_dist = PositionDistribution({1: 0.6, 2: 0.4})
    >>> race_dist = PositionDistribution({1: 0.7, 2: 0.3})
    >>> expected = calculator.expected_driver_points(
    ...     qual_dist=qual_dist,
    ...     race_dist=race_dist,
    ...     rolling_avg=2.5,
    ...     teammate_dist=PositionDistribution({3: 1.0})
    ... )
    >>> print(f"Expected points: {expected.total}")
    Expected points: 164.2
    """

    config: ScoringConfig = None
    engine: ScoringEngine = None

    def __post_init__(self) -> None:
        """Initialize with default config if none provided and set up engine."""
        if self.config is None:
            self.config = ScoringConfig.default()

        if self.engine is None:
            self.engine = ScoringEngine(self.config)

    def calculate_driver_points(
        self,
        qualifying_pos: int,
        race_pos: int,
        rolling_avg: float,
        teammate_pos: int,
        completion_pct: float = 1.0,
        sprint_pos: Optional[int] = None,
        race_format: RaceFormat = RaceFormat.STANDARD,
    ) -> DriverPointsBreakdown:
        """
        Calculate points for a driver with given positions.

        Parameters
        ----------
        qualifying_pos : int
            Qualifying position (1-20)
        race_pos : int
            Race finish position (1-20)
        rolling_avg : float
            8-race rolling average finish position
        teammate_pos : int
            Teammate's race finish position (1-20)
        completion_pct : float, optional
            Race completion percentage (0.0 to 1.0), by default 1.0
        sprint_pos : Optional[int], optional
            Sprint race position (1-8) for sprint weekends, by default None
        race_format : RaceFormat, optional
            Race weekend format, by default RaceFormat.STANDARD

        Returns
        -------
        DriverPointsBreakdown
            Detailed breakdown of points by component

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> points = calculator.calculate_driver_points(
        ...     qualifying_pos=1,
        ...     race_pos=2,
        ...     rolling_avg=4.0,
        ...     teammate_pos=5,
        ...     completion_pct=1.0,
        ...     race_format=RaceFormat.STANDARD
        ... )
        >>> points.total
        162.0
        >>> points.qualifying
        50.0
        >>> points.race
        97.0
        """
        # Create DriverWeekendData with positions
        positions = DriverPositions(
            qualifying=qualifying_pos,
            race=race_pos,
            sprint_finish=sprint_pos,
        )

        data = DriverWeekendData(
            format=race_format,
            positions=positions,
            completion_percentage=completion_pct,
            rolling_average=rolling_avg,
            teammate_position=teammate_pos,
        )

        # Use engine to calculate points
        points_dict = self._calculate_driver_components(data)

        # Convert to breakdown object
        return self._create_driver_breakdown(points_dict)

    def calculate_constructor_points(
        self,
        driver1_qualifying: int,
        driver1_race: int,
        driver2_qualifying: int,
        driver2_race: int,
        race_format: RaceFormat = RaceFormat.STANDARD,
    ) -> Dict[str, float]:
        """
        Calculate points for a constructor with given positions.

        Parameters
        ----------
        driver1_qualifying : int
            First driver qualifying position (1-20)
        driver1_race : int
            First driver race position (1-20)
        driver2_qualifying : int
            Second driver qualifying position (1-20)
        driver2_race : int
            Second driver race position (1-20)
        race_format : RaceFormat, optional
            Type of race weekend, by default RaceFormat.STANDARD

        Returns
        -------
        Dict[str, float]
            Points breakdown by component

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> points = calculator.calculate_constructor_points(
        ...     driver1_qualifying=1,
        ...     driver1_race=1,
        ...     driver2_qualifying=4,
        ...     driver2_race=3
        ... )
        >>> points["qualifying"]
        57.0
        >>> points["race"]
        157.0
        >>> sum(points.values())
        214.0
        """
        # Create ConstructorWeekendData with positions
        positions = ConstructorPositions(
            driver1_qualifying=driver1_qualifying,
            driver1_race=driver1_race,
            driver2_qualifying=driver2_qualifying,
            driver2_race=driver2_race,
        )

        data = ConstructorWeekendData(
            format=race_format,
            positions=positions,
        )

        # Use engine to calculate points
        return self._calculate_constructor_components(data)

    def expected_driver_points(
        self,
        qual_dist: PositionDistribution,
        race_dist: PositionDistribution,
        rolling_avg: float,
        teammate_dist: PositionDistribution,
        completion_prob: float = 0.95,
        sprint_dist: Optional[PositionDistribution] = None,
        race_format: RaceFormat = RaceFormat.STANDARD,
        joint_qual_race: Optional[JointDistribution] = None,
    ) -> DriverPointsBreakdown:
        """
        Calculate expected points breakdown from probability distributions.

        Parameters
        ----------
        qual_dist : PositionDistribution
            Distribution over qualifying positions
        race_dist : PositionDistribution
            Distribution over race positions
        rolling_avg : float
            8-race rolling average finish position
        teammate_dist : PositionDistribution
            Distribution over teammate's race positions
        completion_prob : float, optional
            Probability of completing the race, by default 0.95
        sprint_dist : Optional[PositionDistribution], optional
            Distribution over sprint positions, by default None
        race_format : RaceFormat, optional
            Race weekend format, by default RaceFormat.STANDARD
        joint_qual_race : Optional[JointDistribution], optional
            Joint distribution between qualifying and race positions.
            If provided, this will be used for overtake points calculations.
            If None, assumes independence between qual_dist and race_dist.

        Returns
        -------
        DriverPointsBreakdown
            Detailed breakdown of expected points by component

        Examples
        --------
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> calculator = ScoringCalculator()
        >>> # Create position distributions
        >>> qual_dist = PositionDistribution({1: 0.7, 2: 0.3})
        >>> race_dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> teammate_dist = PositionDistribution({3: 0.8, 4: 0.2})
        >>> # Calculate expected points
        >>> points = calculator.expected_driver_points(
        ...     qual_dist=qual_dist,
        ...     race_dist=race_dist,
        ...     rolling_avg=3.0,
        ...     teammate_dist=teammate_dist
        ... )
        >>> points.total  # Expected total points
        164.5
        """
        # Initialize component points
        points = {}

        # Calculate qualifying points
        points["qualifying"] = self._expected_position_points(
            qual_dist, self.config.qualifying_points
        )

        # Calculate race points
        points["race"] = self._expected_position_points(
            race_dist, self.config.race_points
        )

        # Calculate sprint points if applicable
        if race_format == RaceFormat.SPRINT and sprint_dist:
            points["sprint"] = self._expected_position_points(
                sprint_dist, self.config.sprint_points
            )

        # Calculate overtake points
        if joint_qual_race:
            # Use joint distribution for overtakes if provided
            points["overtake"] = self._expected_overtake_points(joint_qual_race)
        else:
            # Otherwise create independent joint distribution
            points["overtake"] = self._expected_overtake_points_from_marginals(
                qual_dist, race_dist
            )

        # Calculate improvement points
        points["improvement"] = self._expected_improvement_points(
            race_dist, rolling_avg
        )

        # Calculate teammate points
        points["teammate"] = self._expected_teammate_points(race_dist, teammate_dist)

        # Calculate completion points
        points["completion"] = self._expected_completion_points(completion_prob)

        # Create breakdown object
        return self._create_driver_breakdown(points)

    def expected_constructor_points(
        self,
        driver1_qual_dist: PositionDistribution,
        driver1_race_dist: PositionDistribution,
        driver2_qual_dist: PositionDistribution,
        driver2_race_dist: PositionDistribution,
        race_format: RaceFormat = RaceFormat.STANDARD,
    ) -> Dict[str, float]:
        """
        Calculate expected constructor points from probability distributions.

        Parameters
        ----------
        driver1_qual_dist : PositionDistribution
            First driver's qualifying position distribution
        driver1_race_dist : PositionDistribution
            First driver's race position distribution
        driver2_qual_dist : PositionDistribution
            Second driver's qualifying position distribution
        driver2_race_dist : PositionDistribution
            Second driver's race position distribution
        race_format : RaceFormat, optional
            Race weekend format, by default RaceFormat.STANDARD

        Returns
        -------
        Dict[str, float]
            Expected points breakdown by component

        Examples
        --------
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> calculator = ScoringCalculator()
        >>> # Create position distributions
        >>> driver1_qual = PositionDistribution({1: 0.7, 2: 0.3})
        >>> driver1_race = PositionDistribution({1: 0.6, 2: 0.4})
        >>> driver2_qual = PositionDistribution({3: 0.6, 4: 0.4})
        >>> driver2_race = PositionDistribution({3: 0.5, 4: 0.5})
        >>> # Calculate expected constructor points
        >>> points = calculator.expected_constructor_points(
        ...     driver1_qual_dist=driver1_qual,
        ...     driver1_race_dist=driver1_race,
        ...     driver2_qual_dist=driver2_qual,
        ...     driver2_race_dist=driver2_race
        ... )
        >>> points["qualifying"]  # Expected qualifying points
        56.8
        >>> points["race"]  # Expected race points
        152.6
        """
        # Initialize components
        result = {"qualifying": 0.0, "race": 0.0}

        # Calculate qualifying points for both drivers
        result["qualifying"] += self._expected_position_points(
            driver1_qual_dist, self.config.constructor_qualifying_points
        )

        result["qualifying"] += self._expected_position_points(
            driver2_qual_dist, self.config.constructor_qualifying_points
        )

        # Calculate race points for both drivers
        result["race"] += self._expected_position_points(
            driver1_race_dist, self.config.constructor_race_points
        )

        result["race"] += self._expected_position_points(
            driver2_race_dist, self.config.constructor_race_points
        )

        return result

    def simulate_driver_points(
        self,
        qual_dist: PositionDistribution,
        race_dist: PositionDistribution,
        rolling_avg: float,
        teammate_dist: PositionDistribution,
        n_samples: int = 1000,
        race_format: RaceFormat = RaceFormat.STANDARD,
        sprint_dist: Optional[PositionDistribution] = None,
        completion_prob: float = 0.95,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate driver points using Monte Carlo sampling.

        Parameters
        ----------
        qual_dist : PositionDistribution
            Distribution over qualifying positions
        race_dist : PositionDistribution
            Distribution over race positions
        rolling_avg : float
            8-race rolling average finish position
        teammate_dist : PositionDistribution
            Distribution over teammate's race positions
        n_samples : int, optional
            Number of simulations to run, by default 1000
        race_format : RaceFormat, optional
            Race weekend format, by default RaceFormat.STANDARD
        sprint_dist : Optional[PositionDistribution], optional
            Distribution over sprint positions, by default None
        completion_prob : float, optional
            Probability of completing the race, by default 0.95
        random_seed : Optional[int], optional
            Random seed for reproducibility, by default None

        Returns
        -------
        np.ndarray
            Array of simulated total points for each simulation

        Examples
        --------
        >>> import numpy as np
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> calculator = ScoringCalculator()
        >>> qual_dist = PositionDistribution({1: 0.7, 2: 0.3})
        >>> race_dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> teammate_dist = PositionDistribution({3: 0.8, 4: 0.2})
        >>> # Run 1000 simulations
        >>> simulations = calculator.simulate_driver_points(
        ...     qual_dist=qual_dist,
        ...     race_dist=race_dist,
        ...     rolling_avg=3.0,
        ...     teammate_dist=teammate_dist,
        ...     n_samples=1000,
        ...     random_seed=42
        ... )
        >>> np.mean(simulations)  # Average points
        164.5
        >>> np.std(simulations)  # Standard deviation
        2.3
        >>> np.percentile(simulations, 10)  # 10th percentile
        162.0
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize arrays to store results
        results = np.zeros(n_samples)

        # Run simulations
        for i in range(n_samples):
            # Sample positions from distributions
            qual_pos = self._sample_position(qual_dist)
            race_pos = self._sample_position(race_dist)
            teammate_pos = self._sample_position(teammate_dist)

            # Sample sprint position if applicable
            sprint_pos = None
            if race_format == RaceFormat.SPRINT and sprint_dist:
                sprint_pos = self._sample_position(sprint_dist)

            # Sample completion (bernoulli trial)
            completion = 1.0 if np.random.random() < completion_prob else 0.0

            # Calculate points for this simulation
            points = self.calculate_driver_points(
                qualifying_pos=qual_pos,
                race_pos=race_pos,
                rolling_avg=rolling_avg,
                teammate_pos=teammate_pos,
                completion_pct=completion,
                sprint_pos=sprint_pos,
                race_format=race_format,
            )

            # Store total points
            results[i] = points.total

        return results

    def calculate_qualifying_points(self, position: int) -> float:
        """
        Calculate qualifying points for position.

        Parameters
        ----------
        position : int
            Qualifying position (1-20)

        Returns
        -------
        float
            Points for the position

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> calculator.calculate_qualifying_points(1)
        50.0
        >>> calculator.calculate_qualifying_points(10)
        32.0
        """
        return self.config.qualifying_points.get(position, 0.0)

    def calculate_race_points(self, position: int) -> float:
        """
        Calculate race points for position.

        Parameters
        ----------
        position : int
            Race position (1-20)

        Returns
        -------
        float
            Points for the position

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> calculator.calculate_race_points(1)
        100.0
        >>> calculator.calculate_race_points(10)
        73.0
        """
        return self.config.race_points.get(position, 0.0)

    def calculate_sprint_points(self, position: int) -> float:
        """
        Calculate sprint points for position.

        Parameters
        ----------
        position : int
            Sprint position (1-8)

        Returns
        -------
        float
            Points for the position (0.0 if position > 8)

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> calculator.calculate_sprint_points(1)
        8.0
        >>> calculator.calculate_sprint_points(9)  # No points beyond P8
        0.0
        """
        return self.config.sprint_points.get(position, 0.0)

    def calculate_overtake_points(self, qualifying_pos: int, race_pos: int) -> float:
        """
        Calculate overtake points.

        Parameters
        ----------
        qualifying_pos : int
            Qualifying position (1-20)
        race_pos : int
            Race position (1-20)

        Returns
        -------
        float
            Overtake points (0.0 if no positions gained)

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> calculator.calculate_overtake_points(10, 5)  # Gained 5 positions
        15.0
        >>> calculator.calculate_overtake_points(5, 10)  # Lost positions
        0.0
        """
        positions_gained = max(0, qualifying_pos - race_pos)
        return positions_gained * self.config.overtake_multiplier

    def calculate_improvement_points(self, race_pos: int, rolling_avg: float) -> float:
        """
        Calculate improvement points vs rolling average.

        Parameters
        ----------
        race_pos : int
            Race position (1-20)
        rolling_avg : float
            8-race rolling average position

        Returns
        -------
        float
            Improvement points (0.0 if no improvement)

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> calculator.calculate_improvement_points(1, 3.0)  # 2 positions ahead
        4.0
        >>> calculator.calculate_improvement_points(5, 3.0)  # Worse than average
        0.0
        """
        positions_ahead = max(0, round(rolling_avg) - race_pos)
        if positions_ahead <= 0:
            return 0.0

        # Find applicable improvement points
        return self.config.improvement_points.get(
            positions_ahead,
            self.config.improvement_points.get(
                max(self.config.improvement_points), 0.0
            ),
        )

    def calculate_teammate_points(self, driver_pos: int, teammate_pos: int) -> float:
        """
        Calculate points for beating teammate.

        Parameters
        ----------
        driver_pos : int
            Driver's race position (1-20)
        teammate_pos : int
            Teammate's race position (1-20)

        Returns
        -------
        float
            Teammate points (0.0 if behind teammate)

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> calculator.calculate_teammate_points(1, 5)  # 4 positions ahead
        5.0
        >>> calculator.calculate_teammate_points(10, 5)  # Behind teammate
        0.0
        """
        if driver_pos >= teammate_pos:
            return 0.0

        margin = teammate_pos - driver_pos

        # Find applicable threshold
        thresholds = sorted(self.config.teammate_points.items())
        for threshold, points in thresholds:
            if margin <= threshold:
                return points

        # If beyond all thresholds, return points for maximum threshold
        if thresholds:
            return thresholds[-1][1]

        return 0.0

    def calculate_completion_points(self, completion_pct: float) -> float:
        """
        Calculate completion stage points.

        Parameters
        ----------
        completion_pct : float
            Race completion percentage (0.0 to 1.0)

        Returns
        -------
        float
            Completion points

        Examples
        --------
        >>> calculator = ScoringCalculator()
        >>> calculator.calculate_completion_points(1.0)  # Full race
        12.0
        >>> calculator.calculate_completion_points(0.6)  # 60% completion
        6.0
        >>> calculator.calculate_completion_points(0.0)  # No completion
        0.0
        """
        # Count stages completed
        stages_completed = 0
        for threshold in sorted(self.config.completion_thresholds):
            if completion_pct >= threshold:
                stages_completed += 1
            else:
                break

        return stages_completed * self.config.completion_stage_points

    # Private helper methods
    def _calculate_driver_components(self, data: DriverWeekendData) -> Dict[str, float]:
        """Calculate points for each component of driver scoring."""
        # Initialize components dictionary
        components = {}

        # Base points
        components["qualifying"] = self.calculate_qualifying_points(
            data.positions.qualifying
        )

        components["race"] = self.calculate_race_points(data.positions.race)

        # Sprint points if applicable
        if data.format == RaceFormat.SPRINT and data.positions.sprint_finish:
            components["sprint"] = self.calculate_sprint_points(
                data.positions.sprint_finish
            )

        # Overtake points
        components["overtake"] = self.calculate_overtake_points(
            data.positions.qualifying, data.positions.race
        )

        # Improvement points
        components["improvement"] = self.calculate_improvement_points(
            data.positions.race, data.rolling_average
        )

        # Teammate points
        components["teammate"] = self.calculate_teammate_points(
            data.positions.race, data.teammate_position
        )

        # Completion points
        components["completion"] = self.calculate_completion_points(
            data.completion_percentage
        )

        return components

    def _calculate_constructor_components(
        self, data: ConstructorWeekendData
    ) -> Dict[str, float]:
        """Calculate points for each component of constructor scoring."""
        # Initialize components dictionary
        components = {
            "qualifying": 0.0,
            "race": 0.0,
        }

        # Calculate qualifying points for both drivers
        components["qualifying"] += self.config.constructor_qualifying_points.get(
            data.positions.driver1_qualifying, 0.0
        )
        components["qualifying"] += self.config.constructor_qualifying_points.get(
            data.positions.driver2_qualifying, 0.0
        )

        # Calculate race points for both drivers
        components["race"] += self.config.constructor_race_points.get(
            data.positions.driver1_race, 0.0
        )
        components["race"] += self.config.constructor_race_points.get(
            data.positions.driver2_race, 0.0
        )

        return components

    def _create_driver_breakdown(
        self, components: Dict[str, float]
    ) -> DriverPointsBreakdown:
        """Create a DriverPointsBreakdown from component dictionary."""
        return DriverPointsBreakdown(
            qualifying=components.get("qualifying", 0.0),
            race=components.get("race", 0.0),
            sprint=components.get("sprint", 0.0),
            overtake=components.get("overtake", 0.0),
            improvement=components.get("improvement", 0.0),
            teammate=components.get("teammate", 0.0),
            completion=components.get("completion", 0.0),
        )

    def _expected_position_points(
        self, dist: PositionDistribution, points_mapping: Dict[int, float]
    ) -> float:
        """Calculate expected points for a position distribution."""
        return sum(prob * points_mapping.get(pos, 0.0) for pos, prob in dist.items())

    def _expected_overtake_points(self, joint_dist: JointDistribution) -> float:
        """Calculate expected overtake points from joint distribution."""
        expected_points = 0.0

        for (qual_pos, race_pos), prob in joint_dist.items():
            # Only count positions gained (not lost)
            positions_gained = max(0, qual_pos - race_pos)
            expected_points += prob * positions_gained * self.config.overtake_multiplier

        return expected_points

    def _expected_overtake_points_from_marginals(
        self, qual_dist: PositionDistribution, race_dist: PositionDistribution
    ) -> float:
        """Calculate expected overtake points assuming independence."""
        expected_points = 0.0

        # Assuming independence between qualifying and race positions
        for qual_pos, qual_prob in qual_dist.items():
            for race_pos, race_prob in race_dist.items():
                # Calculate joint probability assuming independence
                joint_prob = qual_prob * race_prob

                # Only count positions gained (not lost)
                positions_gained = max(0, qual_pos - race_pos)
                expected_points += (
                    joint_prob * positions_gained * self.config.overtake_multiplier
                )

        return expected_points

    def _expected_improvement_points(
        self, race_dist: PositionDistribution, rolling_avg: float
    ) -> float:
        """Calculate expected improvement points vs rolling average."""
        rounded_avg = round(rolling_avg)
        expected_points = 0.0

        for pos, prob in race_dist.items():
            positions_ahead = max(0, rounded_avg - pos)
            if positions_ahead > 0:
                # Get applicable improvement points or max if beyond table
                improvement_points = self.config.improvement_points.get(
                    positions_ahead,
                    self.config.improvement_points.get(
                        max(self.config.improvement_points), 0.0
                    ),
                )
                expected_points += prob * improvement_points

        return expected_points

    def _expected_teammate_points(
        self, driver_dist: PositionDistribution, teammate_dist: PositionDistribution
    ) -> float:
        """Calculate expected teammate points considering all combinations."""
        expected_points = 0.0

        # Iterate over all possible driver and teammate positions
        for driver_pos, driver_prob in driver_dist.items():
            for tm_pos, tm_prob in teammate_dist.items():
                # Only award points if driver is ahead of teammate
                if driver_pos < tm_pos:
                    margin = tm_pos - driver_pos

                    # Find applicable threshold
                    thresholds = sorted(self.config.teammate_points.items())
                    points = 0.0

                    for threshold, threshold_points in thresholds:
                        if margin <= threshold:
                            points = threshold_points
                            break

                    # If beyond all thresholds, use max threshold
                    if not points and thresholds:
                        points = thresholds[-1][1]

                    expected_points += driver_prob * tm_prob * points

        return expected_points

    def _expected_completion_points(self, completion_prob: float) -> float:
        """Calculate expected completion points based on completion probability."""
        # Full completion points (all stages)
        max_points = (
            len(self.config.completion_thresholds) * self.config.completion_stage_points
        )

        # Points for completing the race
        points_if_completed = completion_prob * max_points

        # DNF probability
        dnf_prob = 1.0 - completion_prob
        if dnf_prob <= 0.0:
            return max_points

        # Expected DNF points assuming uniform distribution of DNFs across race distance
        expected_dnf_points = 0.0
        prev_threshold = 0.0

        # Calculate points for each stage
        for i, threshold in enumerate(sorted(self.config.completion_thresholds)):
            # Probability of DNF in this interval (relative to overall DNF prob)
            interval_prob = (threshold - prev_threshold) / 1.0 * dnf_prob

            # Points for completing all previous stages
            points = i * self.config.completion_stage_points
            expected_dnf_points += interval_prob * points

            prev_threshold = threshold

        # DNF after last threshold but before finish
        last_interval_prob = (1.0 - prev_threshold) / 1.0 * dnf_prob
        expected_dnf_points += last_interval_prob * max_points

        return points_if_completed + expected_dnf_points

    def _sample_position(self, dist: PositionDistribution) -> int:
        """Sample a position from a position distribution."""
        positions = list(dist.position_probs.keys())
        probabilities = list(dist.position_probs.values())

        return int(np.random.choice(positions, p=probabilities))
